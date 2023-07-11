from .processor import Processor, Transformer
from .data import BeamData
from .path import beam_key
import pandas as pd

import json
import pathlib
import numpy as np

from functools import partial
from .utils import get_edit_ratio, get_edit_distance, is_notebook, BeamURL, normalize_host
from sqlalchemy.engine import create_engine
import openai
import re
from typing import Any, List, Mapping, Optional, Dict

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic import BaseModel, Field, PrivateAttr
from transformers.pipelines import Conversation
import transformers
from .utils import beam_device, BeamURL
import torch


class BeamSQL(Processor):

    # Build a beam class that provides an abstraction layer to different databases and lets us develop our tools without committing to a database technology.
    #
    # The class will be based on sqlalchemy+pandas but it can be inherited by subclasses that use 3rd party packages such as pyathena.
    #
    # some key features:
    # 1. the interface will be based on url addresses as in the BeamPath class
    # 2. two levels will be supported, db level where each index is a table and table level where each index is a column.
    # 3. minimizing the use of schemas and inferring the schemas from existing pandas dataframes and much as possible
    # 4. adding pandas like api whenever possible, for example, selecting columns with __getitem__, uploading columns and tables with __setitem__, loc, iloc
    # 5. the use of sqlalchemy and direct raw sql queries will be allowed.

    def __init__(self, *args, llm=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._connection = None
        self._engine = None
        self._table = None
        self._database = None
        self._index = None
        self._columns = None
        self._llm = llm

    @property
    def llm(self):
        return self._llm

    @property
    def database(self):
        return self._database

    @property
    def table(self):
        return self._table

    @property
    def index(self):
        return self._index

    @property
    def columns(self):
        return self._columns

    def set_database(self, database):
        self._database = database

    def set_llm(self, llm):
        self._llm = llm

    def set_index(self, index):
        self._index = index

    def set_columns(self, columns):
        self._columns = columns

    def set_table(self, table):
        self._table = table

    def __getitem__(self, item):

        if not isinstance(item, tuple):
            item = (item,)

        if self.table is None:
            axes = ['table', 'index', 'columns']
        else:
            axes = ['index', 'columns']

        for i, ind_i in enumerate(item):
            a = axes.pop(0)
            if a == 'table':
                self.set_table(ind_i)
            elif a == 'index':
                self.set_index(ind_i)
            elif a == 'columns':
                self.set_columns(ind_i)

        return self

    def sql(self, query, **kwargs):
        return pd.read_sql(query, self._connection, **kwargs)

    def get_sample(self, n=1, **kwargs):
        raise NotImplementedError

    def get_schema(self):
        raise NotImplementedError

    def nlp(self, query, **kwargs):

        schema = self.get_schema()

        prompt = f"Task: generate an SQL query that best describes the following text:\n {query}\n\n" \
                 f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" \
                 f"Additional instructions:\n\n" \
                 f"1. The queried table name is: {self.database}.{self.table}\n" \
                 f"2. Assume that the schema for the queried table is:\n{schema}\n\n" \
                 f"3. Here are 4 example rows {self.get_sample(n=4)}\n\n" \
                 f"4. In your response use only valid column names that best match the text\n\n" \
                 f"5. Important: your response must contain only the SQL query and nothing else, and it must be valid.\n\n" \
                 f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n" \
                 f"Response: \"\"\"\n{{text input here}}\n\"\"\""

        response = self.llm.ask(prompt, **kwargs)

        query = response.choices[0].text
        query = re.sub(r'\"\"\"', '', query)

        return self.sql(query)

    @staticmethod
    def df2table(df, name, metadata=None):

        from sqlalchemy import Table, Column, String, Integer, Float, Boolean, DateTime, Date, Time, BigInteger
        from sqlalchemy.schema import MetaData

        if metadata is None:
            metadata = MetaData()

        # Define the SQLAlchemy table object based on the DataFrame
        columns = [column for column in df.columns]
        types = {column: df.dtypes[column].name for column in df.columns}
        table = Table(name, metadata, *(Column(column, types[column]) for column in columns))

        return table

    @property
    def engine(self):
        raise NotImplementedError

    def __enter__(self):
        self._connection = self.engine.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._connection.close()
        self._connection = None


class BeamAthena(BeamSQL):
    def __init__(self, s3_staging_dir, role_session_name=None, region_name=None, access_key=None, secret_key=None,
                 *args, **kwargs):

        self.access_key = beam_key('aws_access_key', access_key)
        self.secret_key = beam_key('aws_secret_key', secret_key)
        self.s3_staging_dir = s3_staging_dir

        if role_session_name is None:
            role_session_name = "PyAthena-session"
        self.role_session_name = role_session_name

        if region_name is None:
            region_name = "eu-north-1"
        self.region_name = region_name

        state = {'s3_staging_dir': self.s3_staging_dir, 'role_session_name': self.role_session_name,
                      'region_name': self.region_name, 'access_key': self.access_key, 'secret_key': self.secret_key}

        super().__init__(*args, state=state, **kwargs)

    def get_sample(self, n=1, **kwargs):

        from pyathena.pandas.util import as_pandas

        query = f"SELECT * FROM {self.database}.{self.table} LIMIT {n}"

        cursor = self.connection.cursor()
        cursor.execute(query)
        df = as_pandas(cursor)

        return df

    def get_schema(self):

        query = f"DESCRIBE {self.database}.{self.table}"

        cursor = self.connection.cursor()
        cursor.execute(query)
        # Fetch the result
        result = cursor.fetchall()

        return result

    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_engine('athena+pyathena://', creator=lambda: self.connection)
        return self._engine

    @property
    def connection(self):

        if self._connection is None:

            from pyathena import connect

            self._connection = connect(s3_staging_dir=self.s3_staging_dir,
                                       role_session_name=self.role_session_name,
                                       region_name=self.region_name, aws_access_key_id=self.access_key,
                                       aws_secret_access_key=self.secret_key)

        return self._connection

    def sql(self, query):

        from pyathena.pandas.util import as_pandas

        cursor = self.connection.cursor()
        cursor.execute(query)
        df = as_pandas(cursor)
        bd = BeamData(df)

        return bd


class LLMResponse:
    def __init__(self, response, llm):
        self.response = response
        self.llm = llm
    @property
    def text(self):
        return self.llm.extract_text(self.response)

    # @property
    # def choices(self):
    #     return self.llm.extract_choices(self.response)


class BeamLLM(LLM, Processor):

    model: Optional[str] = Field(None)
    scheme: Optional[str] = Field(None)
    usage: Any
    instruction_history: Any
    _chat_history: Any = PrivateAttr()
    _url: Any = PrivateAttr()
    temperature: float = Field(1.0, ge=0.0, le=1.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    n: int = Field(1, ge=1)
    stream: bool = Field(False)
    stop: Optional[str] = Field(None)
    max_tokens: Optional[int] = Field(None)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = Field(None)

    def __init__(self, *args, temperature=1, top_p=1, n=1, stream=False, stop=None, max_tokens=None, presence_penalty=0,
                 frequency_penalty=0.0, logit_bias=None, scheme='unknown', **kwargs):
        super().__init__(*args, **kwargs)

        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stream = stream
        self.stop = stop
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.scheme = scheme
        self._url = None
        self.instruction_history = []

        if not hasattr(self, 'model'):
            self.model = None
        self.usage = {"prompt_tokens": 0,
                      "completion_tokens": 0,
                      "total_tokens": 0}

        self._chat_history = None
        self.reset_chat()

    @property
    def url(self):

        if self._url is None:
            self._url = BeamURL(scheme=self.scheme, path=self.model)

        return str(self._url)

    @property
    def conversation(self):
        return self._chat_history

    def reset_chat(self):
        self._chat_history = Conversation()

    @property
    def chat_history(self):
        ch = list(self._chat_history.iter_texts())
        return [{'role': 'user' if m[0] else 'assistant', 'content': m[1]} for m in ch]

    def add_to_chat(self, text, is_user=True):
        if is_user:
            self._chat_history.add_user_input(text)
        else:
            self._chat_history.append_response(text)
            self._chat_history.mark_processed()

    @property
    def _llm_type(self) -> str:
        return "beam_llm"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:

        res = self.ask(prompt, stop=stop).text
        return res

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"is_chat": self.is_chat,
                'usuage': self.usage}

    @property
    def is_chat(self):
        raise NotImplementedError

    @property
    def is_completions(self):
        return not self.is_chat

    def chat_completion(self, **kwargs):
        raise NotImplementedError

    def completion(self, **kwargs):
        raise NotImplementedError

    def update_usage(self, response):
        raise NotImplementedError

    def get_default_params(self, temperature=None,
             top_p=None, n=None, stream=None, stop=None, max_tokens=None, presence_penalty=None, frequency_penalty=None, logit_bias=None):

        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if n is None:
            n = self.n
        if stream is None:
            stream = self.stream
        if presence_penalty is None:
            presence_penalty = self.presence_penalty
        if frequency_penalty is None:
            frequency_penalty = self.frequency_penalty
        if logit_bias is None:
            logit_bias = self.logit_bias

        return {'temperature': temperature,
                'top_p': top_p,
                'n': n,
                'stream': stream,
                'stop': stop,
                'max_tokens': max_tokens,
                'presence_penalty': presence_penalty,
                'frequency_penalty': frequency_penalty,
                'logit_bias': logit_bias}

    def chat(self, message, name=None, system=None, system_name=None, reset_chat=False, temperature=None,
             top_p=None, n=None, stream=None, stop=None, max_tokens=None, presence_penalty=None, frequency_penalty=None,
             logit_bias=None, **kwargs):

        '''

        :param name:
        :param system:
        :param system_name:
        :param reset_chat:
        :param temperature:
        :param top_p:
        :param n:
        :param stream:
        :param stop:
        :param max_tokens:
        :param presence_penalty:
        :param frequency_penalty:
        :param logit_bias:
        :return:
        '''

        default_params = self.get_default_params(temperature=temperature,
                                                 top_p=top_p, n=n, stream=stream,
                                                 stop=stop, max_tokens=max_tokens,
                                                 presence_penalty=presence_penalty,
                                                 frequency_penalty=frequency_penalty,
                                                 logit_bias=logit_bias)

        if reset_chat:
            self.reset_chat()

        messages = []
        if system is not None:
            system = {'system': system}
            if system_name is not None:
                system['system_name'] = system_name
            messages.append(system)

        messages.extend(self.chat_history)

        self.add_to_chat(message, is_user=True)
        message = {'role': 'user', 'content': message}
        if name is not None:
            message['name'] = name

        messages.append(message)

        kwargs = default_params
        if logit_bias is not None:
            kwargs['logit_bias'] = logit_bias
        if max_tokens is not None:
            kwargs['max_tokens'] = max_tokens
        if stop is not None:
            kwargs['stop'] = stop

        response = self.chat_completion(messages=messages, **kwargs)

        self.update_usage(response)
        response = LLMResponse(response, self)

        self.add_to_chat(response.text, is_user=False)

        return response

    def docstring(self, text, element_type, name=None, docstring_format=None, parent=None, parent_name=None,
                  parent_type=None, children=None, children_type=None, children_name=None, **kwargs):

        if docstring_format is None:
            docstring_format = f"in \"{docstring_format}\" format, "
        else:
            docstring_format = ""

        prompt = f"Task: write a full python docstring {docstring_format}for the following {element_type}\n\n" \
                 f"========================================================================\n\n" \
                 f"{text}\n\n" \
                 f"========================================================================\n\n"

        if parent is not None:
            prompt = f"{prompt}" \
                     f"where its parent {parent_type}: {parent_name}, has the following docstring\n\n" \
                     f"========================================================================\n\n" \
                     f"{parent}\n\n" \
                     f"========================================================================\n\n"

        if children is not None:
            for i, (c, cn, ct) in enumerate(zip(children, children_name, children_type)):
                prompt = f"{prompt}" \
                         f"and its #{i} child: {ct} named {cn}, has the following docstring\n\n" \
                         f"========================================================================\n\n" \
                         f"{c}\n\n" \
                         f"========================================================================\n\n"

        prompt = f"{prompt}" \
                 f"Response: \"\"\"\n{{docstring text here (do not add anything else)}}\n\"\"\""

        if not self.is_completions:
            try:
                res = self.chat(prompt, **kwargs)
            except Exception as e:
                print(f"Error in response: {e}")
                try:
                    print(f"{name}: switching to gpt-4 model")
                    res = self.chat(prompt, model='gpt-4', **kwargs)
                except:
                    print(f"{name}: error in response")
                    res = None
        else:
            res = self.ask(prompt, **kwargs)

        # res = res.choices[0].text

        return res

    def ask(self, question, max_tokens=None, temperature=None, top_p=None, frequency_penalty=None,
            presence_penalty=None, stop=None, n=None, stream=None, logprobs=None, logit_bias=None, echo=False, **kwargs):
        """
        Ask a question to the model
        :param n:
        :param logprobs:
        :param stream:
        :param echo:
        :param question:
        :param max_tokens:
        :param temperature: 0.0 - 1.0
        :param top_p:
        :param frequency_penalty:
        :param presence_penalty:
        :param stop:
        :return:
        """

        default_params = self.get_default_params(temperature=temperature,
                                                 top_p=top_p, n=n, stream=stream,
                                                 stop=stop, max_tokens=max_tokens,
                                                 presence_penalty=presence_penalty,
                                                 frequency_penalty=frequency_penalty,
                                                 logit_bias=logit_bias)

        if not self.is_completions:
            kwargs = {**default_params, **kwargs}
            response = self.chat(question, reset_chat=True, **kwargs)
        else:
            response = self.completion(prompt=question, logprobs=logprobs, echo=echo, **default_params)

            self.update_usage(response)
            response = LLMResponse(response, self)

        self.instruction_history.append({'question': question, 'response': response.text, 'type': 'ask'})

        return response

    def reset_instruction_history(self):
        self.instruction_history = []

    def summary(self, text, n_words=100, n_paragraphs=None, **kwargs):
        """
        Summarize a text
        :param text:  text to summarize
        :param n_words: number of words to summarize the text into
        :param n_paragraphs:   number of paragraphs to summarize the text into
        :param kwargs: additional arguments for the ask function
        :return: summary
        """
        if n_paragraphs is None:
            prompt = f"Task: summarize the following text into {n_words} words\nText: {text}\nResponse: \"\"\"\n{{text input here}}\n\"\"\""
        else:
            prompt = f"Task: summarize the following text into {n_paragraphs} paragraphs\nText: {text}\nResponse: \"\"\"\n{{text input here}}\n\"\"\""

        res = self.ask(prompt, **kwargs).text
        return res

    def extract_text(self, res):
        raise NotImplementedError

    def question(self, text, question, **kwargs):
        """
        Answer a yes-no question
        :param text: text to answer the question from
        :param question: question to answer
        :param kwargs: additional arguments for the ask function
        :return: answer
        """
        prompt = f"Task: answer the following question\nText: {text}\nQuestion: {question}\nResponse: \"\"\"\n{{text input here}}\n\"\"\""

        res = self.ask(prompt, **kwargs).text
        return res

    def yes_or_no(self, question, text=None, **kwargs):
        """
        Answer a yes or no question
        :param text: text to answer the question from
        :param question:  question to answer
        :param kwargs: additional arguments for the ask function
        :return: answer
        """

        if text is None:
            preface = ''
        else:
            preface = f"Text: {text}\n"

        prompt = f"{preface}Task: answer the following question with yes or no\nQuestion: {question}\nResponse: \"\"\"\n{{text input here}}\n\"\"\""

        res = self.ask(prompt, **kwargs).text

        res = res.lower().strip()
        res = res.split(" ")[0]

        i = pd.Series(['no', 'yes']).apply(partial(get_edit_ratio, s2=res)).idxmax()
        # print(res)
        return bool(i)

    def quant_analysis(self, text, source=None, **kwargs):
        """
        Perform a quantitative analysis on a text
        :param text: text to perform the analysis on
        :param kwargs: additional arguments for the ask function
        :return: analysis
        """
        prompt = f"Task: here is an economic news article from {source}\nText: {text}\nResponse: \"\"\"\n{{text input here}}\n\"\"\""
        res = self.ask(prompt, **kwargs).text
        return res

    def names_of_people(self, text, **kwargs):
        """
        Extract names of people from a text
        :param text: text to extract names from
        :param kwargs: additional arguments for the ask function
        :return: list of names
        """
        prompt = f"Task: extract names of people from the following text, return in a list of comma separated values\nText: {text}\nResponse: \"\"\"\n{{text input here}}\n\"\"\""

        res = self.ask(prompt, **kwargs).text
        res = res.strip().split(",")

        return res

    def answer_email(self, input_email_thread, responder_from, receiver_to, **kwargs):
        """
        Answer a given email thread as an chosen entity
        :param input_email_thread_test: given email thread to answer to
        :param responder_from: chosen entity name which will answer the last mail from the thread
        :param receiver_to: chosen entity name which will receive the generated mail
        :param kwargs: additional arguments for the prompt
        :return: response mail
        """

        prompt = f"{input_email_thread}\n---generate message---\nFrom: {responder_from}To: {receiver_to}\n\n###\n\n"
        # prompt = f"Text: {text}\nTask: answer the following question with yes or no\nQuestion: {question}\nResponse: \"\"\"\n{{text input here}}\n\"\"\""

        res = self.ask(prompt, **kwargs).text
        return res

    def classify(self, text, classes, **kwargs):
        """
        Classify a text
        :param text: text to classify
        :param classes: list of classes
        :param kwargs: additional arguments for the ask function
        :return: class
        """
        prompt = f"Task: classify the following text into one of the following classes\nText: {text}\nClasses: {classes}\nResponse: \"\"\"\n{{text input here}}\n\"\"\""

        res = self.ask(prompt, **kwargs).text
        res = res.lower().strip()

        i = pd.Series(classes).str.lower().str.strip().apply(partial(get_edit_ratio, s2=res)).idxmax()

        return classes[i]

    def features(self, text, features=None, **kwargs):
        """
        Extract features from a text
        :param text: text to extract features from
        :param kwargs: additional arguments for the ask function
        :return: features
        """

        if features is None:
            features = []

        features = [f.lower().strip() for f in features]

        prompt = f"Task: Out of the following set of terms: {features}\n" \
                 f"list in comma separated values (csv) the terms that describe the following Text:\n" \
                 f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" \
                 f" {text}\n" \
                 f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" \
                 f"Important: do not list any other term that did not appear in the aforementioned list.\n" \
                 f"Response: \"\"\"\n{{text input here}}\n\"\"\""

        res = self.ask(prompt, **kwargs).text

        llm_features = res.split(',')
        llm_features = [f.lower().strip() for f in llm_features]
        features = [f for f in llm_features if f in features]

        return features

    def entities(self, text, humans=True, **kwargs):
        """
        Extract entities from a text
        :param humans:  if True, extract people, else extract all entities
        :param text: text to extract entities from
        :param kwargs: additional arguments for the ask function
        :return: entities
        """
        if humans:
            prompt = f"Task: extract people from the following text in a comma separated list\nText: {text}\nResponse: \"\"\"\n{{text input here}}\n\"\"\""
        else:
            prompt = f"Task: extract entities from the following text in a comma separated list\nText: {text}\nResponse: \"\"\"\n{{text input here}}\n\"\"\""

        res = self.ask(prompt, **kwargs).text

        entities = res.split(',')
        entities = [e.lower().strip() for e in entities]

        return entities

    def title(self, text, n_words=None, **kwargs):
        """
        Extract title from a text
        :param text: text to extract title from
        :param kwargs: additional arguments for the ask function
        :return: title
        """
        if n_words is None:
            prompt = f"Task: extract title from the following text\nText: {text}\nResponse: \"\"\"\n{{text input here}}\n\"\"\""
        else:
            prompt = f"Task: extract title from the following text. Restrict the answer to {n_words} words only." \
                     f"\nText: {text}\nResponse: \"\"\"\n{{text input here}}\n\"\"\""

        res = self.ask(prompt, **kwargs).text

        return res

    def similar_keywords(self, text, keywords, **kwargs):
        """
        Find similar keywords to a list of keywords
        :param text: text to find similar keywords from
        :param keywords: list of keywords
        :param kwargs: additional arguments for the ask function
        :return: list of similar keywords
        """

        keywords = [e.lower().strip() for e in keywords]
        prompt = f"Keywords: {keywords}\nTask: find similar keywords in the following text\nText: {text}\n\nResponse: \"\"\"\n{{text input here}}\n\"\"\""

        res = self.ask(prompt, **kwargs).text
        res = res.split(',')
        res = [e.lower().strip() for e in res]

        res = list(set(res) - set(keywords))

        return res

    def is_keyword_found(self, text, keywords, **kwargs):
        """
        chek if one or more key words found in given text
        :param text: text to looks for
        :param keywords:  key words list
        :param kwargs: additional arguments for the ask function
        :return: yes if one of the keywords found else no
        """
        prompt = f"Text: {text}\nTask: answer with yes or no if Text contains one of the keywords \nKeywords: {keywords}\nResponse: \"\"\"\n{{text input here}}\n\"\"\""

        res = self.ask(prompt, **kwargs).text
        res = res.lower().strip().replace('"', "")

        i = pd.Series(['no', 'yes']).apply(partial(get_edit_ratio, s2=res)).idxmax()
        return bool(i)

    def get_similar_terms(self, keywords, **kwargs):
        """
        chek if one or more key words found in given text
        :param keywords:  key words list
        :param kwargs: additional arguments for the ask function
        :return: similar terms
        """
        prompt = f"keywords: {keywords}\nTask: return all semantic terms for given Keywords \nResponse: \"\"\"\n{{text input here}}\n\"\"\""

        res = self.ask(prompt, **kwargs).text
        res = res.lower().strip()
        return res


class OpenAIBase(BeamLLM):

    api_key: Optional[str] = Field(None)
    api_base: Optional[str] = Field(None)
    organization: Optional[str] = Field(None)

    def __init__(self, api_key=None, api_base=None, organization=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.api_key = api_key
        self.api_base = api_base
        self.organization = organization

    def update_usage(self, response):

        if 'usage' in response:
            response = response['usage']

            self.usage["prompt_tokens"] += response["prompt_tokens"]
            self.usage["completion_tokens"] += response["completion_tokens"]
            self.usage["total_tokens"] += response["prompt_tokens"] + response["completion_tokens"]

    def sync_openai(self):
        openai.api_key = self.api_key
        openai.api_base = self.api_base
        openai.organization = self.organization

    def chat_completion(self, **kwargs):
        self.sync_openai()
        # todo: remove this when logit_bias is supported
        kwargs.pop('logit_bias')
        return openai.ChatCompletion.create(model=self.model, **kwargs)

    def completion(self, **kwargs):
        self.sync_openai()
        # todo: remove this when logit_bias is supported
        kwargs.pop('logit_bias')
        return openai.Completion.create(engine=self.model, **kwargs)

    def extract_text(self, res):
        if not self.is_chat:
            res = res.choices[0].text
        else:
            res = res.choices[0].message.content
        return res


class FastChatLLM(OpenAIBase):

    def __init__(self, model=None, hostname=None, port=None, *args, **kwargs):

        api_base = f"http://{normalize_host(hostname, port)}/v1"
        api_key = "EMPTY"  # Not support yet
        organization = "EMPTY"  # Not support yet

        kwargs['scheme'] = 'fastchat'
        super().__init__(api_key=api_key, api_base=api_base, organization=organization,
                         *args, **kwargs)

        self.model = model

        # if is_notebook():
        #     import nest_asyncio
        #     nest_asyncio.apply()

    @property
    def is_chat(self):
        return True


class HuggingFaceLLM(BeamLLM):
    config: Any
    tokenizer: Any
    model: Any
    pipline_kwargs: Any
    input_device: Optional[str] = Field(None)

    def __init__(self, model, tokenizer=None, dtype=None, chat=False, input_device=None, compile=True, *args,
                 model_kwargs=None,
                 config_kwargs=None, pipline_kwargs=None, **kwargs):

        kwargs['scheme'] = 'huggingface'
        super().__init__(*args, **kwargs)

        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

        transformers.logging.set_verbosity_error()

        if model_kwargs is None:
            model_kwargs = {}

        if config_kwargs is None:
            config_kwargs = {}

        if pipline_kwargs is None:
            pipline_kwargs = {}

        self.pipline_kwargs = pipline_kwargs

        self.input_device = input_device

        self.config = AutoConfig.from_pretrained(model, trust_remote_code=True, **config_kwargs)
        tokenizer_name = tokenizer or model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True,
                                                          config=self.config, **model_kwargs)
        if compile:
            self.model = torch.compile(self.model)

    def update_usage(self, response):
        pass

    def extract_text(self, res):
        if type(res) is list:
            res = res[0]

        if type(res) is Conversation:
            res = res.generated_responses[-1]
        else:
            res = res['generated_text']

        return res

    @property
    def is_chat(self):
        return True

    @property
    def is_completions(self):
        return True

    def completion(self, prompt=None, **kwargs):

        pipeline = transformers.pipeline('text-generation', model=self.model,
                                         tokenizer=self.tokenizer, device=self.input_device, return_full_text=False)

        return pipeline(prompt, pad_token_id=pipeline.tokenizer.eos_token_id, **self.pipline_kwargs)

    def chat_completion(self, **kwargs):

        pipeline = transformers.pipeline('conversational', model=self.model,
                                         tokenizer=self.tokenizer, device=self.input_device)

        return pipeline(self.conversation, pad_token_id=pipeline.tokenizer.eos_token_id, **self.pipline_kwargs)


class OpenAI(OpenAIBase):

    _models: Any = PrivateAttr()

    def __init__(self, model='gpt-3.5-turbo', api_key=None, organization=None, *args, **kwargs):

        api_key = beam_key('openai_api_key', api_key)

        kwargs['scheme'] = 'openai'
        super().__init__(api_key=api_key, api_base='https://api.openai.com/v1',
                         organization=organization, *args, **kwargs)

        self.model = model
        self._models = None

    @property
    def is_chat(self):
        chat_models = ['gpt-4', 'gpt-4-0314', 'gpt-4-32k', 'gpt-4-32k-0314', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301']
        if any([m in self.model for m in chat_models]):
            return True
        return False

    def file_list(self):
        return openai.File.list()

    def build_dataset(self, data=None, question=None, answer=None, path=None) -> object:
        """
        Build a dataset for training a model
        :param data: dataframe with prompt and completion columns
        :param question: list of questions
        :param answer: list of answers
        :param path: path to save the dataset
        :return: path to the dataset
        """
        if data is None:
            data = pd.DataFrame(data={'prompt': question, 'completion': answer})

        records = data.to_dict(orient='records')

        if path is None:
            print('No path provided, using default path: dataset.jsonl')
            path = 'dataset.jsonl'

        # Open a file for writing
        with open(path, 'w') as outfile:
            # Write each data item to the file as a separate line
            for item in records:
                json.dump(item, outfile)
                outfile.write('\n')

        return path

    def retrieve(self, model=None):
        if model is None:
            model = self.model
        return openai.Engine.retrieve(id=model)

    @property
    def models(self):
        if self._models is None:
            models = openai.Model.list()
            models = {m.id: m for m in models.data}
            self._models = models
        return self._models

    def embedding(self, text, model=None):
        if model is None:
            model = self.model
        response = openai.Engine(model).embedding(input=text, model=model)
        embedding = np.array(response.data[1]['embedding'])
        return embedding


def beam_llm(url, username=None, hostname=None, port=None, api_key=None, **kwargs):

    if type(url) != str:
        return url

    url = BeamURL.from_string(url)

    if url.hostname is not None:
        hostname = url.hostname

    if url.port is not None:
        port = url.port

    if url.username is not None:
        username = url.username

    query = url.query
    for k, v in query.items():
        kwargs[k] = v

    if api_key is None and 'api_key' in kwargs:
        api_key = kwargs.pop('api_key')

    model = url.path
    model = model.lstrip('/')
    if not model:
        model = None

    if url.protocol == 'openai':

        api_key = beam_key('openai_api_key', api_key)
        return OpenAI(model=model, api_key=api_key, **kwargs)

    elif url.protocol == 'fastchat':
        return FastChatLLM(model=model, hostname=hostname, port=port, **kwargs)

    elif url.protocol == 'huggingface':
        return HuggingFaceLLM(model=model, **kwargs)

    else:
        raise NotImplementedError
