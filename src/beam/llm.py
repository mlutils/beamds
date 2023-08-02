from .processor import Processor, Transformer
from .data import BeamData
from .path import beam_key
import pandas as pd

import json
import numpy as np

from functools import partial
from .utils import get_edit_ratio, get_edit_distance, is_notebook, BeamURL, normalize_host
import openai
from typing import Any, List, Mapping, Optional, Dict

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic import BaseModel, Field, PrivateAttr
from transformers.pipelines import Conversation
import transformers
from .utils import beam_device, BeamURL
import torch
import requests


class LLMResponse:
    def __init__(self, response, llm):
        self.response = response
        self.llm = llm
    @property
    def text(self):
        return self.llm.extract_text(self.response)

    @property
    def openai_format(self):
        return self.llm.openai_format(self.response)

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

    def __init__(self, *args, temperature=.1, top_p=1, n=1, stream=False, stop=None, max_tokens=None, presence_penalty=0,
                 frequency_penalty=0.0, logit_bias=None, scheme='unknown', model=None, **kwargs):
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

        self.model = model
        if self.model is None:
            self.model = 'unknown'

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

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:

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

    def _chat_completion(self, **kwargs):
        raise NotImplementedError

    def chat_completion(self, **kwargs):

        response = self._chat_completion(**kwargs)
        self.update_usage(response)
        response = LLMResponse(response, self)
        return response

    def completion(self, **kwargs):

        response = self._completion(**kwargs)
        self.update_usage(response)
        response = LLMResponse(response, self)
        return response

    def _completion(self, **kwargs):
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

    def openai_format(self, res):
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

    def _chat_completion(self, **kwargs):
        self.sync_openai()
        # todo: remove this when logit_bias is supported
        kwargs.pop('logit_bias')
        return openai.ChatCompletion.create(model=self.model, **kwargs)

    def _completion(self, **kwargs):
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

    def openai_format(self, res):
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


# class LocalFastChat(BeamLLM):
#
#     def __init__(self, model=None, hostname=None, port=None, *args, **kwargs):
#         kwargs['scheme'] = 'fastchat'
#         super().__init__(*args, **kwargs)
#         self.model = ModelWorker(controller_addr='NA',
#                             worker_addr='NA',
#                             worker_id='default',
#                             model_path=model,
#                             model_names=['my model'],
#                             # limit_worker_concurrency=1,
#                             no_register=True,
#                             device='cuda',
#                             num_gpus=1,
#                             max_gpu_memory=None,
#                             load_8bit=False,
#                             cpu_offloading=False,
#                             gptq_config=None,)
#                             # stream_interval=2)
#
#     @property
#     def is_chat(self):
#         return True
#
#     def chat(self, prompt, **kwargs):
#         return self.ask(prompt, **kwargs)


class FastAPILLM(BeamLLM):

    model: Optional[str] = Field(None)
    hostname: Optional[str] = Field(None)
    headers: Optional[dict] = Field(None)
    consumer: Optional[str] = Field(None)
    _models: Any = PrivateAttr()

    def __init__(self, *args, model=None, hostname=None, port=None, username=None, **kwargs):
        kwargs['scheme'] = 'fastapi'
        super().__init__(*args, **kwargs)
        self.model = model
        self.consumer = username
        self.hostname = normalize_host(hostname, port)
        self._models = None
        self.headers = {'Content-Type': 'application/json'}

    @property
    def models(self):
        if self._models is None:
            res = requests.get(f"http://{self.hostname}/models", headers=self.headers)
            self._models = res.json()
        return self._models

    @property
    def is_chat(self):
        return False

    def _chat_completion(self, **kwargs):
        raise NotImplementedError

    def _completion(self, **kwargs):

        d = {}
        d['model_name'] = self.model
        d['consumer'] = self.consumer
        d['input'] = kwargs.pop('prompt')
        d['hyper_params'] = kwargs

        res = requests.post(f"http://{self.hostname}/predict/loop", headers=self.headers, json=d)
        return res.json()

    def extract_text(self, res):
        if not self.is_chat:
            res = res['res']
        else:
            raise NotImplementedError
        return res

class HuggingFaceLLM(BeamLLM):
    config: Any
    tokenizer: Any
    net: Any
    pipline_kwargs: Any
    input_device: Optional[str] = Field(None)
    eos_pattern: Optional[str] = Field(None)
    _text_generation_pipeline: Any = PrivateAttr()
    _conversational_pipeline: Any = PrivateAttr()

    def __init__(self, model, tokenizer=None, dtype=None, chat=False, input_device=None, compile=True, *args,
                 model_kwargs=None,
                 config_kwargs=None, pipline_kwargs=None, text_generation_kwargs=None, conversational_kwargs=None,
                 eos_pattern=None, **kwargs):

        kwargs['scheme'] = 'huggingface'
        kwargs['model'] = model
        super().__init__(*args, **kwargs)

        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

        transformers.logging.set_verbosity_error()

        if model_kwargs is None:
            model_kwargs = {}

        if config_kwargs is None:
            config_kwargs = {}

        if pipline_kwargs is None:
            pipline_kwargs = {}

        if text_generation_kwargs is None:
            text_generation_kwargs = {}

        if conversational_kwargs is None:
            conversational_kwargs = {}

        self.pipline_kwargs = pipline_kwargs

        self.input_device = input_device
        self.eos_pattern = eos_pattern

        self.config = AutoConfig.from_pretrained(model, trust_remote_code=True, **config_kwargs)
        tokenizer_name = tokenizer or model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        self.net = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True,
                                                        config=self.config, **model_kwargs)

        if compile:
            self.net = torch.compile(self.net)

        self._text_generation_pipeline = transformers.pipeline('text-generation', model=self.net,
                                                               tokenizer=self.tokenizer, device=self.input_device,
                                                               return_full_text=False, **text_generation_kwargs)

        self._conversational_pipeline = transformers.pipeline('conversational', model=self.net,
                                                              tokenizer=self.tokenizer, device=self.input_device,
                                                              **conversational_kwargs)

    def update_usage(self, response):
        pass

    def extract_text(self, res):
        if type(res) is list:
            res = res[0]

        if type(res) is Conversation:
            res = res.generated_responses[-1]
        else:
            res = res['generated_text']

        if self.eos_pattern:
            res = res.split(self.eos_pattern)[0]

        return res

    @property
    def is_chat(self):
        return True

    @property
    def is_completions(self):
        return True

    def _completion(self, prompt=None, **kwargs):

        # pipeline = transformers.pipeline('text-generation', model=self.model,
        #                                  tokenizer=self.tokenizer, device=self.input_device, return_full_text=False)

        res = self._text_generation_pipeline(prompt, pad_token_id=self._text_generation_pipeline.tokenizer.eos_token_id,
                                             **self.pipline_kwargs)

        return res

    def _chat_completion(self, **kwargs):

        # pipeline = transformers.pipeline('conversational', model=self.model,
        #                                  tokenizer=self.tokenizer, device=self.input_device)

        return self._conversational_pipeline(self.conversation,
                                             pad_token_id=self._conversational_pipeline.tokenizer.eos_token_id,
                                             **self.pipline_kwargs)


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

    elif url.protocol == 'fastapi':
        return FastAPILLM(model=model, hostname=hostname, port=port, username=username, **kwargs)

    else:
        raise NotImplementedError


from argparse import Namespace
def simulate_openai_chat(model=None, **kwargs):
    llm = beam_llm(model) if type(model) == str else model
    return llm.chat_completion(**kwargs).openai_format
def simulate_openai_completion(model=None, **kwargs):
    llm = beam_llm(model) if type(model) == str else model
    return llm.completion(**kwargs).openai_format

openai_simulator = Namespace(ChatCompletion=Namespace(create=simulate_openai_chat),
                             Completion=Namespace(create=simulate_openai_completion))