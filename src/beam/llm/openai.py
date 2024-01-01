import json
from typing import Optional, Any
import pandas as pd
import numpy as np

from ..logger import beam_logger as logger
from .core import BeamLLM, CompletionObject
from pydantic import Field, PrivateAttr
from ..path import beam_key

from ..utils import lazy_property


class OpenAIBase(BeamLLM):

    api_key: Optional[str] = Field(None)
    api_base: Optional[str] = Field(None)
    organization: Optional[str] = Field(None)

    def __init__(self, api_key=None, api_base=None, organization=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.api_key = api_key
        self.api_base = api_base
        self.organization = organization

    @lazy_property
    def client(self):
        from openai import OpenAI
        return OpenAI(organization=self.organization, api_key=self.api_key, base_url=self.api_base)

    def update_usage(self, response):

        if 'usage' in response:
            response = response['usage']

            self.usage["prompt_tokens"] += response["prompt_tokens"]
            self.usage["completion_tokens"] += response["completion_tokens"]
            self.usage["total_tokens"] += response["prompt_tokens"] + response["completion_tokens"]

    # def sync_openai(self):
    #     openai.api_key = self.api_key
    #     openai.api_base = self.api_base
    #     openai.organization = self.organization

    def _completion(self, prompt=None, **kwargs):
        # self.sync_openai()
        res = self.client.completion.create(engine=self.model, prompt=prompt, **kwargs)
        return CompletionObject(prompt=prompt, kwargs=kwargs, response=res)

    def _chat_completion(self, messages=None, **kwargs):
        # self.sync_openai()
        res = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        return CompletionObject(prompt=messages, kwargs=kwargs, response=res)

    def verify_response(self, res):
        res = res.response
        finish_reason = res.choices[0].finish_reason
        if finish_reason != 'stop':
            logger.warning(f"finish_reason is {finish_reason}")
        return True

    def extract_text(self, res):

        stream = res.stream
        res = res.response

        if not self.is_chat:
            res = res.choices[0].text
        else:
            if not stream:
                res = res.choices[0].message.content
            else:
                res = res.choices[0].delta.content
        return res

    def openai_format(self, res):

        res = res.response
        return res


class OpenAI(OpenAIBase):

    _models: Any = PrivateAttr()

    def __init__(self, model='gpt-3.5-turbo', api_key=None, organization=None, *args, **kwargs):

        api_key = beam_key('OPENAI_API_KEY', api_key)

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
        import openai
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
            logger.warning('No path provided, using default path: dataset.jsonl')
            path = 'dataset.jsonl'

        # Open a file for writing
        with open(path, 'w') as outfile:
            # Write each data item to the file as a separate line
            for item in records:
                json.dump(item, outfile)
                outfile.write('\n')

        return path

    def retrieve(self, model=None):
        import openai
        if model is None:
            model = self.model
        return openai.Engine.retrieve(id=model)

    @property
    def models(self):
        if self._models is None:
            import openai
            models = openai.models.list()
            models = {m.id: m for m in models.data}
            self._models = models
        return self._models

    def embedding(self, text, model=None):
        if model is None:
            model = self.model
        import openai
        response = openai.Engine(model).embedding(input=text, model=model)
        embedding = np.array(response.data[1]['embedding'])
        return embedding

