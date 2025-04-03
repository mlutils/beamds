import json
from typing import Optional, Any, ClassVar, List
import pandas as pd
import numpy as np

from ..utils import lazy_property as cached_property

from ..logging import beam_logger as logger
from ..path import normalize_host
from .core import BeamLLM, CompletionObject
from .tools import LLMGuidance
from pydantic import Field, PrivateAttr
from ..path import beam_key


class OpenAIBase(BeamLLM):

    api_key: Optional[str] = Field(None)
    api_base: Optional[str] = Field(None)
    organization: Optional[str] = Field(None)
    _models: Any = PrivateAttr(default=None)

    chat_kwargs: ClassVar[List[str]] = \
                  ['frequency_penalty', 'function_call', 'functions', 'logit_bias', 'logprobs', 'max_tokens',
                   'n', 'presence_penalty', 'seed', 'stop', 'stream', 'temperature',
                   'tool_choice', 'tools', 'top_logprobs', 'top_p', 'user', 'extra_headers', 'extra_query',
                   'extra_body', 'timeout']

    completion_kwargs: ClassVar[List[str]] = \
                        ['max_tokens', 'temperature', 'top_p', 'n', 'logprobs', 'stream', 'logit_bias',
                         'stop', 'presence_penalty', 'frequency_penalty', 'best_of', 'echo', 'user', 'extra_headers',
                         'extra_query', 'extra_body', 'timeout']

    def __init__(self, model=None, api_key=None, api_base=None, organization=None, *args, **kwargs):
        super().__init__(*args, model=model, **kwargs)

        self.api_key = api_key
        self.api_base = api_base
        self.organization = organization
        self._models = None

    @cached_property
    def client(self):
        from openai import OpenAI
        http_client = None
        if self.api_base:
            import httpx
            http_client = httpx.Client(transport=httpx.HTTPTransport(verify=False))

        return OpenAI(organization=self.organization, api_key=self.api_key, base_url=self.api_base,
                      http_client=http_client)

    def update_usage(self, response):

        if 'usage' in response:
            response = response['usage']

            self.usage["prompt_tokens"] += response["prompt_tokens"]
            self.usage["completion_tokens"] += response["completion_tokens"]
            self.usage["total_tokens"] += response["prompt_tokens"] + response["completion_tokens"]

    def _completion(self, prompt, **kwargs):
        kwargs = self.filter_completion_kwargs(kwargs)
        res = self.client.completions.create(model=self.model, prompt=prompt,  **kwargs)
        return CompletionObject(prompt=prompt, kwargs=kwargs, response=res)

    def _chat_completion(self, chat, stream=None, guidance=None, **kwargs):
        kwargs = self.filter_chat_kwargs(kwargs)
        messages = chat.openai_format
        if guidance is None:
            res = self.client.chat.completions.create(model=self.model, messages=messages,
                                                      stream=stream, **kwargs)
        else:
            assert isinstance(guidance, LLMGuidance), "guidance must be an instance of LLMGuidance"
            res = self.client.beta.chat.completions.parse(model=self.model, messages=messages,
                                                          response_format=guidance.guided_model, **kwargs)
        return CompletionObject(prompt=messages, kwargs=kwargs, response=res)

    @staticmethod
    def filter_chat_kwargs(kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in OpenAIBase.chat_kwargs}
        return {k: v for k, v in kwargs.items() if v is not None}

    @staticmethod
    def filter_completion_kwargs(kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in OpenAIBase.completion_kwargs}
        return {k: v for k, v in kwargs.items() if v is not None}

    def verify_response(self, res):
        stream = res.stream
        res = res.response

        if not hasattr(res.choices[0], 'finish_reason'):
            return False

        finish_reason = res.choices[0].finish_reason
        if finish_reason != 'stop' and not stream:
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

    def parse_json(self, res):
        res = res.response
        if self.is_chat:
            res = res.choices[0].message
            if hasattr(res, 'parsed'):
                return res.parsed
        return None

    def openai_format(self, res):

        res = res.response
        return res

    def retrieve(self, model=None):
        import openai
        if model is None:
            model = self.model
        return openai.Engine.retrieve(id=model)

    @property
    def models(self):
        if self._models is None:
            models = self.client.models.list()
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


class OpenAILLM(OpenAIBase):

    def __init__(self, model='gpt-3.5-turbo', api_key=None, organization=None, *args, **kwargs):

        api_key = beam_key('OPENAI_API_KEY', api_key)
        kwargs['scheme'] = 'openai'
        super().__init__(model=model, api_key=api_key, api_base='https://api.openai.com/v1',
                         organization=organization, *args, **kwargs)

    @property
    def is_chat(self):
        instruct_models = ['gpt-3.5-turbo-instruct', 'babbage-002', 'davinci-002']
        if self.model in instruct_models:
            return False
        return True

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


class SamurOpenAI(OpenAIBase):

    _is_chat: Any = PrivateAttr(default=None)

    def __init__(self, model=None, hostname=None, api_key=None, port=None, chat=True, tls=False, *args, **kwargs):

        http_scheme = 'http' if not tls else 'https'
        api_base = f"{http_scheme}://{normalize_host(hostname, port)}/openai/v1"
        super().__init__(*args, model=model, api_key=api_key, api_base=api_base,  scheme='samur-openai', **kwargs)
        self._is_chat = chat

    @property
    def is_chat(self):
        return self._is_chat

    @staticmethod
    def filter_keys(kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in ['max_tokens', 'temperature', 'extra_body']}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return kwargs

    @staticmethod
    def add_guidance(kwargs, guidance=None):
        if guidance is not None:
            extra_body = kwargs.get('extra_body', {})
            extra_body = {**extra_body, **guidance.arguments(filter=['guided_regex',
                                                                     'guided_choice',
                                                                     'guided_grammar',
                                                                     'guided_json'])}
            kwargs['extra_body'] = extra_body
        return kwargs

    def _completion(self, prompt, guidance=None, **kwargs):
        kwargs = self.add_guidance(kwargs, guidance)
        kwargs = self.filter_keys(kwargs)
        return super()._completion(prompt, **kwargs)

    def _chat_completion(self, chat, guidance=None, **kwargs):
        kwargs = self.add_guidance(kwargs, guidance)
        kwargs = self.filter_keys(kwargs)
        return super()._chat_completion(chat, **kwargs)


class BeamVLLM(OpenAIBase):

    def __init__(self, model=None, hostname=None, api_key=None, port=None, tls=False, *args, **kwargs):

        http_scheme = 'http' if not tls else 'https'
        api_base = f"{http_scheme}://{normalize_host(hostname, port)}/v1"
        super().__init__(*args, model=model, api_key=api_key, api_base=api_base, scheme='vllm',  **kwargs)

    @property
    def is_chat(self):
        return True


class FastChatLLM(OpenAIBase):

    def __init__(self, model=None, hostname=None, port=None, *args, **kwargs):

        api_base = f"http://{normalize_host(hostname, port)}/v1"
        api_key = "EMPTY"  # Not support yet
        organization = "EMPTY"  # Not support yet

        super().__init__(*args, api_key=api_key, api_base=api_base, organization=organization, model=model,
                         scheme='fastchat', **kwargs)

    @property
    def is_chat(self):
        return True



