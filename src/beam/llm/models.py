import json
import math
from json import JSONDecodeError
from typing import Optional, Any
import requests
import pandas as pd
import numpy as np

from .hf_conversation import Conversation
from ..logger import beam_logger as logger
from .core import BeamLLM, CompletionObject
from pydantic import BaseModel, Field, PrivateAttr
from ..path import beam_key, normalize_host

from .utils import get_conversation_template
from ..utils import lazy_property, strip_suffix
from .openai import OpenAIBase


class FCConversationLLM(BeamLLM):

    def get_prompt(self, messages):

        conv = get_conversation_template(self.adapter)
        for m in messages:

            if m['role'] == 'system':

                role = m['system_name'] if 'system_name' in m else 'system'
                content = m['content']

            else:

                if m['role'] == 'user':
                    role = conv.roles[0]
                else:
                    role = conv.roles[1]

                content = m['content']

            conv.append_message(role, content)

        conv.append_message(conv.roles[1], None)

        return conv.get_prompt()

    @property
    def is_chat(self):
        return True

    @property
    def is_completions(self):
        return True


class TGILLM(FCConversationLLM):

    _info: Any = PrivateAttr(default=None)
    _client: Any = PrivateAttr(default=None)
    timeout: Optional[float] = Field(None)

    def __init__(self, hostname=None, port=None, *args, adapter=None, timeout=None, **kwargs):

        api_base = f"http://{normalize_host(hostname, port)}"

        client_kwargs = {}
        if timeout is not None:
            client_kwargs['timeout'] = timeout

        from text_generation import Client
        self._client = Client(api_base, **client_kwargs)

        req = requests.get(f"{api_base}/info")
        self._info = json.loads(req.text)

        kwargs['model'] = self._info['model_id']
        kwargs['scheme'] = 'tgi'

        if adapter is None:
            adapter = 'raw'
        super().__init__(*args, adapter=adapter, **kwargs)

    def update_usage(self, response):

        self.usage["prompt_tokens"] += 0
        self.usage["completion_tokens"] += response.details.generated_tokens
        self.usage["total_tokens"] += 0 + response.details.generated_tokens

    def openai_format(self, res, **kwargs):
        return super().openai_format(res, tokens=res.response.details.tokens,
                                     completion_tokens=res.response.details.generated_tokens,
                                     total_tokens=res.response.details.generated_tokens)

    @property
    def is_chat(self):
        return False

    @property
    def is_completions(self):
        return True

    def process_kwargs(self, prompt, **kwargs):

        processed_kwargs = {}

        max_tokens = kwargs.pop('max_tokens', None)
        max_new_tokens = None

        if max_tokens is not None:
            max_new_tokens = max_tokens - self.len_function(prompt)

        max_new_tokens = kwargs.pop('max_new_tokens', max_new_tokens)
        if max_new_tokens is not None:
            processed_kwargs['max_new_tokens'] = max_new_tokens

        temperature = kwargs.pop('temperature', None)
        if temperature is not None:
            processed_kwargs['temperature'] = temperature

        top_p = kwargs.pop('top_p', None)
        if top_p is not None and 0 < top_p < 1:
            processed_kwargs['top_p'] = top_p

        best_of = kwargs.pop('n', None)
        if best_of is not None and best_of > 1:
            processed_kwargs['best_of'] = best_of

        stop_sequences = kwargs.pop('stop', None)
        if stop_sequences is None:
            stop_sequences = []
        elif type(stop_sequences) is str:
            stop_sequences = [stop_sequences]

        if self.stop_sequence is not None:
            stop_sequences.append(self.stop_sequence)

        if len(stop_sequences) > 0:
            processed_kwargs['stop_sequences'] = stop_sequences

        if 'top_k' in kwargs:
            processed_kwargs['top_k'] = kwargs.pop('top_k')

        if 'truncate' in kwargs:
            processed_kwargs['truncate'] = kwargs.pop('truncate')

        if 'typical_p' in kwargs:
            processed_kwargs['typical_p'] = kwargs.pop('typical_p')

        if 'watermark' in kwargs:
            processed_kwargs['watermark'] = kwargs.pop('watermark')

        if 'repetition_penalty' in kwargs:
            processed_kwargs['repetition_penalty'] = kwargs.pop('repetition_penalty')

        decoder_input_details = kwargs.pop('logprobs', None)
        if decoder_input_details is not None:
            processed_kwargs['decoder_input_details'] = decoder_input_details

        if 'seed' in kwargs:
            processed_kwargs['seed'] = kwargs.pop('seed')

        do_sample = None
        if temperature is not None and temperature > 0:
            do_sample = True
        do_sample = kwargs.pop('do_sample', do_sample)

        if do_sample is not None:
            processed_kwargs['do_sample'] = do_sample

        return processed_kwargs

    def _completion(self, prompt=None, **kwargs):

        prompt = self.get_prompt([{'role': 'user', 'content': prompt}])
        generate_kwargs = self.process_kwargs(prompt, **kwargs)
        res = self._client.generate(prompt, **generate_kwargs)
        return CompletionObject(prompt=prompt, kwargs=kwargs, response=res)

    def _chat_completion(self, messages=None, **kwargs):

        prompt = self.get_prompt(messages)
        generate_kwargs = self.process_kwargs(prompt, **kwargs)
        res = self._client.generate(prompt, **generate_kwargs)
        return CompletionObject(prompt=prompt, kwargs=kwargs, response=res)

    def extract_text(self, res):

        res = res.response
        text = res.generated_text
        text = strip_suffix(text, self.stop_sequence)
        return text


class FastChatLLM(OpenAIBase):

    def __init__(self, model=None, hostname=None, port=None, *args, **kwargs):

        api_base = f"http://{normalize_host(hostname, port)}/v1"
        api_key = "EMPTY"  # Not support yet
        organization = "EMPTY"  # Not support yet

        kwargs['scheme'] = 'fastchat'
        super().__init__(*args, api_key=api_key, api_base=api_base, organization=organization, model=model, **kwargs)

        self.model = model

    @property
    def is_chat(self):
        return True


class FastAPILLM(FCConversationLLM):

    normalized_hostname: Optional[str] = Field(None)
    headers: Optional[dict] = Field(None)
    consumer: Optional[str] = Field(None)
    protocol: Optional[str] = Field(None)
    _models: Any = PrivateAttr(default=None)

    def __init__(self, *args, model=None, hostname=None, port=None, username=None, protocol='https', **kwargs):

        kwargs['scheme'] = 'fastapi'
        super().__init__(*args, hostname=hostname, port=port, model=model, **kwargs)

        self.consumer = username
        self.normalized_hostname = normalize_host(hostname, port)
        self._models = None
        self.headers = {'Content-Type': 'application/json'}
        self.protocol = protocol

        from requests.packages.urllib3.exceptions import InsecureRequestWarning
        # Suppress only the single InsecureRequestWarning from urllib3
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

    @property
    def models(self):
        if self._models is None:
            res = requests.get(f"{self.protocol}://{self.normalized_hostname}/models", headers=self.headers, verify=False)
            self._models = res.json()
        return self._models

    @property
    def is_chat(self):
        return True

    @property
    def is_completions(self):
        return True

    def process_kwargs(self, prompt, **kwargs):

        kwargs_processed = {}

        max_tokens = kwargs.pop('max_tokens', None)
        max_new_tokens = None

        if max_tokens is not None:
            max_new_tokens = max_tokens - self.len_function(prompt)

        max_new_tokens = kwargs.pop('max_new_tokens', max_new_tokens)
        if max_new_tokens is not None:
            kwargs_processed['max_new_tokens'] = max_new_tokens

        temperature = kwargs.pop('temperature', None)
        if temperature is not None:
            kwargs_processed['temp'] = temperature

        if 'stop_criteria' in kwargs:
            kwargs_processed['stop_criteria'] = kwargs.pop('stop_criteria')

        kwargs_processed['stream'] = kwargs.pop('stream', False)

        return kwargs_processed

    def _chat_completion(self, messages=None, **kwargs):

        prompt = self.get_prompt(messages)
        return self._completion_internal(prompt, **kwargs)

    def _completion(self, prompt=None, **kwargs):

        prompt = self.get_prompt([{'role': 'user', 'content': prompt}])
        return self._completion_internal(prompt, **kwargs)

    def _stream_generator(self, d):

        with requests.post(f"{self.protocol}://{self.normalized_hostname}/predict/stream", headers=self.headers, json=d,
                           stream=True, verify=False) as response:
            for chunk in response.iter_content(chunk_size=1024):

                chunks = chunk.split("\0")
                for c in chunks:
                    if not len(c):
                        continue
                    try:
                        yield json.loads(c.decode())
                    except JSONDecodeError:
                        logger.error(f"Bad chunk: {c}")
                        yield c

    def _completion_internal(self, prompt, stream=False, **kwargs):

        d = dict(model_name=self.model, consumer=self.consumer, input=prompt,
                 hyper_params=self.process_kwargs(prompt, **kwargs))

        if stream:
            res = self._stream_generator(d)
            return CompletionObject(prompt=d['input'], kwargs=d, response=res)

        else:
            res = requests.post(f"{self.protocol}://{self.normalized_hostname}/predict/once", headers=self.headers, json=d,
                                verify=False)
            return CompletionObject(prompt=d['input'], kwargs=d, response=res.json())

    def verify_response(self, res):

        if res.stream:
            return True

        try:

            if not res.response['is_done']:
                logger.warning(f"Model {self.model} is_done=False.")

            assert 'res' in res.response, f"Response does not contain 'res' key"

        except Exception as e:
            logger.error(f"Error in response: {res.response}")
            raise e
        return True

    def extract_text(self, res):
        return res.response['res']


class FastAPIDPLLM(FastAPILLM):

    stop_token_ids: Optional[list] = Field(None)
    stop: Optional[str] = Field(None)
    application: Optional[str] = Field(None)
    task: Optional[str] = Field(None)
    network: Optional[str] = Field(None)
    cut_long_prompt: Optional[bool] = Field(None)
    echo: Optional[bool] = Field(None)
    worker_generate_stream_ep: Optional[str] = Field(None)
    streaming_delimiter: Optional[bytes] = Field(None)
    retrials: Optional[int] = Field(None)
    timeout: Optional[int] = Field(None)

    def __init__(self, *args, application='Botson', task='None', network='NH', cut_long_prompt=False, echo=False,
                 worker_generate_stream_ep='worker_generate_stream/', streaming_delimiter=b"\0",
                 retrials=None, stop_token_ids=None, protocol='http', stop=None, timeout=60, **kwargs):

        kwargs['scheme'] = 'fastapi-dp'
        super().__init__(*args, protocol=protocol, **kwargs)

        self.application = application
        self.task = task
        self.network = network
        self.cut_long_prompt = cut_long_prompt
        self.echo = echo
        self.worker_generate_stream_ep = worker_generate_stream_ep
        self.streaming_delimiter = streaming_delimiter
        self.stop_token_ids = stop_token_ids
        self.stop = stop
        self.retrials = retrials
        self.timeout = timeout
        self.headers = {'Content-Type': 'application/json', 'accept': 'application/json'}

    def process_kwargs(self, prompt, number_of_generations=None, **kwargs):
        kwargs_processed = super().process_kwargs(prompt, **kwargs)
        max_new_tokens = kwargs_processed['max_new_tokens']
        kwargs_processed['number_of_generations'] = int(math.ceil(max_new_tokens / 32))
        return kwargs_processed

    def post_process(self, res, max_new_tokens):

        response = json.loads(res.content.split(self.streaming_delimiter)[-2].decode())
        response['text'] = self.stem_message(response['text'], max_new_tokens)

        return response

    def _completion(self, prompt=None, **kwargs):

        prompt = self.get_prompt([{'role': 'user', 'content': prompt}])
        return self._completion_internal(prompt, **kwargs)

    def _chat_completion(self, messages=None, **kwargs):

        prompt = self.get_prompt(messages)
        return self._completion_internal(prompt, **kwargs)

    def _completion_internal(self, prompt, stop=None, stop_token_ids=None, cut_long_prompt=None,
                             echo=None, task=None, network=None, application=None, **kwargs):

        kwargs_processed = self.process_kwargs(prompt, **kwargs)
        d = dict(model=self.model, user=self.consumer, max_new_tokens=kwargs['max_new_tokens'],
                 prompt=prompt, temperature=kwargs['temperature'], stop_token_ids=stop_token_ids or self.stop_token_ids,
                 stop=stop or self.stop, cut_long_prompt=cut_long_prompt or self.cut_long_prompt,
                 echo=echo or self.echo, task=self.task or self.task, network=network or self.network,
                 application=application or self.application,
                 number_of_generations=kwargs_processed['number_of_generations'])

        res = requests.post(f"{self.protocol}://{self.normalized_hostname}/{self.worker_generate_stream_ep}",
                            headers=self.headers, json=d, verify=False, timeout=self.timeout)
        res = self.post_process(res, kwargs['max_new_tokens'])
        return CompletionObject(prompt=d['prompt'], kwargs=d, response=res)

    def verify_response(self, res):
        try:

            if res.response['error_code']:
                logger.warning(f"Model {self.model}: error_code={res.response['error_code']}")

            assert 'text' in res.response, f"Response does not contain 'text' key"

        except Exception as e:
            logger.error(f"Error in response: {res.response['error_code']}")
            raise e
        return True

    def extract_text(self, res):
        return res.response['text']

    def openai_format(self, res):
        return super().openai_format(res, tokens=res.response['tokens'],
                                     completion_tokens=res.response['completion_tokens'],
                                     total_tokens=res.response['total_tokens'])


class HuggingFaceLLM(BeamLLM):
    pipline_kwargs: Any
    input_device: Optional[str] = Field(None)
    eos_pattern: Optional[str] = Field(None)
    tokenizer_name: Optional[str] = Field(None)
    _text_generation_pipeline: Any = PrivateAttr(default=None)
    _conversational_pipeline: Any = PrivateAttr(default=None)

    def __init__(self, model, tokenizer=None, dtype=None, chat=False, input_device=None, compile=True, *args,
                 model_kwargs=None,
                 config_kwargs=None, pipline_kwargs=None, text_generation_kwargs=None, conversational_kwargs=None,
                 eos_pattern=None, **kwargs):

        kwargs['scheme'] = 'huggingface'
        kwargs['model'] = model
        super().__init__(*args, **kwargs)

        import transformers
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

        from transformers import AutoModelForCausalLM, AutoConfig

        self.config = AutoConfig.from_pretrained(model, trust_remote_code=True, **config_kwargs)
        self.tokenizer_name = tokenizer or model

        self.net = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True,
                                                        config=self.config, **model_kwargs)

        if compile:
            import torch
            self.net = torch.compile(self.net)

        self._text_generation_pipeline = transformers.pipeline('text-generation', model=self.net,
                                                               tokenizer=self.tokenizer, device=self.input_device,
                                                               return_full_text=False, **text_generation_kwargs)

        self._conversational_pipeline = transformers.pipeline('conversational', model=self.net,
                                                              tokenizer=self.tokenizer, device=self.input_device,
                                                              **conversational_kwargs)

    @lazy_property
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=True)

    def extract_text(self, res):

        res = res.response
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

        return CompletionObject(prompt=prompt, kwargs=kwargs, response=res)

    def _chat_completion(self, **kwargs):

        # pipeline = transformers.pipeline('conversational', model=self.model,
        #                                  tokenizer=self.tokenizer, device=self.input_device)

        res = self._conversational_pipeline(self.conversation,
                                             pad_token_id=self._conversational_pipeline.tokenizer.eos_token_id,
                                             **self.pipline_kwargs)

        return CompletionObject(prompt=self.conversation, kwargs=kwargs, response=res)