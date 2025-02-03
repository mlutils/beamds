import json
from dataclasses import dataclass
from functools import partial
from typing import Optional, Any, Dict, List, Mapping, Union
from uuid import uuid4 as uuid
import pandas as pd
from pydantic import BaseModel

from ..utils import lazy_property as cached_property

from ..logging import beam_logger as logger
from .response import LLMResponse
from .utils import estimate_tokens, split_to_tokens
from .chat import BeamChat
from ..utils import (parse_text_to_protocol, get_edit_ratio, retry, BeamDict, NullClass,
                     pretty_print_dict)
from ..path import BeamURL, BeamResource

# from langchain.llms.base import LLM
# from langchain.callbacks.manager import CallbackManagerForLLMRun

LLM = BaseModel
CallbackManagerForLLMRun = NullClass
# import pydantic and check if its version is less than 2.0.0
import pydantic

if pydantic.__version__ < '2.0.0':
    try:
        from langchain.llms.base import LLM
        from langchain.callbacks.manager import CallbackManagerForLLMRun
    except ImportError:
        logger.warning("langchain not found, using pydantic only as the LLM base")
else:
    from langchain import __version__ as langchain_version
    if langchain_version < '0.3.0':
        logger.warning("pydantic version >= 2.0.0 is incompatible with langchain < 0.3.0, using pydantic only as the LLM base")
    else:
        from langchain.llms.base import LLM
        from langchain.callbacks.manager import CallbackManagerForLLMRun

from pydantic import Field, PrivateAttr
from .utils import get_conversation_template
from .tools import LLMTool


# make CompletionObject with dataclasses
@dataclass
class CompletionObject:
    prompt: str
    kwargs: Dict
    response: Any
    valid: bool = True


class ChatCompletionChunk(BeamDict):
    def __repr__(self):
        return pretty_print_dict(self, 'ChatCompletionChunk')


class ChatCompletion(BeamDict):
    def __repr__(self):
        return pretty_print_dict(self, 'ChatCompletion')


class Completion(BeamDict):
    def __repr__(self):
        return pretty_print_dict(self, 'Completion')


class PedanticBeamResource(LLM, BeamResource):
    url: Optional[BeamURL] = Field(None)
    resource_type: Optional[str] = Field(None)
    scheme: Optional[str] = Field(None)

    # url: BeamURL
    # resource_type: str
    # scheme: str

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *args, resource_type=None, scheme=None, **kwargs):
        super().__init__(*args, resource_type=resource_type, scheme=scheme, **kwargs)


# class BeamLLM(LLM, Processor, metaclass=CombinedMeta):
class BeamLLM(PedanticBeamResource):
    model: Optional[str] = Field(None)
    scheme: Optional[str] = Field(None)
    usage: Dict = Field(default_factory=dict)
    instruction_history: List = Field(default_factory=list)
    _chat: Any = PrivateAttr(default=None)
    _url: Any = PrivateAttr(default=None)
    _debug_langchain: Any = PrivateAttr(default=None)
    temperature: float = Field(1.0, ge=0.0, le=1.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    n: int = Field(1, ge=1)
    message_stream: bool = Field(False)
    stop: Optional[Union[str, List[str]]] = Field(None)
    max_tokens: Optional[int] = Field(None)
    max_new_tokens: Optional[int] = Field(None)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = Field(None)
    parse_retries: int = Field(3, ge=0)
    ask_retries: int = Field(1, ge=1)
    sleep: float = Field(1.0, ge=0.0)
    adapter: Optional[str] = Field(None)
    system: Optional[str] = Field(None)
    _len_function: Any = PrivateAttr(default=None)

    # To be used with pydantic classes and lazy_property
    _lazy_cache: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)
    _path_to_tokenizer: Any = PrivateAttr(default=None)

    _assistant_budget: Any = PrivateAttr(default=None)
    _assistant_docstrings: Any = PrivateAttr(default=None)
    _conv: Any = PrivateAttr(default=None)
    _tools: Any = PrivateAttr(default=None)
    _init_is_done: Any = PrivateAttr(default=None)

    def __init__(self, *args, temperature=.1, top_p=1, n=1, stream=False, stop=None, max_tokens=None,
                 presence_penalty=0,
                 frequency_penalty=0.0, logit_bias=None, scheme='unknown', model=None, max_new_tokens=None,
                 ask_retries=1,
                 debug_langchain=False, len_function=None, tokenizer=None, path_to_tokenizer=None, parse_retries=3,
                 sleep=1,
                 adapter=None, tools=None, systerm=None, **kwargs):
        super().__init__(resource_type='llm', scheme=scheme, **kwargs)

        if temperature is not None:
            temperature = float(temperature)
        self.temperature = temperature
        if top_p is not None:
            top_p = float(top_p)
        self.top_p = top_p
        if n is not None:
            n = int(n)
        self.n = n
        self.message_stream = bool(stream)
        self.stop = stop
        if max_tokens is not None:
            max_tokens = int(max_tokens)
        self.max_tokens = max_tokens
        if max_new_tokens is not None:
            max_new_tokens = int(max_new_tokens)
        self.max_new_tokens = max_new_tokens
        if presence_penalty is not None:
            presence_penalty = float(presence_penalty)
        self.presence_penalty = presence_penalty
        if frequency_penalty is not None:
            frequency_penalty = float(frequency_penalty)
        self.frequency_penalty = frequency_penalty

        self.parse_retries = int(parse_retries)
        self.ask_retries = int(ask_retries)
        self.sleep = float(sleep)
        self.logit_bias = logit_bias
        self.scheme = scheme
        self.system = systerm
        self._url = None
        self._assistant_budget = None
        self._assistant_docstrings = None
        self._debug_langchain = debug_langchain
        self.instruction_history = []

        self.model = model
        if self.model is None:
            self.model = 'unknown'

        if adapter is None:
            adapter = self.model
        self.adapter = adapter

        self._conv = get_conversation_template(self.adapter)
        self.stop = self.build_default_stop_sequences(stop)

        self.usage = {"prompt_tokens": 0,
                      "completion_tokens": 0,
                      "total_tokens": 0}

        self._chat = BeamChat(scheme=self.scheme, adapter=self.adapter)

        self._path_to_tokenizer = path_to_tokenizer
        self._tokenizer = tokenizer
        self._len_function = len_function

        self.reset_chat()
        self._lazy_cache = {}
        self._tools = tools

    def build_default_stop_sequences(self, stop: Optional[Union[str, List[str]]] = None) -> Optional[List[str]]:
        adapter_stop = self.stop_sequence
        if type(adapter_stop) is not list:
            adapter_stop = [adapter_stop]
        adapter_stop.append(self.sep)
        model_stop = self.build_stop_sequences(stop, default_stop=adapter_stop)
        return model_stop

    def build_stop_sequences(self, stop, default_stop=None):

        if default_stop is None:
            default_stop = self.stop
        if default_stop is None:
            default_stop = []
        elif type(default_stop) is not list:
            default_stop = [default_stop]

        all_stops = default_stop.copy()

        if stop is None:
            stop = []
        elif type(stop) is not list:
            stop = [stop]

        all_stops.extend(stop)
        all_stops = list(set(all_stops))

        # filter out None values
        all_stops = [i for i in all_stops if i is not None]
        # filter out empty strings
        all_stops = [i for i in all_stops if i.strip()]

        if len(all_stops) == 0:
            return None
        elif len(all_stops) == 1:
            return all_stops[0]
        else:
            return all_stops

    def reset_chat(self):
        self._chat.reset()

    def add_tool(self, name=None, tool_type='function', func=None, description=None, tool=None, **kwargs):
        if self._tools is None:
            self._tools = {}
        if name in self._tools:
            return
        if type(name) is LLMTool:
            tool = name
        if tool is None:
            tool = LLMTool(name=name, tool_type=tool_type, description=description, func=func, **kwargs)
        self._tools[name] = tool

    def remove_tool(self, name):
        if self._tools is None:
            return
        if name in self._tools:
            self._tools.pop(name)
        if len(self._tools) == 0:
            self._tools = None

    def judge(self, prompt, response, **kwargs):
        # boolean judgement of response by LLM for the prompt

        judge_prompt = (f"Evaluate the following response by an LLM for accuracy and relevance to the given prompt.\n"
                        f"Original Prompt: {prompt}\n"
                        f"LLM Response: {response}\n"
                        f"Based on the factual correctness, relevance to the original prompt, and logical consistency, "
                        f"is the provided response accurate? "
                        f"Respond with 'True' for accurate or 'False' for inaccurate.")

        res = self.ask(judge_prompt, **kwargs)

        return res.bool

    @property
    def tools(self):
        return self._tools

    def tool_message(self, tools=None):

        tools = tools or self.tools
        if tools is None:
            return None

        all_tools_message = ""
        for tool in tools:
            all_tools_message = f"{all_tools_message}\n\n{tool.name}:\n{tool.dict()}\n"

        message = (f"\nBelow is a list of available tools:\n"
                   f"{all_tools_message}"
                   f"To activate a tool use the following pattern with a JSON formatted argument:\n"
                   f"[TOOL][<tool name>]{{\"args\": [<list of args>], "
                   f"\"kwargs\": {{<a dictionary of kwargs>}}}}[/TOOL]\n\n")
        return message

    @property
    def stop_sequence(self):
        return self._conv.stop_str

    @property
    def sep(self):
        return self._conv.sep

    @property
    def sep2(self):
        return self._conv.sep2

    @property
    def roles(self):
        return self._conv.roles

    @cached_property
    def tokenizer(self):
        if self._tokenizer is not None:
            tokenizer = self._tokenizer
        elif self._path_to_tokenizer is not None:
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=self._path_to_tokenizer)
        else:
            try:
                import tiktoken
                enc = tiktoken.encoding_for_model(self.model).encode
            except ImportError:
                enc = split_to_tokens
            except KeyError:
                try:
                    enc = tiktoken.encoding_for_model("gpt-4").encode
                except:
                    enc = split_to_tokens

            def tokenizer(text):
                return {"input_ids": enc(text)}

        return tokenizer

    @property
    def url(self):

        if self._url is None:
            self._url = BeamURL(scheme=self.scheme, path=self.model)

        return str(self._url)

    def extract_choices(self, response):
        raise NotImplementedError

    @property
    def _llm_type(self) -> str:
        return "beam_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:

        res = self.ask(prompt, stop=stop)
        if self._debug_langchain:
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Response: {res.text}")

        return res.text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"is_chat": self.is_chat,
                'usage': self.usage}

    def len_function(self, prompt):

        if self.tokenizer is not None:
            return len(self.tokenizer(prompt)['input_ids'])

        if self._len_function is None:
            return estimate_tokens(prompt)

        return self._len_function(prompt)

    @property
    def is_chat(self):
        raise NotImplementedError

    @property
    def is_completions(self):
        return not self.is_chat

    def _chat_completion(self, chat, **kwargs) -> CompletionObject:
        raise NotImplementedError

    def chat_completion(self, chat, stream=False, parse_retries=3, sleep=1, ask_retries=1,
                        prompt_type='chat_completion', **kwargs):

        chat_completion_function = self._chat_completion
        if ask_retries > 1:
            chat_completion_function = retry(func=chat_completion_function, retries=ask_retries,
                                             sleep=sleep, logger=logger,
                                             name=f"LLM chat-completion with model: {self.model}")

        completion_object = chat_completion_function(chat, stream=stream, **kwargs)
        response = LLMResponse(completion_object.response, self, chat=True, stream=stream,
                               parse_retries=parse_retries, sleep=sleep, prompt_type=prompt_type,
                               prompt_kwargs=completion_object.kwargs, prompt=completion_object.prompt)

        return response

    def completion(self, prompt, parse_retries=3, sleep=1, ask_retries=1, prompt_type='completion', stream=False,
                   **kwargs):

        completion_function = self._completion
        if ask_retries > 1:
            completion_function = retry(func=completion_function, retries=ask_retries,
                                        sleep=sleep, logger=logger,
                                        name=f"LLM completion with model: {self.model}")

        completion_object = completion_function(prompt, stream=stream, **kwargs)
        response = LLMResponse(completion_object.response, self, chat=False, stream=stream,
                               parse_retries=parse_retries, sleep=sleep, prompt_type=prompt_type,
                               prompt_kwargs=completion_object.kwargs, prompt=completion_object.prompt,
                               verify=completion_object.valid)
        return response

    def _completion(self, message, **kwargs) -> CompletionObject:
        raise NotImplementedError

    def get_default_params(self, temperature=None, top_p=None, n=None, stream=None, stop=None, max_tokens=None,
                           presence_penalty=None, frequency_penalty=None, logit_bias=None, max_new_tokens=None,
                           parse_retries=None, ask_retries=None, sleep=None):

        temperature = temperature or (self.temperature if self.temperature != self.__fields__['temperature'].default
                                      else None)
        top_p = top_p or (self.top_p if self.top_p != self.__fields__['top_p'].default else None)
        n = n or (self.n if self.n != self.__fields__['n'].default else None)
        stream = stream or self.message_stream

        stop = self.build_stop_sequences(stop)

        presence_penalty = (presence_penalty or
                            (self.presence_penalty if self.presence_penalty
                             != self.__fields__['presence_penalty'].default else None))
        frequency_penalty = frequency_penalty or (self.frequency_penalty if self.frequency_penalty
                                                  != self.__fields__['frequency_penalty'].default else None)
        logit_bias = logit_bias or self.logit_bias
        max_new_tokens = max_new_tokens or self.max_new_tokens
        max_tokens = max_tokens or self.max_tokens
        ask_retries = ask_retries or self.ask_retries
        parse_retries = parse_retries or self.parse_retries
        sleep = sleep or self.sleep

        kwargs = {'temperature': temperature,
                  'top_p': top_p,
                  'n': n,
                  'stream': stream,
                  'stop': stop,
                  'max_tokens': max_tokens,
                  'presence_penalty': presence_penalty,
                  'frequency_penalty': frequency_penalty,
                  'parse_retries': parse_retries,
                  'ask_retries': ask_retries,
                  'sleep': sleep}

        if max_new_tokens is not None:
            kwargs['max_new_tokens'] = max_new_tokens
        if logit_bias is not None:
            kwargs['logit_bias'] = logit_bias

        return kwargs

    def chat(self, message, system=None, reset_chat=False, temperature=None,
             top_p=None, n=None, stream=None, stop=None, max_tokens=None, presence_penalty=None, frequency_penalty=None,
             logit_bias=None, max_new_tokens=None, parse_retries=None, sleep=None, ask_retries=None, tools=None,
             prompt_type='chat_completion', guidance=None, image=None, images=None,
             overwrite_system_message=True, **kwargs):

        """

        @param message:
        @param system:
        @param reset_chat:
        @param temperature:
        @param top_p:
        @param n:
        @param stream:
        @param stop:
        @param max_tokens:
        @param presence_penalty:
        @param frequency_penalty:
        @param logit_bias:
        @param max_new_tokens:
        @param parse_retries:
        @param sleep:
        @param ask_retries:
        @param tools:
        @param prompt_type:
        @param guidance:
        @param image:
        @param images:
        @param overwrite_system_message:
        @param kwargs:
        @return:
        """

        default_params = self.get_default_params(temperature=temperature,
                                                 top_p=top_p, n=n, stream=stream,
                                                 stop=stop, max_tokens=max_tokens,
                                                 presence_penalty=presence_penalty,
                                                 frequency_penalty=frequency_penalty,
                                                 logit_bias=logit_bias, max_new_tokens=max_new_tokens,
                                                 parse_retries=parse_retries, sleep=sleep, ask_retries=ask_retries)

        system = system or self.system

        if reset_chat:
            self.reset_chat()

        if system:
            self._chat.add_system_message(system, overwrite=overwrite_system_message)

        images = images or []
        if image is not None:
            images.insert(0, image)

        if type(message) is str:
            self._chat.add_user_message(message, images=images)
        elif type(message) is BeamChat:
            self._chat = message
        else:
            raise ValueError("message should be a string or a BeamChat object")

        tools = tools or self.tools
        if tools:
            self._chat.add_tool_message(self.tool_message(tools), overwrite=True)

        response = self.chat_completion(chat=self._chat, prompt_type=prompt_type, guidance=guidance, tools=tools,
                                        system=system, **default_params, **kwargs)

        if not response.is_valid:
            return response
        elif stream:
            def gen():

                generated_text = []
                for res in response:
                    if res.text is not None:
                        generated_text.append(res.text)
                    yield res

                generated_text = "".join(generated_text)
                self.update_usage(res.response)
                self._chat.add_assistant_message(generated_text)

            return gen()

        else:
            self.update_usage(response.response)
            self._chat.add_assistant_message(response.text)
            return response

    def chat_with_assistant(self, message=None, docstrings=None, budget=3, **kwargs):

        if docstrings is not None:
            self._assistant_budget = budget
            self._assistant_docstrings = docstrings
            self.reset_chat()
        else:
            budget = self._assistant_budget
            docstrings = self._assistant_docstrings

        docs = ""
        for i, (k, v) in enumerate(docstrings.items()):
            docs += (f"{i}. Function: {k}\n\n{v}\n\n"
                     f"[SEP]\n")

        system = (f"You are given a list of docstrings of several functions. Each docstring details the function name, "
                  f"its purpose, its arguments and its keyworded arguments.\n"
                  f"Your task is to handle a chat session with a user. The user will ask you to apply some functionality "
                  f"or execute some action. You need to chose the function you see fit out of the list of available functions "
                  f"and you need to fill its arguments according to the user request. If the request of the user is unclear and "
                  f"you are not able to assign a proper function, you can ask for clarification. If some of the arguments which "
                  f"are required in the function are missing from the user request, you should ask he or she to provide them. "
                  f"If you are confident that you can assign a function to the user request and fill all the required arguments, "
                  f"and fill as much as possible keyworded arguments, you need to respond in a single valid JSON object "
                  f"of the form {{\"function\": <function name>, \"args\": [list of arguments], \"kwargs\": {{<dictionary of kwargs>}} }}\n\n"
                  f"[SEP]\n\n"
                  f"{docs}\n")

        if message is None:
            message = "Hello, how can you help me?"

        if len(self.chat_history) < 2 * budget - 2:
            res = self.chat(message, system=system, **kwargs)
        else:

            self.add_to_chat(message, is_user=True)

            instruction = (
                f"You are given a list of docstrings of several functions. Each docstring details the function name, "
                f"its purpose, its arguments and its keyworded arguments.\n"
                f"In addition, you are given a chat session between a user and a chatbot assistant. The user asks the chatbot "
                f"apply some functionality or execute some action. You need to chose the function you see fit out"
                f" of the list of available functions and to fill its arguments and as much as possible "
                f"keyworded arguments according to the user request. You are required to respond in a single "
                f"valid JSON object of the form {{\"function\": <function name>, \"args\": [list of arguments], "
                f"\"kwargs\": {{<dictionary of kwargs>}} }}\n\n"
                f"[SEP]\n\n"
                f"Docstrings:\n\n"
                f"{docs}\n"
                f"Chat session:\n\n"
                f"{self.chat_history}"
                f"[SEP]\n\n"
                f"Response: \"\"\"\n{{text input here}}\n\"\"\"")

            res = self.ask(instruction, system=system, **kwargs)

        return res

    def explain_traceback(self, traceback, n_words=100, **kwargs):
        prompt = (f"Task: explain the following traceback and suggest a correction. Be concise don't use more than"
                  f"{n_words} words and don't use newlines.\n\n"
                  f"[SEP]\n\n"
                  f"{traceback}\n\n"
                  f"[SEP]\n\n"
                  f"Response: \"\"\"\n{{text input here}}\n\"\"\"")

        res = self.ask(prompt, **kwargs)
        return res.text

    def fix_protocol(self, text, protocol='json', **kwargs):

        protocols_text = {'json': 'JSON', 'html': 'HTML', 'xml': 'XML', 'csv': 'CSV', 'yaml': 'YAML', 'toml': 'TOML'}

        prompt = (f"Task: fix the following {protocols_text[protocol]} object. "
                  f"Return a valid {protocols_text[protocol]} object without anything else\n\n"
                  f"[SEP]\n\n"
                  f"{text}\n\n"
                  f"[SEP]\n\n"
                  f"Response: \"\"\"\n{{text input here}}\n\"\"\"")
        res = self.ask(prompt).text

        res = parse_text_to_protocol(res, protocol=protocol)

        return res

    def docstring(self, text, element_type, name=None, docstring_format=None, parent=None, parent_name=None,
                  parent_type=None, children=None, children_type=None, children_name=None, **kwargs):

        if docstring_format is None:
            docstring_format = f"in \"{docstring_format}\" format, "
        else:
            docstring_format = ""

        prompt = f"Task: write a full python docstring {docstring_format}for the following {element_type}\n\n" \
                 f"[SEP]\n\n" \
                 f"{text}\n\n" \
                 f"[SEP]\n\n"

        if parent is not None:
            prompt = f"{prompt}" \
                     f"where its parent {parent_type}: {parent_name}, has the following docstring\n\n" \
                     f"[SEP]\n\n" \
                     f"{parent}\n\n" \
                     f"[SEP]\n\n"

        if children is not None:
            for i, (c, cn, ct) in enumerate(zip(children, children_name, children_type)):
                prompt = f"{prompt}" \
                         f"and its #{i} child: {ct} named {cn}, has the following docstring\n\n" \
                         f"[SEP]\n\n" \
                         f"{c}\n\n" \
                         f"[SEP]\n\n"

        prompt = f"{prompt}" \
                 f"Response: \"\"\"\n{{docstring text here (do not add anything else)}}\n\"\"\""

        if not self.is_completions:
            try:
                res = self.chat(prompt, **kwargs)
            except Exception as e:
                logger.error(f"Error in response: {e}")
                try:
                    logger.warning(f"{name}: switching to gpt-4 model")
                    res = self.chat(prompt, model='gpt-4', **kwargs)
                except:
                    logger.error(f"{name}: error in response")
                    res = None
        else:
            res = self.ask(prompt, **kwargs)

        return res

    def openai_format(self, res, finish_reason="stop", tokens=None, completion_tokens=0, prompt_tokens=0,
                      total_tokens=0):

        stream = res.stream
        text = self.extract_text(res)

        # an example of openai format:
        # ChatCompletionChunk(id='chatcmpl-8ooipEUzmkwlqMVnUJHZzWQ4uKyKj', choices=[
        #     Choice(delta=ChoiceDelta(content=None, function_call=None, role=None, tool_calls=None),
        #            finish_reason='stop', index=0, logprobs=None)], created=1707122067, model='gpt-4-0613',
        #                     object='chat.completion.chunk', system_fingerprint=None)
        #
        # Completion(id='cmpl-8op1yf5pOL7oEd7EXulOFbODmRSFp', choices=[
        #     CompletionChoice(finish_reason='length', index=0, logprobs=None,
        #                      text='\n\nI am an AI and do not have emotions, but thank you for asking')],
        #            created=1707123254, model='gpt-3.5-turbo-instruct', object='text_completion',
        #            system_fingerprint=None,
        #            usage=CompletionUsage(completion_tokens=16, prompt_tokens=4, total_tokens=20))

        if stream:
            Class = ChatCompletionChunk
            choice = {
                "index": 0,
                "delta": {"content": text,
                          "function_call": None,
                          "role": None,
                          "tool_calls": None},
                "logprobs": None,
                "finish_reason": finish_reason,
            }

        elif res.chat:
            Class = ChatCompletion

            if len(self.tools):

                tool_calls = []
                for t in self.tools:
                    et = t(res)
                    if et is not None and et.success:

                        arguments = {}
                        for i, k in enumerate(et.args):
                            arguments[t.args[i]] = k
                        for k, v in et.kwargs.items():
                            arguments[k] = v

                        tool_res = {
                            "id": f"call_{uuid()}",
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "arguments": json.dumps(arguments)
                            }}
                        tool_calls.append(tool_res)

            if len(tool_calls):
                choice = {
                    "finish_reason": finish_reason,
                    "index": 0,
                    "message": {
                        "content": None,
                        "role": "assistant",
                        "tool_calls": tool_calls
                    }
                }
            else:

                choice = {
                    "finish_reason": finish_reason,
                    "index": 0,
                    "message": {
                        "content": text,
                        "role": "assistant"
                    }
                }
        else:

            Class = Completion
            choice = {
                "finish_reason": finish_reason,
                "index": 0,
                "logprobs": tokens,
                "text": text
            }

        res = {
            "choices": [
                choice
            ],
            "created": res.created,
            "id": res.id,
            "model": res.model,
            "object": res.object,
            "system_fingerprint": None,
            "usage": {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens
            }
        }

        return Class(**res)

    def ask(self, prompt, system=None, max_tokens=None, temperature=None, top_p=None, frequency_penalty=None,
            max_new_tokens=None,
            presence_penalty=None, stop=None, n=None, stream=None, logprobs=None, logit_bias=None, echo=False,
            parse_retries=None, sleep=None, ask_retries=None, prompt_type='completion', tools=None,
            image=None, images=None, guidance=None, **kwargs):
        """

        @param prompt:
        @param max_tokens:
        @param temperature:
        @param top_p:
        @param frequency_penalty:
        @param max_new_tokens:
        @param presence_penalty:
        @param stop:
        @param n:
        @param stream:
        @param logprobs:
        @param logit_bias:
        @param echo:
        @param parse_retries:
        @param sleep:
        @param ask_retries:
        @param prompt_type:
        @param tools:
        @param guidance:
        @param kwargs:
        @return:
        """

        system = system or self.system
        tools = tools or self.tools

        default_params = self.get_default_params(temperature=temperature,
                                                 top_p=top_p, n=n, stream=stream,
                                                 stop=stop, max_tokens=max_tokens,
                                                 presence_penalty=presence_penalty,
                                                 frequency_penalty=frequency_penalty,
                                                 logit_bias=logit_bias,
                                                 max_new_tokens=max_new_tokens, ask_retries=ask_retries,
                                                 parse_retries=parse_retries, sleep=sleep)

        chat = BeamChat(scheme=self.scheme, adapter=self.adapter, messages=[{'role': 'user', 'content': prompt}],
                        system_message=system, tool_message=self.tool_message(tools))


        if not self.is_completions:
            kwargs = {**default_params, **kwargs}
            response = self.chat(chat, reset_chat=True, prompt_type=f'simulated_{prompt_type}_with_chat',
                                 guidance=guidance, image=image, images=images, **kwargs)
        else:

            prompt = chat.fastchat_format
            response = self.completion(prompt, logprobs=logprobs, echo=echo,
                                       prompt_type=prompt_type, guidance=guidance, **default_params)

        if not response.is_valid:
            return response
        elif stream:
            def gen():
                for res in response:
                    yield res

                if not self.is_completions:
                    # currently openai stream does not support update usage
                    self.update_usage(res.response)
                    self.instruction_history.append({'question': prompt, 'response': res.response, 'type': 'ask'})

            return gen()

        else:
            self.update_usage(response.response)
            self.instruction_history.append({'question': prompt, 'response': response, 'type': 'ask'})
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

    def verify_response(self, res):
        return True

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
                 f"[SEP]\n" \
                 f" {text}\n" \
                 f"[SEP]\n" \
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
        check if one or more key words found in given text
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
        check if one or more key words found in given text
        :param keywords:  key words list
        :param kwargs: additional arguments for the ask function
        :return: similar terms
        """
        prompt = f"keywords: {keywords}\nTask: return all semantic terms for given Keywords \nResponse: \"\"\"\n{{text input here}}\n\"\"\""

        res = self.ask(prompt, **kwargs).text
        res = res.lower().strip()
        return res

    def update_usage(self, response):
        pass

    def stem_message(self, message, n_tokens, skip_special_tokens=True):
        if self.tokenizer is not None and hasattr(self.tokenizer, 'decode'):
            tokens = self.tokenizer(message)['input_ids'][:n_tokens + 1]  # +1 for the start token
            return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
        else:
            return ''.join(split_to_tokens(message)[:n_tokens])
