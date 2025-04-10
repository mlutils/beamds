import json
from typing import Dict, List, Any
from argparse import Namespace

from ..llm import beam_llm
from ..llm import LLMGuidance
from ..processor import MetaDispatcher
from ..utils import check_type, safe_getmembers, cached_property
from ..logging import beam_logger as logger
import inspect
import threading
import queue
import re

# do not remove the following imports
import torch
import numpy as np
import pandas as pd
# do not remove the above imports

from pydantic import BaseModel


class MethodSelection(BaseModel):
    method: str


class ArgumentsSelection(BaseModel):
    args: List[Any]
    kwargs: Dict[str, Any]


class BeamAssistant(MetaDispatcher):

    def __init__(self, obj, *args, llm=None, llm_kwargs=None, summary_len=10, eval_arguments=True, retries=3,
                 _summary=None, execute=True, **kwargs):

        super().__init__(obj, *args, summary_len=summary_len,
                         eval_arguments=eval_arguments, retries=retries, **kwargs)

        self.summary_len = self.get_hparam('summary_len', summary_len)
        self.eval_arguments = self.get_hparam('eval_arguments', eval_arguments)
        self.execute = self.get_hparam('execute', execute)
        self.retries = self.get_hparam('retries', retries)
        self.guidance = {'method_selection': LLMGuidance(guided_json=MethodSelection),
                         'arguments_selection': LLMGuidance(guided_json=ArgumentsSelection)}

        llm = self.get_hparam('llm', llm)
        llm_kwargs = llm_kwargs or {}

        self._llm = None
        if llm is not None:
            self._llm = beam_llm(llm, **llm_kwargs)

        self._summary_queue = queue.Queue()
        self._summarize_thread = None
        if _summary is not None:
            self._summary_queue.put(_summary)
        else:
            # summarize the object in a different thread
            self._summarize_thread = threading.Thread(target=self.summarize, daemon=True)
            self._summarize_thread.start()

    @cached_property
    def summary(self):
        if self._summarize_thread is not None:
            # wait for the thread to finish and get the summary from the queue
            self._summarize_thread.join()
        return self._summary_queue.get()

    @cached_property
    def doc(self):
        return self.real_object.__doc__

    @cached_property
    def source(self):
        if self.type in ['class', 'instance']:
            # iterate over all parent classes and get the source
            sources = []
            base_cls = self.real_object if self.type == 'class' else self.real_object.__class__
            for cls in inspect.getmro(base_cls):
                if cls.__module__ != 'builtins':
                    sources.append(inspect.getsource(cls))
            # sum all the sources
            return '\n'.join(sources)
        else:
            return inspect.getsource(self.real_object)

    @property
    def name(self):
        if self.type == 'class':
            return self.real_object.__name__
        elif self.type == 'instance':
            return self.real_object.__class__.__name__
        else:
            return self.real_object.__name__

    @cached_property
    def type_name(self):
        if self.type == 'class':
            return 'class'
        elif self.type == 'instance':
            return 'class instance'
        elif self.type == 'function':
            return 'function'
        elif self.type == 'method':
            return 'class method'
        else:
            raise ValueError(f"Unknown type: {self.type}")

    def summarize(self, **kwargs):
        prompt = (f"Summarize the {self.type_name}: {self.name} with up to {self.summary_len} sentences "
                  f"given the following source code:\n\n{self.source}\n"
                  f"Your answer:\n\n")

        summary = self.ask(prompt, system=False, **kwargs).text
        # put the summary in the queue
        self._summary_queue.put(summary)

    def ask(self, query, system=True, **kwargs):
        if system:
            query = f"{self.system_prompt}\n{query}"
        else:
            query = f"{query}"
        return self.llm.ask(query, **kwargs)

    @cached_property
    def system_prompt(self):
        return (f"Your job is to help a programmer to execute a python code from natural language queries.\n"
                f"You are given a {self.type_name} named {self.name} with the following description:\n"
                f"{self.summary}\n")

    def init_instance(self, query, ask_kwargs=None, user_kwargs=None, eval_arguments=None):

        eval_arguments = eval_arguments or self.eval_arguments
        user_kwargs = user_kwargs or {}

        # create an instance of the class and return a NLPDispatcher object
        def is_class_method(member):
            # First, ensure that member is a method bound to a class
            if inspect.ismethod(member) and inspect.isclass(member.__self__):
                # Now that we've confirmed member is a method, check the name conditions
                if not member.__name__.startswith('_') and member.__name__ != 'from_nlp':
                    return True
            return False

        classmethods = [name for name, member in inspect.getmembers(self.real_object, predicate=is_class_method)]

        example_output = {'method': 'method_name'}
        query = (f"Choose the best classmethod that should be used to build a class instance according to the "
                  f"following query:\n"
                  f"Query: {query}\n"
                  f"Methods: {classmethods}\n"
                  f"Return your answer as a JSON object of the following form:\n"
                  f"{json.dumps(example_output)}\n"
                  f"Your answer:\n\n")

        ask_kwargs = ask_kwargs or {}
        response = self.ask(query, guidance=self.guidance['method_selection'], **ask_kwargs).json

        constructor_name = response['method']

        if constructor_name not in classmethods:
            raise ValueError(f"Constructor {constructor_name} not found in the list of class constructors")

        constructor = getattr(self.real_object, constructor_name)
        logger.info(f"Using classmethod {constructor_name} to build the class instance")

        constructor_sourcecode = inspect.getsource(constructor)
        init_sourcecode = inspect.getsource(self.real_object.__init__)

        json_output_example = {"args": ['arg1', 'arg2'], "kwargs": {'kwarg1': 'value1', 'kwarg2': 'value2'}}
        query = (f"Build a suitable dictionary of arguments and keyword arguments to build a class instance according "
                  f"to the following query:\n"
                  f"Query: {query}\n"
                  f"with the classmethod: {constructor_name} (of class {self.name}) with source-code:\n"
                  f"{constructor_sourcecode}\n"
                  f"and the class __init__ method source-code:\n"
                  f"{init_sourcecode}\n"
                  f"Return your answer as a JSON object of the following form:\n"
                  f"{json_output_example}\n"
                  f"Your answer:\n\n")

        d = self.ask(query, guidance=self.guidance['arguments_selection'], **ask_kwargs).json
        args = d.get('args', [])
        kwargs = d.get('kwargs', {})

        logger.info(f"Using args: {args} and kwargs: {kwargs} to build the class instance")

        instance = constructor(*args, **kwargs)

        return BeamAssistant(instance, llm=self.llm, summary_len=self.summary_len, _summary=self.summary)

    def exec_method(self, query, method_name=None, ask_kwargs=None, user_kwargs=None, eval_arguments=None,
                    execute=True):

        eval_arguments = eval_arguments or self.eval_arguments

        assert method_name is not None or self.type == 'function', \
            'method_name must be provided for non-function objects'

        if method_name is None:
            method_description = f"function: {self.name}"
        else:
            method_description = f"class method: {method_name} (of class {self.name})"

        ask_kwargs = ask_kwargs or {}
        user_kwargs = user_kwargs or {}
        args_hint = ''
        if user_kwargs:
            args_hint = (f"These are arguments that the user want to pass to the method "
                         f"(their names represent their meaning but they do not necessarily "
                         f"match to the method signature):\n"
                         f"{self.arguments_repr(**user_kwargs)}\n")

        if method_name is not None:
            method = getattr(self.real_object, method_name)
        else:
            method = self.real_object

        sourcecode = inspect.getsource(method)

        json_output_example = {"args": [2, 'some_str'],
                               "kwargs": {'k1': 3.5, 'k2': None}}

        if eval_arguments:
            json_output_example["args"].append('<eval>np.arange(100)</eval>')
            json_output_example["kwargs"]['ke'] = "<eval>torch.randn(20)</eval>"

        if args_hint:
            json_output_example["args"].append('<user_arg>arg_name</user_arg>')
            json_output_example["kwargs"]['ku'] = "<user_kwarg>arg_name</user_kwarg>"

        i = 1
        possible_argument_types = ("As argument, you can use the following types:\n"
                                   "1. JSON serializable objects\n"
                                   )

        if eval_arguments:
            i += 1
            possible_argument_types += (f"{i}. python statements (wrapped with <eval></eval> tags). "
                                        "In your statement you can use the following python packages: "
                                        "(1) numpy as np (2) torch (3) pandas as pd\n")

        if args_hint:
            i += 1
            possible_argument_types += f"{i}. User-defined arguments (wrapped with <user_arg></user_arg> tags)\n"

        prompt = (f"Build a suitable dictionary of arguments and keyword arguments to answer the following query:\n"
                  f"Query: {query}\n"
                  f"with the {method_description} with source-code:\n"
                  f"{sourcecode}\n"
                  f"{args_hint}"
                  f"{possible_argument_types}\n"
                  f"Return your answer as a JSON object of the following form:\n"
                  f"{json.dumps(json_output_example)}\n"
                  f"Your answer:\n\n")

        d = self.ask(prompt, guidance=self.guidance['arguments_selection'], **ask_kwargs).json

        args = d.get('args', [])
        kwargs = d.get('kwargs', {})

        # iterate over args and kwargs and look for tags <eval> and <user_arg>
        # use re to match the pattern <tag>content</tag>

        if not execute:
            return Namespace(args=args, kwargs=kwargs, method=method)

        eval_pattern = re.compile(r"<eval>(.*)</eval>")
        user_arg_pattern = re.compile(r"<user_arg>(.*)</user_arg>")

        def process_value(v):

            if not isinstance(v, str):
                return v

            eval_match = eval_pattern.match(v)
            user_arg_match = user_arg_pattern.match(v)
            if eval_match:
                # extract the content and evaluate it
                try:
                    v = eval(eval_match.group(1))
                except:
                    raise ValueError(f"Failed to evaluate the expression: {eval_match.group(1)}")
            elif user_arg_match:
                # extract the content and use it as a user-defined argument
                v = user_arg_match.group(1)
                v = user_kwargs[v]['value']

            return v

        logger.info(f"Using args: {args} and kwargs: {kwargs} to answer the query")

        for k, v in kwargs.items():
            kwargs[k] = process_value(v)

        for i, v in enumerate(args):
            args[i] = process_value(v)

        return method(*args, **kwargs)

    @staticmethod
    def arguments_repr(**kwargs):
        args_repr = ''
        for k, v in kwargs.items():
            args_repr += f"{k}: {v['str']} (type: {type(v['value'])}, {v['metadata']})\n"
        return args_repr

    def choose_method(self, query, ask_kwargs=None, user_kwargs=None):

        ask_kwargs = ask_kwargs or {}
        user_kwargs = user_kwargs or {}
        args_hint = ''
        if user_kwargs:
            args_hint = (f"These are arguments that the user want to pass to the method "
                         f"(their names do not necessarily match to the method signature):\n"
                         f"{self.arguments_repr(**user_kwargs)}\n")

        method_list = safe_getmembers(self.real_object, predicate=inspect.isroutine)
        json_output_example = json.dumps({'method': 'method_name'})
        class_doc = inspect.getdoc(self)
        class_doc = f"{class_doc}\n" if class_doc else ""

        prompt = (f"Choose the suitable method that should be used to answer the following query:\n"
                  f"Query: {query}\n"
                  f"Class: {self.__class__.__name__}\n"
                  f"{class_doc}"
                  f"Attributes: {method_list}\n"
                  f"{args_hint}"
                  f"Return your answer as a JSON object, see example below:\n"
                  f"{json_output_example}\n"
                  f"Your answer:\n\n")

        response = self.ask(prompt, guidance=self.guidance['method_selection'], **ask_kwargs).json
        method_name = response['method']

        if method_name not in [m[0] for m in method_list]:
            raise ValueError(f"Method {method_name} not found in the list of methods")

        logger.info(f"Using method {method_name} to answer the query")

        return method_name

    def exec(self, query, method=None, ask_kwargs=None, eval_arguments=None, execute=None, **kwargs):
        # execute code according to the prompt

        if execute is None:
            execute = self.execute

        user_kwargs = {k: {'value': v, 'str': str(v), 'metadata': check_type(v)} for k, v in kwargs.items()}

        if self.type == 'class':
            # create an instance of the class
            res = self.init_instance(query, ask_kwargs=ask_kwargs, user_kwargs=user_kwargs, eval_arguments=eval_arguments)
        elif self.type == 'instance':
            if method is None:
                method = self.choose_method(query, ask_kwargs=ask_kwargs, user_kwargs=user_kwargs)
            res = self.exec_method(query, method_name=method, ask_kwargs=ask_kwargs, execute=execute,
                                   user_kwargs=user_kwargs, eval_arguments=eval_arguments)
        elif self.type == 'function':
            res = self.exec_method(query, ask_kwargs=ask_kwargs, execute=execute,
                                   user_kwargs=user_kwargs, eval_arguments=eval_arguments)
        else:
            raise ValueError(f"Unknown type: {self.type}")
        return res

    def chat(self, query):
        suggestion = self.ask(query).text
        return suggestion

    def gen_code(self, query):
        code = self.ask(
            f"Return an executable python code that performs the following task:\n"
            f"{query}\\n"
            f"Use a markdown syntax, i.e. the tags \'\'\'python and \'\'\' at the beginning and the end of the code "
            f"section. "
            f"The final result should be assigned to a variable name \'result\'.\n"
            f"Your answer:\n\n").text
        # use re to extract the code
        code = re.search(r"\'\'\'python(.*)\'\'\'", code, re.DOTALL).group(1)
        return code

    def exec_eval(self, query):
        code = self.gen_code(query)
        try:
            exec(code, globals(), locals())
            return result
        except:
            raise ValueError(f"Failed to execute the code:\n{code}")
