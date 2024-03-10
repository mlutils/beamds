import json

from .. import resource
from ..core import MetaDispatcher
from ..utils import lazy_property
from ..logger import beam_logger as logger
import inspect
import threading
import queue
import re


class NLPDispatcher(MetaDispatcher):

    def __init__(self, obj, *args, llm=None, llm_kwargs=None, summary_len=10, _summary=None, **kwargs):

        super().__init__(obj, *args, summary_len=summary_len, **kwargs)

        self.summary_len = self.get_hparam('summary_len', summary_len)
        llm = self.get_hparam('llm', llm)
        llm_kwargs = llm_kwargs or {}

        self.llm = None
        if llm is not None:
            self.llm = resource(llm, **llm_kwargs)

        self.summary_queue = queue.Queue()
        self.summarize_thread = None
        if _summary is not None:
            self.summary_queue.put(_summary)
        else:
            # summarize the object in a different thread
            self.summarize_thread = threading.Thread(target=self.summarize, daemon=True)
            self.summarize_thread.start()

    @lazy_property
    def summary(self):
        if self.summarize_thread is not None:
            # wait for the thread to finish and get the summary from the queue
            self.summarize_thread.join()
        return self.summary_queue.get()

    @lazy_property
    def doc(self):
        return self.obj.__doc__

    @lazy_property
    def source(self):
        if self.type in ['class', 'instance']:
            # iterate over all parent classes and get the source
            sources = []
            for cls in inspect.getmro(self.obj):
                if cls.__module__ != 'builtins':
                    sources.append(inspect.getsource(cls))
            # sum all the sources
            return '\n'.join(sources)
        else:
            return inspect.getsource(self.obj)

    @property
    def name(self):
        return self.obj.__name__

    @lazy_property
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

        summary = self.ask(prompt, **kwargs).text
        # put the summary in the queue
        self.summary_queue.put(summary)

    def ask(self, query, **kwargs):
        query = f"{self.system_prompt}\n{query}"
        return self.llm.ask(query, **kwargs)

    @lazy_property
    def system_prompt(self):
        return (f"Your job is to help a programmer to execute a python code from natural language queries.\n"
                f"You are given a {self.type_name} named {self.name} with the following description:\n"
                f"{self.summary}\n")

    def init_instance(self, query, ask_kwargs=None):

        # create an instance of the class and return a NLPDispatcher object
        def is_class_method(member):
            # First, ensure that member is a method bound to a class
            if inspect.ismethod(member) and inspect.isclass(member.__self__):
                # Now that we've confirmed member is a method, check the name conditions
                if not member.__name__.startswith('_') and member.__name__ != 'from_nlp':
                    return True
            return False

        classmethods = [name for name, member in inspect.getmembers(self.obj, predicate=is_class_method)]

        example_output = {'method': 'method_name'}
        query = (f"Choose the suitable classmethod that should be used to build a class instance according to the "
                  f"following query:\n"
                  f"Query: {query}\n"
                  f"Methods: {classmethods}\n"
                  f"Return your answer as a JSON object of the following form:\n"
                  f"{json.dumps(example_output)}\n"
                  f"Your answer:\n\n")

        ask_kwargs = ask_kwargs or {}
        response = self.ask(query, **ask_kwargs).json

        constructor_name = response['method']

        if constructor_name not in classmethods:
            raise ValueError(f"Constructor {constructor_name} not found in the list of class constructors")

        constructor = getattr(self.obj, constructor_name)
        logger.info(f"Using classmethod {constructor_name} to build the class instance")

        constructor_sourcecode = inspect.getsource(constructor)
        init_sourcecode = inspect.getsource(self.obj.__init__)

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

        d = self.ask(query, **ask_kwargs).json
        args = d.get('args', [])
        kwargs = d.get('kwargs', {})

        logger.info(f"Using args: {args} and kwargs: {kwargs} to build the class instance")

        instance = constructor(*args, **kwargs)

        return NLPDispatcher(instance, llm=self.llm, summary_len=self.summary_len, _summary=self.summary)

    def exec_function(self, query, ask_kwargs=None):

        ask_kwargs = ask_kwargs or {}

        json_output_example = {"args": ['arg1', 'arg2'], "kwargs": {'kwarg1': 'value1', 'kwarg2': 'value2'}}

        prompt = (f"Build a suitable dictionary of arguments and keyword arguments to answer the following query:\n"
                  f"Query: {query}\n"
                  f"with the function: {self.name} with source-code:\n"
                  f"{self.source}\n"
                  f"Return your answer as a JSON object of the following form:\n"
                  f"{json_output_example}\n"
                  f"Your answer:\n\n")

        d = self.ask(prompt, **ask_kwargs).json

        args = d.get('args', [])
        kwargs = d.get('kwargs', {})

        logger.info(f"Using args: {args} and kwargs: {kwargs} to answer the query")

        return self.obj(*args, **kwargs)

    def exec_method(self, query, method_name, ask_kwargs=None):

        ask_kwargs = ask_kwargs or {}

        method = getattr(self, method_name)
        sourcecode = inspect.getsource(method)

        json_output_example = {"args": ['arg1', 'arg2'], "kwargs": {'kwarg1': 'value1', 'kwarg2': 'value2'}}

        prompt = (f"Build a suitable dictionary of arguments and keyword arguments to answer the following query:\n"
                  f"Query: {query}\n"
                  f"with the class method: {method_name} (of class {self.__class__.__name__}) with source-code:\n"
                  f"{sourcecode}\n"
                  f"Return your answer as a JSON object of the following form:\n"
                  f"{json_output_example}\n"
                  f"Your answer:\n\n")

        d = self.ask(prompt, **ask_kwargs).json

        args = d.get('args', [])
        kwargs = d.get('kwargs', {})

        logger.info(f"Using args: {args} and kwargs: {kwargs} to answer the query")

        return method(*args, **kwargs)

    def choose_method(self, query, ask_kwargs=None):

        ask_kwargs = ask_kwargs or {}

        method_list = inspect.getmembers(self, predicate=inspect.isroutine)
        method_list = [m for m in method_list if not m[0].startswith('_')]
        json_output_example = json.dumps({'method': 'method_name'})
        class_doc = inspect.getdoc(self)
        class_doc = f"{class_doc}\n" if class_doc else ""

        prompt = (f"Choose the suitable method that should be used to answer the following query:\n"
                  f"Query: {query}\n"
                  f"Class: {self.__class__.__name__}\n"
                  f"{class_doc}"
                  f"Attributes: {method_list}\n"
                  f"Return your answer as a JSON object of the following form:\n"
                  f"{json_output_example}\n"
                  f"Your answer:\n\n")

        response = self.ask(prompt, **ask_kwargs).json
        method_name = response['method']

        if method_name not in [m[0] for m in method_list]:
            raise ValueError(f"Method {method_name} not found in the list of methods")

        logger.info(f"Using method {method_name} to answer the query")

        return method_name

    def do(self, query, method=None, ask_kwargs=None):
        # execute code according to the prompt
        if self.type == 'class':
            # create an instance of the class
            res = self.init_instance(query, ask_kwargs=ask_kwargs)
        elif self.type == 'instance':
            if method is None:
                method = self.choose_method(query, ask_kwargs=ask_kwargs)
            res = self.exec_method(query, method=method, ask_kwargs=ask_kwargs)
        elif self.type == 'function':
            res = self.exec_function(query, ask_kwargs=ask_kwargs)
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
            f"Use the tags [PYTHON] and [\PYTHON] at the beginning and the end of the code section. "
            f"The final result should be assigned to a variable name \'result\'.\n"
            f"Your answer:\n\n").text
        # use re to extract the code
        code = re.search(r"\[PYTHON\](.*)\[\\PYTHON\]", code, re.DOTALL).group(1)
        return code

    def exec(self, query):
        code = self.gen_code(query)
        try:
            exec(code, globals(), locals())
            return result
        except:
            raise ValueError(f"Failed to execute the code:\n{code}")


