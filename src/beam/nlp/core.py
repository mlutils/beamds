from .. import resource
from ..core import MetaDispatcher
from ..utils import lazy_property
import inspect


class NLPDispatcher(MetaDispatcher):

    def __init__(self, obj, *args, llm=None, llm_kwargs=None, summary_len=10, **kwargs):

        super().__init__(obj, *args, summary_len=summary_len, **kwargs)

        self.summary_len = self.get_hparam('summary_len', summary_len)
        llm = self.get_hparam('llm', llm)
        llm_kwargs = llm_kwargs or {}

        self.llm = None
        if llm is not None:
            self.llm = resource(llm, **llm_kwargs)

    @lazy_property
    def doc(self):
        return self.obj.__doc__

    @lazy_property
    def source(self):
        if self.type == 'class':
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
        prompt = (f"Summarize the {self.type_name} {self.name} with {self.summary_len} sentences "
                  f"given the following source code:\n\n{self.source}")

        self.llm.ask(prompt, **kwargs)


    # def call(self, prompt, **kwargs):

    def do(self, prompt, method=None, **kwargs):
        # execute code according to the prompt
        pass

    def chat(self, prompt):
        # suggest me how to use the object but dont really use it
        pass

