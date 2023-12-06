import json
from ..utils import lazy_property
from dataclasses import dataclass
from .resource import beam_llm
from ..core import Processor


class LLMTask(Processor):

    def __init__(self, name=None, description=None, system=None, input_kwargs=None, output_kwargs=None,
                 input_format='.json', output_format='.json', sep='\n', llm=None, *args, **kwargs):

        super().__init__(*args, name=name, **kwargs)
        self.name = name
        self.description = description
        if input_kwargs is None:
            input_kwargs = {}
        if output_kwargs is None:
            output_kwargs = {}
        self.input_kwargs = input_kwargs
        self.output_kwargs = output_kwargs
        self.input_format = input_format
        self.output_format = output_format
        self.sep = sep
        self.system = system
        self._llm = llm

    @lazy_property
    def llm(self):
        llm = beam_llm(self._llm)
        return llm

    @llm.setter
    def llm(self, value):
        self.clear_cache('llm')
        self._llm = value

    def __str__(self, **kwargs):
        message = f""
        return message

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def parse(self, response):
        raise NotImplementedError


@dataclass
class LLMToolParameter:
    # a class that holds an OpenAI tool parameter object

    name: str
    type: str
    description: str
    default: str
    choices: list = None
    required: bool = False

    def __str__(self):
        message = json.dumps(repr(self), indent=4)
        return message

    def __repr__(self):
        obj = {'name': self.name,
                              'type': self.type,
                              'description': self.description,
                              'default': self.default,
                              'choices': self.choices,
                              'required': self.required}
        return obj


class LLMTool:
    # a class that holds an OpenAI tool object

    def __init__(self, name=None, type='function', description=None, func=None, required=None, **kwargs):

        self.name = name or 'func'
        self.type = type
        self.description = description or 'See properties for more information.'
        self.func = func
        self.parameters = kwargs
        self.required = required or []

    def __str__(self):
        message = json.dumps({'type': self.type,
                              self.type: {'name': self.name,
                                          'description': self.description,
                                          'parameters': self.parameters,
                                          'required': self.required}}, indent=4)
        return message

