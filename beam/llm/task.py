from ..utils import jupyter_like_traceback, cached_property
from .resource import beam_llm
from ..processor import Processor


class LLMTask(Processor):

    def __init__(self, name=None, description=None, system=None, input_kwargs=None, output_kwargs=None,
                 output_format='json', sep='\n', llm=None, *args, **kwargs):

        super().__init__(*args, name=name, **kwargs)
        self.name = name
        self.description = description
        if input_kwargs is None:
            input_kwargs = {}
        if output_kwargs is None:
            output_kwargs = {}
        self.input_kwargs = input_kwargs
        self.output_kwargs = output_kwargs
        self.output_format = output_format
        self.sep = sep
        self.system = system
        self._llm = llm

    @cached_property
    def llm(self):
        llm = beam_llm(self._llm)
        return llm

    def set_llm(self, value):
        self.clear_cache('llm')
        self._llm = value

    def prompt(self, **kwargs):
        message = (f"System message: {self.system}\n"
                   f"Task: {self.name}\n"
                   f"Input arguments description: {self.input_kwargs}\n"
                   f"Task description: {self.description}\n"
                   f"Input arguments values: {kwargs}\n"
                   f"Output arguments description: {self.output_kwargs}\n"
                   f"Output format: {self.output_format}\n"
                   f"Your answer should start here:\n\n")
        return message

    def __call__(self, llm_kwargs=None, **kwargs):

        if llm_kwargs is None:
            llm_kwargs = {}
        prompt = self.prompt(**kwargs)

        res = self.llm.ask(prompt, **llm_kwargs)

        try:
            result = res.parse(protocol=self.output_format)
            success = True
        except:
            result = jupyter_like_traceback()
            success = False

        res.add_task_result(result, success=success)

        return res

    def parse(self, response):
        raise NotImplementedError


