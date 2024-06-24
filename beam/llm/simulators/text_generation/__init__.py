from argparse import Namespace
from ...resource import beam_llm


class Client:
    def __init__(self, model=None, **kwargs):
        self.llm = beam_llm(model, **kwargs) if type(model) == str else model

    def generate(self, request, **kwargs):

        if 'stop_sequences' in kwargs:
            kwargs['stop'] = kwargs.pop('stop_sequences')

        text = self.llm.ask(request, **kwargs).text
        return Namespace(generated_text=text)
