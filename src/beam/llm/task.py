

class LLMTask:

    def __init__(self, name=None, description=None, system=None, input_kwargs=None, output_kwargs=None,
                 input_format='.json', output_format='.json', sep='\n', **kwargs):

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

    def prompt(self, **kwargs):
        message = f""
        return message

    def parse(self, response, **kwargs):
        return response


class LLMTool:
    pass