from .processor import Processor
from .path import beam_path
from .llm import beam_llm
import json


class HexBeam(Processor):
    # this class is used to analyze, explore and research Ghidra exported data

    def __init__(self, analysis, llm, description=None, **kwargs):
        super().__init__(**kwargs)
        self.analysis = analysis
        self.llm = beam_llm(llm)
        self._functions_map = None
        self.description = description

    @classmethod
    def from_analysis_path(cls, path, llm, **kwargs):
        analysis = beam_path(path).read()
        return cls(analysis, llm, **kwargs)

    @property
    def functions_map(self):
        if self._functions_map is None:
            self._functions_map = {v['name']: k for k, v in enumerate(self.analysis['functions'])}
        return self._functions_map

    @property
    def system_messages(self):
        message = (f"You are an expert cyber security researcher. "
                   f"You analyze a {self.analysis['metadata']['architecture']} architecture program.")

        if self.description is not None:
            message = f"{message}\nA high level description of the program: {self.description}."

        return message

    def get_function_assembly(self, function_name):
        i = self.functions_map[function_name]
        function_info = self.analysis['functions'][i]

        calls = {self.analysis['functions'][self.functions_map[f]]['start_address']: f
                 for f in function_info['calls']}

        def replace(l):
            for k, v in calls.items():
                l = l.replace(f"0x{k}", v)
            return l

        text = '\n\t'.join([replace(l) for l in function_info['assembly']])

        text = f"0x{function_info['start_address']}:\n\t{text}\n0x{function_info['end_address']}:"
        return text

    def decompile_function(self, analysis):
        message = (f"{self.system_messages}\n\n"
                   f"Decompile the following {self.analysis['metadata']['architecture']} function: \n\n"
                   f"Function name: {analysis['metadata']['name']}\n"
                   f"Docstring: {analysis['metadata']['doc']}\n"
                   f"========================================================================\n\n"
                   f"{analysis['assembly']} \n"
                   f"========================================================================\n\n"
                   f"Return decompiled version of the function into C++ language, containing the input and output variable names\n")

        return self.llm.ask(message,).text

    def analyze_function(self, function_name, save=True):

        analysis = self._analyze_function(function_name)
        decompiled = self.decompile_function(analysis)
        analysis = {'assembly': analysis['assembly'], 'metadata': analysis['metadata'], 'decompiled': decompiled}

        if save:
            i = self.functions_map[function_name]
            self.analysis['functions'][i]['analysis'] = analysis

        return analysis

    def _analyze_function(self, function_name):

        message = (f"{self.system_messages}\n\n"
                   f"Analyze the following {self.analysis['metadata']['architecture']} function: \n\n"
                   f"Function name: {function_name}\n"
                   f"========================================================================\n\n"
                   f"{self.get_function_assembly(function_name)} \n"
                   f"========================================================================\n\n"
                   f"First task: \n"
                   f"Rewrite the assembly code with modified and informative variable and function names.\n"
                   f"In addition, you can write comments in the assembly to explain complex code parts.\n"
                   f"========================================================================\n\n"
                   f"Second task: \n"
                   f"Return a valid json object containing the following kes: [name, doc, variables, functions]\n\n"
                   f"name: a modified name which describes the function purpose \n"
                   f"doc: a full docstring of the function, containing the input and output variable names\n"
                   f"and the description of the function.\n"
                   f"variables: a dictionary mapping of the new variable names to the old names\n"
                   f"functions: a dictionary mapping of the new functions names to the old names\n"
                   f"========================================================================\n\n"
                   f"Don't add anything else to the answer, don't add special characters.\n"
                   f"Your response structure should be as follows:\n\n"
                   f"assembly code\n"
                   f"```\n"
                   f"{{Your answer to task 1 here}}\n"
                   f"```\n"
                   f"JSON object\n"
                   f"```\n"
                   f"{{Your answer to task 2 here}}\n"
                   )

        res = self.llm.ask(message,).text
        res = res.split('```')
        res = {'assembly': res[1], 'metadata': json.loads(res[3])}

        return res
