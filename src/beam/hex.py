from .processor import Processor
from .path import beam_path
from .llm import beam_llm
import json
from .logger import beam_logger as logger


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

    @staticmethod
    def choose_better_name(name1, name2):
        if name1 is None:
            return name2
        if name2 is None:
            return name1

        alpha_1 = sum([c.isalpha() for c in name1])
        alpha_2 = sum([c.isalpha() for c in name2])

        return name1 if alpha_1 > alpha_2 else name2

    def assess_vulnerabilities(self, function_name, save=True):

        i = self.functions_map[function_name]

        if 'analysis' not in self.analysis['functions'][i]:
            logger.info(f"Analyzing function: {function_name}")
            self.analyze_function(function_name, save=True)
            logger.info(f"The function docstring: {self.analysis['functions'][i]['analysis']['metadata']['doc']}")

        message = (f"{self.system_messages}\n\n"
                   f"You are given the following {self.analysis['metadata']['architecture']} function:\n"
                   f"Function name: {function_name}\n"
                   f"Docstring: {self.analysis['functions'][i]['analysis']['metadata']['doc']}\n"
                   f"========================================================================\n\n"
                   f"{self.analysis['functions'][i]['analysis']['assembly']} \n"
                   f"========================================================================\n\n"
                   f"Your task is to assess possible vulnerabilities in the function that can be exploited\n"
                   f"by an attacker. You can assume that the attacker has full access to the function's input\n"
                   f"You need to classify the risk of each vulnerability as [very-low, low, medium, high, very-high]\n"
                   f"========================================================================\n\n"
                   f"Your response should be a valid JSON object with the list of detected vulnerabilities,\n"
                   f"where each vulnerability is a dictionary with the following keys: [name, risk, explanation]\n")

        res = self.llm.ask(message).text
        res = json.loads(res)
        return res

    def decompile_function(self, analysis):
        message = (f"{self.system_messages}\n\n"
                   f"Decompile the following {self.analysis['metadata']['architecture']} function: \n\n"
                   f"Function name: {analysis['metadata']['name']}\n"
                   f"Docstring: {analysis['metadata']['doc']}\n"
                   f"========================================================================\n\n"
                   f"{analysis['assembly']} \n"
                   f"========================================================================\n\n"
                   f"Return decompiled version of the function into C++ language, containing the input and output variable names\n"
                   f"Your answer should contain only valid C++ code. Explanations as comments are allowed\n")


        return self.llm.ask(message,).text

    def analyze_function(self, function_name, save=True):

        analysis = self._analyze_function(function_name)
        decompiled = self.decompile_function(analysis)
        analysis = {'assembly': analysis['assembly'], 'metadata': analysis['metadata'], 'decompiled': decompiled}

        if save:
            i = self.functions_map[function_name]
            self.analysis['functions'][i]['analysis'] = analysis

        return analysis

    def research_question(self, question, budget=10, entry_point='entry', entry_point_alt_name=None):

        states = [{'previous_function': None, 'previous_answer': None, 'current_function': entry_point,
                 'current_function_alt_name': entry_point_alt_name, 'previous_function_alt_name': None}]

        message = (f"{self.system_messages}\n\n"
                   f"You are given the following research question: {question}\n"
                   f"========================================================================\n\n"
                   f"Your task is to explore the code step by step iteratively and find the answer to the question.\n"
                   f"In each step you are given a function in the program and your previous answer to the question.\n"
                   f"In each step you need to modified your answer according to the provided function and to choose the next function to explore.\n"
                   f"The next function can be chosen from the list of calls in the current function"
                   f"alternatively you can choose to return to the previous function that called the current one or to the entry point"
                   f"Your number-of-steps budget is limited to {{budget}} so you need to carefully design your steps."
                   f"If you think you are done, you need to raise a flag is_done=True, else is_done should be False\n"
                   f"========================================================================\n\n"
                   f"The current state of your research is:\n"
                   f"previous function: {{previous_function}}\n"
                   f"current function: {{current_function}}\n"
                   f"previous answer: {{previous_answer}}\n"
                   f"current function docstring: {{current_function_docstring}}\n"
                   f"current function assembly: {{current_function_assembly}}\n"
                   f"========================================================================\n\n"
                   f"Your response should be a valid JSON object with the following keys: "
                   f"[explanation, next_step, is_done]\n"
                   f"explanation: a string containing your answer to the research question."
                   f"In addition you can add information about relevant functions and variables which may help your"
                   f"decision process in the next steps. You are also advised to enter the list of functions you have"
                   f"already visited to avoid duplicate visits in the same function.\n"
                   f"next_step: a string containing the name of the next function to explore "
                   f"(use entry to return to the entry point and previous to return to the previous function).\n"
                   f"is_done: a boolean value indicating if you are done with the research\n")

        logger.info(f"Starting analysis to answer the research question: {question}. The entry point is: {entry_point}")

        is_done = False
        while budget > 0 and not is_done:

            logger.info(f"Current budget: {budget}")
            logger.info(f"Current function: {states[-1]['current_function']}")
            logger.info(f"Current answer: {states[-1]['previous_answer']}")

            state = states[-1]
            i = self.functions_map[state['current_function']]
            current_function_name = HexBeam.choose_better_name(state['current_function'],
                                                               state['current_function_alt_name'])
            previous_function_name = HexBeam.choose_better_name(state['previous_function'],
                                                                state['previous_function_alt_name'])

            if 'analysis' not in self.analysis['functions'][i]:
                logger.info(f"Analyzing function: {current_function_name}")
                self.analyze_function(current_function_name, save=True)
                logger.info(f"The function docstring: {self.analysis['functions'][i]['analysis']['metadata']['doc']}")

            current_function_docstring = self.analysis['functions'][i]['analysis']['metadata']['doc']
            current_function_assembly = self.analysis['functions'][i]['analysis']['assembly']

            current_mesage = message.format(previous_function=previous_function_name,
                                            current_function=current_function_name,
                                            previous_answer=state['previous_answer'],
                                            current_function_docstring=current_function_docstring,
                                            current_function_assembly=current_function_assembly,
                                            budget=budget)

            res = self.llm.ask(current_mesage,).text
            res = json.loads(res)

            next_function = res['next_step']
            if next_function in ['entry', 'Entry', 'ENTRY']:
                current_function = entry_point
                entry_point_alt_name = entry_point_alt_name
            elif next_function in ['previous', 'Previous', 'PREVIOUS']:
                current_function = state['previous_function']
                current_function_alt_name = state['previous_function_alt_name']
            elif next_function in self.analysis['functions'][i]['analysis']['metadata']['functions']:
                current_function_alt_name = next_function
                current_function = self.analysis['functions'][i]['analysis']['metadata']['functions'][next_function]
            else:
                current_function_alt_name = next_function
                current_function = next_function

            is_done = res['is_done']
            budget -= 1
            states.append({'previous_function': state['current_function'],
                            'previous_function_alt_name': state['current_function_alt_name'],
                            'previous_answer': res['explanation'],
                            'current_function': current_function,
                            'current_function_alt_name': current_function_alt_name})

        return states

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
                   f"Your response structure should be as follows:\n\n"
                   f"assembly code\n"
                   f"```\n"
                   f"{{Your answer to task 1 here: only valid assembly code}}\n"
                   f"```\n"
                   f"JSON object\n"
                   f"```\n"
                   f"{{Your answer to task 2 here: only valid json object without any special characters}}\n"
                   )

        res = self.llm.ask(message,).text
        res = res.split('```')
        res = {'assembly': res[1], 'metadata': json.loads(res[3])}

        return res
