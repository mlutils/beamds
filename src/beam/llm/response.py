import re
import time
import uuid

from ..logger import beam_logger as logger
from ..utils import parse_text_to_protocol, retry


class LLMResponse:
    def __init__(self, response, llm, prompt=None, prompt_kwargs=None, chat=False, stream=False, parse_retrials=3,
                 sleep=1, prompt_type='completion', **kwargs):
        self.response = response
        self._prompt = prompt
        self._prompt_kwargs = prompt_kwargs
        self.parse_retrials = parse_retrials
        self.sleep = sleep
        self.llm = llm
        self.id = f'beamllm-{uuid.uuid4()}'
        self.model = llm.model
        self.created = int(time.time())
        self.chat = chat
        self.object = "chat.completion" if chat else "text_completion"
        self.stream = stream
        self.prompt_type = prompt_type
        self._task_result = None
        self._task_success = None
        assert self.verify(), "Response is not valid"

    def __str__(self):
        return self.text

    def __bool__(self):
        return self.bool

    def __int__(self):
        return self.int

    def __float__(self):
        return self.float

    def __iter__(self):
        if not self.stream:
            yield self
        else:
            for r in self.response:
                yield LLMResponse(r, self.llm, prompt=self.prompt, chat=self.chat, stream=self.stream,
                                  prompt_kwargs=self._prompt_kwargs, parse_retrials=self.parse_retrials,
                                  sleep=self.sleep, prompt_type=self.prompt_type)

    def add_task_result(self, task_result, success=True):
        self._task_result = task_result
        self._task_success = success

    @property
    def int(self):
        try:
            return int(self.text)
        except:
            return None

    @property
    def float(self):
        try:
            return float(self.text)
        except:
            return None

    @property
    def task_result(self):
        return self._task_result

    @property
    def task_success(self):
        return self._task_success

    @property
    def prompt(self):
        return self._prompt

    @property
    def prompt_kwargs(self):
        return self._prompt_kwargs

    def verify(self):
        return self.llm.verify_response(self)

    @property
    def text(self):
        return self.llm.extract_text(self)

    @property
    def openai_format(self):
        return self.llm.openai_format(self)

    def parse_text(self, text, protocol='json'):
        return self._protocol(text, protocol=protocol)

    def parse(self, protocol='json'):
        return self._protocol(self.text, protocol=protocol)

    def _protocol(self, text, protocol='json'):

        if self.parse_retrials == 0:
            return parse_text_to_protocol(text, protocol=protocol)
        try:
            return parse_text_to_protocol(text, protocol=protocol)
        except:
            retry_protocol = (retry(retrials=self.parse_retrials, sleep=self.sleep,  logger=logger, name=f"fix-{protocol} with {self.model}")
                              (self.llm.fix_protocol))
            return retry_protocol(text, protocol=protocol)

    @property
    def judge(self):
        # boolean judgement of response by LLM for the prompt

        if hasattr(self.llm, 'judge'):
            return self.llm.judge(self)

        judge_prompt = (f"Evaluate the following response by an LLM for accuracy and relevance to the given prompt.\n"
                        f"Original Prompt: {self.prompt}\n"
                        f"LLM Response: {self.text}\n"
                        f"Based on the factual correctness, relevance to the original prompt, and logical consistency, "
                        f"is the provided response accurate? "
                        f"Respond with 'True' for accurate or 'False' for inaccurate.")

        res = self.llm.ask(judge_prompt)

        return res.bool

    @property
    def bool(self):

        text = re.findall(r'\w+', self.text.lower())
        n_words = len(text)

        if 'true' in text:
            return True
        elif 'false' in text:
            return False
        elif 'yes' in text:
            return True
        elif 'no' in text and n_words == 1:
            return False
        return None

    @property
    def json(self):
        json_text = self.llm.extract_text(self)
        json_text = json_text.replace(r'/_', '_')
        json_text = json_text.replace('False', 'false')
        json_text = json_text.replace('True', 'true')
        return self._protocol(json_text, protocol='json')

    @property
    def html(self):
        text = self.llm.extract_text(self)
        return self._protocol(text, protocol='html')

    @property
    def xml(self):
        text = self.llm.extract_text(self)
        return self._protocol(text, protocol='xml')

    @property
    def csv(self):
        text = self.llm.extract_text(self)
        return self._protocol(text, protocol='csv')

    @property
    def yaml(self):
        text = self.llm.extract_text(self)
        # text = re.search('yaml\n([\s\S]*?)\n', text).group(1)
        return self._protocol(text, protocol='yaml')

    @property
    def toml(self):
        text = self.llm.extract_text(self)
        return self._protocol(text, protocol='toml')

    @property
    def choices(self):
        return self.llm.extract_choices(self.response)
