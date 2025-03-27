from pydantic import BaseModel

from ..utils import Timer
from ..llm import LLMGuidance
from .feature import BeamFeature
from ..logging import beam_logger as logger


class LLMFeature(BeamFeature):
    prompt_suffix = "Return your response as a json file with the following schema:\n\n{schema}\n{text}\n\nYour Response:\n\n"

    def __init__(self, instruction: str, schema: BaseModel, *args, timeout=None, name=None, **kwargs):
        if name is None:
            name = 'llm_feature'
        super().__init__(name, *args, **kwargs)
        self._schema = schema
        self.guidance = LLMGuidance(guided_json=schema)
        self.instruction = instruction

        if timeout is None:
            timeout = self.hparams.get('llm_timeout', 60)
        self.timeout = timeout

    @property
    def schema(self):
        return self._schema.model_json_schema()

    def generate_prompt(self, text):
        suffix = self.prompt_suffix.format(schema=self.schema, text=text)
        prompt = f"{self.instruction}\n\n{suffix}"
        return prompt

    def apply(self, text):
        prompt = self.generate_prompt(text)
        task_with_timeout = Timer(timeout=self.timeout, task=self.llm.ask, graceful=False)

        try:
            res = task_with_timeout.run(prompt, guidance=self.guidance)
            res = res.json
        except Exception as e:
            logger.error(f"Failed to process LLMFeature: {e} (instruction={self.instruction[:30]}, text={text[:30]})")
            res = None

        return res