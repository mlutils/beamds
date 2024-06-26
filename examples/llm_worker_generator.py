import threading
from queue import Queue
from uuid import uuid4 as uuid
from beam.utils import cached_property

from beam.config import BeamConfig, BeamParam
from beam import beam_logger as logger
from beam.distributed.celery_worker import CeleryWorker
from beam.processor import Processor


class LLMServeConfig(BeamConfig):

    defaults = {}
    parameters = [
        BeamParam('broker', str, 'amqp://localhost:5672//', 'The broker URL', model=False),
        BeamParam('backend', str, 'redis://localhost:6379/0', 'The backend URL', model=False),
        BeamParam('model', str, None, 'name of the model to be used with vLLM', model=False),
        BeamParam('revision', str, None, 'A version of the model (could be none)', model=False),
        BeamParam('n-steps', int, 32, 'check for sequence completion after n-steps', model=False),
        BeamParam('batch-size', int, 4, 'The batch size', model=False),
    ]


class VLLMGenerator(Processor):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.n_steps = hparams.n_steps
        self.batch_size = hparams.batch_size

    @cached_property
    def model(self):
        from vllm import LLM
        return LLM(self.hparams.get('model'), revision=self.hparams.get('revision'),
                   trust_remote_code=True)

    def generate(self, prompts, sampling_params=None):
        outputs = self.model.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

        generated_text = []
        for output in outputs:
            # prompt = output.prompt
            generated_text.append(output.outputs[0].text)

        return generated_text


if __name__ == '__main__':

    hparams = LLMServeConfig()
    llm_worker = VLLMGenerator(hparams)

    celery_worker = CeleryWorker(llm_worker, name='llm_worker', broker=hparams.broker, backend=hparams.backend)
    # add a centralized thread that run the actual generation
    celery_worker.run('generate')
    logger.info(f"Starting celery worker: {celery_worker.name}")

