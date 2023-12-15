import threading
from queue import Queue
from uuid import uuid4 as uuid

from src.beam.config import BeamHparams, BeamParam
from src.beam import beam_logger as logger
from src.beam.distributed.worker import BeamWorker, BatchExecutor
from src.beam.core import Processor
from src.beam.utils import lazy_property
from collections import defaultdict


class LLMServeHparams(BeamHparams):

    defaults = {}
    use_basic_parser = False
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
        self.model = None
        self.n_steps = hparams.n_steps
        self.batch_size = hparams.batch_size
        from vllm import LLM

        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        self.model = LLM(self.hparams.get('model'), revision=self.hparams.get('revision'),
                            trust_remote_code=True)


    # @lazy_property
    # def model(self):
    #     from vllm import LLM
    #     return LLM(self.hparams.get('model'), revision=self.hparams.get('revision'),
    #                trust_remote_code=True)

    def generate(self, prompts, sampling_params=None):
        outputs = self.model.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        return outputs


if __name__ == '__main__':

    hparams = LLMServeHparams()
    llm_worker = VLLMGenerator(hparams)

    celery_worker = BeamWorker(llm_worker, name='llm_worker', broker=hparams.broker, backend=hparams.backend)
    # add a centralized thread that run the actual generation
    celery_worker.run('generate')
    logger.info(f"Starting celery worker: {celery_worker.name}")

