import queue
import threading

from src.beam.config import BeamHparams, BeamParam
from src.beam import beam_logger as logger
from src.beam.distributed.worker import BeamWorker
from src.beam.core import Processor
from src.beam.utils import lazy_property


class LLMServeHparams(BeamHparams):

    defaults = {}
    use_basic_parser = False
    parameters = [
        BeamParam('broker', str, 'amqp://localhost:5672//', 'The broker URL', model=False),
        BeamParam('backend', str, 'redis://localhost:6379/0', 'The backend URL', model=False),
        BeamParam('model', str, None, 'name of the model to be used with vLLM', model=False),
        BeamParam('n-steps', int, 32, 'check for sequence completion after n-steps', model=False),
        BeamParam('batch-size', int, 4, 'The batch size', model=False),
    ]


class LLMBatchWorker(Processor):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = None
        self.n_steps = hparams.n_steps
        self.batch_size = hparams.batch_size
        self._request_queue = queue.Queue()
        self._response_queue = queue.Queue()

    @lazy_property
    def model(self):
        from vllm import LLM
        return LLM(self.hparams.get('model'))

    def generate(self, text):
        # Add request to queue
        self._request_queue.put(text)
        # Wait for response
        result = self._response_queue.get()
        return result

    def run(self, *attributes):
        # fetch from queue, generate n-steps tokens check done sequences and return to queue
        # running 1 generated token is done with self.model.step(batch_of_sequences)
        while True:
            # Check if enough requests are in the queue
            if self._request_queue.qsize() >= self.batch_size:
                batch = [self._request_queue.get() for _ in range(self.batch_size)]
                results = self.process_batch(batch)
                for result in results:
                    self._response_queue.put(result)


if __name__ == '__main__':

    hparams = LLMServeHparams()
    llm_worker = BeamWorker(hparams)

    # Start the batch processing in a separate thread
    threading.Thread(target=llm_worker.run, daemon=True).start()

    celery_worker = BeamWorker(llm_worker, name='llm_worker', broker=hparams.broker, backend=hparams.backend)
    # add a centralized thread that run the actual generation
    celery_worker.run('generate')

