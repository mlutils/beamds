import atexit
import inspect
import threading
import time
from collections import defaultdict
from queue import Queue
from typing import Tuple, Dict, Any
from uuid import uuid4 as uuid
from multiprocessing import Process

from dataclasses import dataclass, field

from ..core import Processor
from ..utils import lazy_property
from ..path import BeamURL
from ..logger import beam_logger as logger
from ..parallel import parallel, task


class BeamWorker(Processor):

    def __init__(self, obj, *args, name=None, n_workers=1, daemon=False, broker=None, backend=None,
                 broker_username=None, broker_password=None, broker_port=None, broker_scheme=None, broker_host=None,
                 backend_username=None, backend_password=None, backend_port=None, backend_scheme=None, backend_host=None,
                 **kwargs):

        super().__init__(*args, name=name, n_workers=n_workers, daemon=daemon, **kwargs)

        if broker_scheme is None:
            broker_scheme = 'amqp'
        self.broker_url = BeamURL(url=broker, username=broker_username, password=broker_password, port=broker_port,
                           scheme=broker_scheme, host=broker_host)

        if backend_scheme is None:
            backend_scheme = 'redis'
        self.backend_url = BeamURL(url=backend, username=backend_username, password=backend_password, port=backend_port,
                                   scheme=backend_scheme, host=backend_host)

        self.obj = obj
        self.n_workers = self.hparams.get('n_workers')
        self.daemon = self.hparams.get('daemon')

        logger.info(f"Broker: {self.broker_url.url}, Backend: {self.backend_url.url}, "
                    f"n_workers: {self.n_workers}, daemon: {self.daemon}")

    @lazy_property
    def type(self):
        if inspect.isfunction(self.obj):
            return 'function'
        return 'class'

    @lazy_property
    def broker(self):
        from celery import Celery
        return Celery(self.name, broker=self.broker_url.url, backend=self.backend_url.url)

    def start_worker(self):
        from celery.apps.worker import Worker
        worker = Worker(app=self.broker, loglevel='info', traceback=True)
        worker.start()

    def run(self, *attributes):
        if self.type == 'function':
            self.broker.task(name='function')(self.obj)
        else:
            for at in attributes:
                self.broker.task(name=at)(getattr(self.obj, at))

        if self.n_workers == 1 and not self.daemon:
            # Run in the main process
            self.start_worker()
        else:
            # Start multiple workers in separate processes
            processes = [Process(target=self.start_worker, daemon=self.daemon) for _ in range(self.n_workers)]
            for p in processes:
                p.start()


@dataclass
class Task:
    req_id: str
    args: Tuple = tuple()
    kwargs: Dict = field(default_factory=dict)
    done: bool = False
    in_progress: bool = False
    result: Any = None
    success: bool = False
    exception: Exception = None
    traceback: str = None
    start_time: float = None
    end_time: float = None


class BatchExecutor(Processor):

    def __init__(self, *args, batch_size=None, **kwargs):
        super().__init__(*args, batch_size=batch_size, **kwargs)
        self.model = None
        self.batch_size = self.hparams.get('batch_size')
        self.request_queue = Queue()
        self.response_queue = defaultdict(Queue)

        # Start the batch processing in a separate thread
        self.centralized_thread = threading.Thread(target=self._centralized_batch_executor, daemon=True)
        self.centralized_thread.start()

        atexit.register(self._cleanup)

    def _cleanup(self):
        if self.centralized_thread is not None:
            self.centralized_thread.join()

    def execute(self, *args, **kwargs):
        # Add request to queue
        req_id = str(uuid())
        response_queue = self.response_queue[req_id]

        logger.info(f"Putting task with req_id: {req_id}")
        self.request_queue.put(Task(req_id, args, kwargs, start_time=time.time()))
        # Wait for response
        result = response_queue.get()
        logger.info(f"Got result for task with req_id: {req_id}")

        return result

    def process_batch(self, *tasks, **kwargs):
        raise NotImplementedError

    def _centralized_batch_executor(self):
        # fetch from queue, generate n-steps tokens check done sequences and return to queue
        # running 1 generated token is done with self.model.step(batch_of_sequences)

        tasks = []

        while True:

            while len(tasks) < self.batch_size:
                tasks.append(self.request_queue.get_nowait())

            tasks = list(self.process_batch(tasks))
            for i, t in enumerate(tasks):
                if t.done:
                    t = tasks.pop(i)
                    t.end_time = time.time()
                    self.response_queue[t.req_id].put(t)
