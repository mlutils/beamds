import atexit
import io
import pickle
import time
from collections import defaultdict
from queue import Queue, Empty
from threading import Thread
from uuid import uuid4 as uuid

from ..logging import beam_logger as logger
from ..utils import find_port
from ..processor import MetaDispatcher
from ..importer import lazy_importer as lzi
from ..importer import torch


class BeamServer(MetaDispatcher):

    def __init__(self, obj, *args, use_torch=True, batch=None, max_wait_time=1.0, max_batch_size=10,
                 tls=False, n_threads=4, queue_size=None, application=None, **kwargs):

        super().__init__(obj, *args, asynchronous=False, **kwargs)

        if use_torch and lzi.has('torch'):
            self.load_function = torch.load
            self.dump_function = torch.save
            self.serialization_method = 'torch'
            # check if torch version is >= 2.6.0
            self.lf_kwargs = {'weights_only': False}
        else:
            self.load_function = pickle.load
            self.dump_function = pickle.dump
            self.serialization_method = 'pickle'
            self.lf_kwargs = {}

        self.max_wait_time = max_wait_time
        self.max_batch_size = max_batch_size
        self.tls = tls
        self.n_threads = n_threads
        self.queue_size = queue_size

        self._request_queue = None
        self._response_queue = None
        self.application = application

        if batch:

            if type(batch) is bool:
                self.batch = ['predict', '__call__']
            elif type(batch) is str:
                self.batch = [batch]
            elif type(batch) is list:
                self.batch = batch
            else:
                raise ValueError(f"Unknown batch type: {batch}")

            # Initialize and start batch inference thread
            self.centralized_thread = Thread(target=self._centralized_batch_executor)
            self.centralized_thread.daemon = True
            self.centralized_thread.start()
        else:
            self.centralized_thread = None
            self.batch = []

        atexit.register(self._cleanup)

    def get_info(self):
        info = super().get_info()
        info.serialization = self.serialization_method
        return info

    def set_variable(self, client, name, value, *args, **kwargs):

        if client == 'beam':
            value = self.load_function(value, **self.lf_kwargs)

        setattr(self.obj, name, value)
        return {'success': True}

    def get_variable(self, client, name):

        logger.info(f"Getting variable: {name}")
        value = getattr(self.obj, name)
        logger.info(f"value: {value}")

        if client == 'beam':
            io_results = io.BytesIO()
            self.dump_function(value, io_results)
            io_results.seek(0)
            return io_results

        return value

    def _cleanup(self):
        if self.centralized_thread is not None:
            self.centralized_thread.join()

    @property
    def request_queue(self):
        if self._request_queue is None:
            self._request_queue = Queue()
        return self._request_queue

    @property
    def response_queue(self):
        if self._response_queue is None:
            self._response_queue = defaultdict(Queue)
        return self._response_queue

    @classmethod
    def build_algorithm_from_path(cls, path, alg, override_hparams=None, dataset=None, alg_args=None, alg_kwargs=None,
                             dataset_args=None, dataset_kwargs=None, **argv):

        from ..experiment import Experiment
        experiment = Experiment.reload_from_path(path, override_hparams=override_hparams, **argv)
        alg = experiment.algorithm_generator(alg, dataset=dataset, alg_args=alg_args, alg_kwargs=alg_kwargs,
                                                  dataset_args=dataset_args, dataset_kwargs=dataset_kwargs)
        return cls(alg)

    def run_non_blocking(self, **kwargs):
        return self.run(non_blocking=True, **kwargs)

    def run(self, non_blocking=False, **kwargs):
        if non_blocking:
            run_thread = Thread(target=self.run_thread, kwargs=kwargs)
            run_thread.daemon = True
            run_thread.start()
        else:
            run_thread = self.run_thread(**kwargs)
        return run_thread

    def run_thread(self, host=None, port=None, **kwargs):
        if host is None:
            host = "0.0.0.0"
        port = find_port(port=port, get_port_from_beam_port_range=True, application=self.application)
        logger.info(f"Opening a {self.application} inference serve on port: {port}")
        return self._run(host=host, port=port, **kwargs)

    def _run(self, host="0.0.0.0", port=None, **kwargs):
        raise NotImplementedError

    def _centralized_batch_executor(self):

        from ..data import BeamData
        while True:
            logger.info(f"Starting a new batch inference")
            batch = []

            while len(batch) < self.max_batch_size:

                if len(batch) == 1:
                    start_time = time.time()
                    elapsed_time = 0
                elif len(batch) > 1:
                    elapsed_time = time.time() - start_time
                    if elapsed_time > self.max_wait_time:
                        logger.info(f"Max wait time reached, moving to execution")
                        break

                try:
                    if len(batch) > 0:
                        logger.info(f"Waiting for task, for {self.max_wait_time-elapsed_time} seconds")
                        task = self.request_queue.get(timeout=self.max_wait_time-elapsed_time)
                    else:
                        logger.info(f"Waiting for task")
                        task = self.request_queue.get()
                    batch.append(task)
                    logger.info(f"Got task with req_id: {task['req_id']}")
                except Empty:
                    logger.info(f"Empty queue, moving to execution")
                    break

            if len(batch) > 0:
                logger.info(f"Executing batch of size: {len(batch)}")

                methods = defaultdict(list)
                for task in batch:
                    methods[task['method']].append(task)

                for method, tasks in methods.items():

                    logger.info(f"Executing method: {method} with {len(tasks)} tasks")
                    func = getattr(self.obj, method)

                    # currently we support only batching with a single argument
                    data = {task['req_id']: task['args'][0] for task in tasks}

                    is_beam_data = type(data) is BeamData

                    bd = BeamData.simple(data)
                    bd = bd.apply(func)

                    if is_beam_data:
                        results = {task['req_id']: bd[task['req_id']] for task in tasks}
                    else:
                        results = {task['req_id']: bd[task['req_id']].values for task in tasks}

                    for req_id, result in results.items():

                        logger.info(f"Putting result for task: {req_id}")
                        self.response_queue[req_id].put(result)

    @property
    def metadata(self):
        return None

    def batched_query_algorithm(self, method, args, kwargs):

        # Generate a unique request ID
        req_id = str(uuid())
        response_queue = self.response_queue[req_id]

        logger.info(f"Putting task with req_id: {req_id}")
        self.request_queue.put({'req_id': req_id, 'method': method, 'args': args, 'kwargs': kwargs})

        # Wait for the result
        result = response_queue.get()

        logger.info(f"Got result for task with req_id: {req_id}")
        del self.response_queue[req_id]
        return result

    def call(self, client, *args, **kwargs):

        if self.type == 'function':
            return self.query_algorithm(client, None, *args, **kwargs)
        else:
            return self.query_algorithm(client, '__call__', *args, **kwargs)

    def query_algorithm(self, client, method, args, kwargs, return_raw_results=False):

        if client == 'beam':
            args = self.load_function(args, **self.lf_kwargs)
            kwargs = self.load_function(kwargs, **self.lf_kwargs)

        if method in self.batch:
            results = self.batched_query_algorithm(method, args, kwargs)
        else:
            if method is None:
                method = self.obj
            else:
                method = getattr(self.obj, method)
            results = method(*args, **kwargs)

        if not return_raw_results and client == 'beam':
            io_results = io.BytesIO()
            self.dump_function(results, io_results)
            io_results.seek(0)
            return io_results

        return results
