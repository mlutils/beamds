from flask import Flask, jsonify, request, send_file
from .experiment import Experiment
import requests
import io
try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

import pickle
from .utils import find_port, normalize_host
from gevent.pywsgi import WSGIServer
from .logger import beam_logger as logger
import types
from functools import partial
from queue import Queue
from threading import Thread
from uuid import uuid4 as uuid
from collections import defaultdict
import time
import atexit


def beam_remote(obj, host=None, port=None, debug=False):
    server = BeamServer(obj)
    server.run(host=host, port=port, debug=debug)


class BeamClient(object):

    def __init__(self, host):
        self.host = host
        self._info = None
        self._serialization = None

    @property
    def load_function(self):
        if self.serialization == 'torch':
            if not has_torch:
                raise ImportError('Cannot use torch serialization without torch installed')
            return torch.load
        else:
            return pickle.load

    @property
    def dump_function(self):
        if self.serialization == 'torch':
            if not has_torch:
                raise ImportError('Cannot use torch serialization without torch installed')
            return torch.save
        else:
            return pickle.dump

    @property
    def serialization(self):
        if self._serialization is None:
            self._serialization = self.info['serialization']
        return self._serialization

    @property
    def info(self):
        if self._info is None:
            self._info = requests.get(f'http://{self.host}/').json()
        return self._info

    def get(self, path):

        response = requests.get(f'http://{self.host}/{path}')
        return response

    def post(self, path, *args, **kwargs):

        io_args = io.BytesIO()
        self.dump_function(args, io_args)
        io_args.seek(0)

        io_kwargs = io.BytesIO()
        self.dump_function(kwargs, io_kwargs)
        io_kwargs.seek(0)

        response = requests.post(f'http://{self.host}/{path}', files={'args': io_args, 'kwargs': io_kwargs}, stream=True)

        if response.status_code == 200:
            response = self.load_function(io.BytesIO(response.raw.data))

        return response

    def __call__(self, *args, **kwargs):
        return self.post('call', *args, **kwargs)

    def __getattr__(self, item):
        return partial(self.post, f'alg/{item}')


class BeamServer(object):

    def __init__(self, obj, use_torch=True, batch=False, max_wait_time=1.0, max_batch_size=10):

        self.app = Flask(__name__)
        self.app.add_url_rule('/', view_func=self.get_info)

        self.obj = obj

        if use_torch and has_torch:
            self.load_function = torch.load
            self.dump_function = torch.save
            self.serialization_method = 'torch'
        else:
            self.load_function = pickle.load
            self.dump_function = pickle.dump
            self.serialization_method = 'pickle'

        self.batch = batch
        self.max_wait_time = max_wait_time
        self.max_batch_size = max_batch_size
        self._request_queue = None
        self._response_queue = None

        if batch:
            # Initialize and start batch inference thread
            self.centralized_thread = Thread(target=self._centralized_batch_executor)
            self.centralized_thread.daemon = True
            self.centralized_thread.start()
        else:
            self.centralized_thread = None

        atexit.register(self._cleanup)

        if callable(obj):
            self.type = 'function'
            self.app.add_url_rule('/call', view_func=self.call_function, methods=['POST'])
        elif isinstance(obj, object):
            self.type = 'class'
            self.app.add_url_rule('/alg/<method>', view_func=self.query_algorithm, methods=['POST'])

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

    @staticmethod
    def build_algorithm_from_path(path, Alg, override_hparams=None, Dataset=None, alg_args=None, alg_kwargs=None,
                             dataset_args=None, dataset_kwargs=None, **argv):

        experiment = Experiment.reload_from_path(path, override_hparams=override_hparams, **argv)
        alg = experiment.algorithm_generator(Alg, Dataset=Dataset, alg_args=alg_args, alg_kwargs=alg_kwargs,
                                                  dataset_args=dataset_args, dataset_kwargs=dataset_kwargs)
        return BeamServer(alg)

    def run(self, host="0.0.0.0", port=None, debug=False, use_reloader=True):
        port = find_port(port=port, get_port_from_beam_port_range=True, application='flask')
        logger.info(f"Opening a flask inference server on port: {port}")

        # when debugging with pycharm set debug=False
        # if needed set use_reloader=False
        # see https://stackoverflow.com/questions/25504149/why-does-running-the-flask-dev-server-run-itself-twice

        if port is not None:
            port = int(port)

        if debug:
            self.app.run(host=host, port=port, debug=True, use_reloader=use_reloader)
        else:
            http_server = WSGIServer((host, port), self.app)
            http_server.serve_forever()

    def _centralized_batch_executor(self):
        while True:
            batch = []
            start_time = time.time()

            while len(batch) < self.max_batch_size:
                elapsed_time = time.time() - start_time
                if elapsed_time > self.max_wait_time:
                    break

                try:
                    task = self.request_queue.get(timeout=self.max_wait_time - elapsed_time)
                    batch.append(task)
                except Queue.Empty:
                    break

            if len(batch) > 0:

                methods = defaultdict(list)
                for task in batch:
                    methods[task['method']].append(task)

                for method, tasks in methods.items():

                    func = getattr(self.obj, method)

                    # currently we support only batching with a single argument
                    data = {task['req_id']: task['args'][0] for task in tasks}
                    from .data import BeamData

                    bd = BeamData.simple(data)
                    bd.apply(func)

                    results = {task['req_id']: bd.data[task['req_id']] for task in tasks}

                    for req_id, result in results.items():
                        self.response_queue[req_id].put(result)

    def get_info(self):

        d = {'serialization': self.serialization_method, 'obj': self.type, 'name': None}
        if self.type == 'function':
            d['vars_args'] = self.obj.__code__.co_varnames
        else:
            d['vars_args'] = self.obj.experiment.vars_args

        if hasattr(self.obj, 'name'):
            d['name'] = self.obj.name

        return jsonify(d)

    def batched_query_algorithm(self, method, args, kwargs):

        # Generate a unique request ID
        req_id = str(uuid())
        response_queue = self.response_queue[req_id]
        self.request_queue.put({'req_id': req_id, 'method': method, 'args': args, 'kwargs': kwargs})

        # Wait for the result
        result = response_queue.get()
        del self.response_queue[req_id]
        return result

    def call_function(self):

            args = request.files['args']
            kwargs = request.files['kwargs']

            args = self.load_function(args)
            kwargs = self.load_function(kwargs)

            if self.batch:
                results = self.batched_query_algorithm('__call__', args, kwargs)
            else:
                results = self.obj(*args, **kwargs)

            io_results = io.BytesIO()
            self.dump_function(results, io_results)
            io_results.seek(0)

            return send_file(io_results, mimetype="text/plain")

    def query_algorithm(self, method):

        method = getattr(self.obj, method)

        args = request.files['args']
        kwargs = request.files['kwargs']

        args = self.load_function(args)
        kwargs = self.load_function(kwargs)

        if self.batch:
            results = self.batched_query_algorithm(method, args, kwargs)
        else:
            results = method(*args, **kwargs)

        io_results = io.BytesIO()
        self.dump_function(results, io_results)
        io_results.seek(0)

        return send_file(io_results, mimetype="text/plain")
