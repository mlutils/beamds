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
from .utils import find_port, normalize_host, check_type

from .logger import beam_logger as logger
import types
from functools import partial
from queue import Queue, Full, Empty
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

    def __init__(self, obj, use_torch=True, batch=None, max_wait_time=1.0, max_batch_size=10, tls=False,
                 n_threads=4, **kwargs):

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

        self.max_wait_time = max_wait_time
        self.max_batch_size = max_batch_size
        self.tls = tls
        self.n_threads = n_threads

        self._request_queue = None
        self._response_queue = None

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

    def run(self, host="0.0.0.0", port=None, server='wsgi', use_reloader=True):
        port = find_port(port=port, get_port_from_beam_port_range=True, application='flask')
        logger.info(f"Opening a flask inference server on port: {port}")

        # when debugging with pycharm set debug=False
        # if needed set use_reloader=False
        # see https://stackoverflow.com/questions/25504149/why-does-running-the-flask-dev-server-run-itself-twice

        if port is not None:
            port = int(port)

        if server == 'debug':
            self.app.run(host=host, port=port, debug=True, use_reloader=use_reloader, threaded=True)
        elif server == 'wsgi':
            self.run_wsgi(host, port)
        elif server == 'uwsgi':
            self.run_uwsgi(host, port)
        elif server == 'waitress':
            self.run_waitress(host, port)
        elif server == 'cherrypy':
            self.run_cherrypy(host, port)
        elif server == 'gunicorn':
            self.run_gunicorn(host, port)
        else:
            raise ValueError(f"Unknown server type: {server}")

    def run_uwsgi(self, host, port):

        from uwsgi import run

        uwsgi_opts = {
            'http': f'{host}:{port}',
            'wsgi-file': 'your_wsgi_file.py',  # Replace with your WSGI file
            'callable': 'app',  # Replace with your WSGI application callable
            'processes': self.n_threads,
        }

        if self.tls:
            uwsgi_opts['https-socket'] = f'{host}:{port}'
            uwsgi_opts['https-keyfile'] = 'path/to/keyfile.pem'
            uwsgi_opts['https-certfile'] = 'path/to/certfile.pem'

        run([], **uwsgi_opts)

    def run_waitress(self, host, port):

        from waitress import serve
        serve(self.app, host=host, port=port, threads=self.n_threads)

    def run_cherrypy(self, host, port):

        import cherrypy

        cherrypy.tree.graft(self.app, '/')
        config = {
            'server.socket_host': host,
            'server.socket_port': port,
            'engine.autoreload.on': False,
            'server.thread_pool': self.n_threads
        }
        if self.tls:
            config.update({
                'server.ssl_module': 'builtin',
                'server.ssl_certificate': 'path/to/certfile.pem',
                'server.ssl_private_key': 'path/to/keyfile.pem'
            })
        cherrypy.config.update(config)

        cherrypy.engine.start()
        cherrypy.engine.block()

    def run_gunicorn(self, host, port):

        from gunicorn.app.wsgiapp import WSGIApplication
        options = {
            'bind': f'{host}:{port}',
            'workers': 1,  # Gunicorn forks multiple processes and is generally not thread-safe
            'threads': self.n_threads,
            'accesslog': '-',
        }

        if self.tls:
            options['keyfile'] = 'path/to/keyfile.pem'
            options['certfile'] = 'path/to/certfile.pem'

        app = WSGIApplication()
        app.load_wsgiapp = lambda: self.app
        app.cfg.set(options)
        app.run()

    def run_wsgi(self, host, port):

        from gevent.pywsgi import WSGIServer

        if self.n_threads > 1:
            logger.warning("WSGI server does not support multithreading, setting n_threads to 1")

        if self.tls:
            from gevent.pywsgi import WSGIServer
            from gevent.ssl import SSLContext
            from os.path import join, dirname, realpath

            cert = join(dirname(realpath(__file__)), 'cert.pem')
            key = join(dirname(realpath(__file__)), 'key.pem')
            context = SSLContext()
            context.load_cert_chain(cert, key)
        else:
            context = None

        http_server = WSGIServer((host, port), self.app, ssl_context=context)
        http_server.serve_forever()

    def _centralized_batch_executor(self):

        from .data import BeamData
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

    def get_info(self):

        d = {'serialization': self.serialization_method, 'obj': self.type, 'name': None}
        if self.type == 'function':
            d['vars_args'] = self.obj.__code__.co_varnames
        else:
            d['vars_args'] = self.obj.__init__.__code__.co_varnames
            if hasattr(self.obj, 'hparams'):
                d['hparams'] = vars(self.obj.hparams)
            else:
                d['hparams'] = None

        if hasattr(self.obj, 'name'):
            d['name'] = self.obj.name

        return jsonify(d)

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

    def call_function(self):

            args = request.files['args']
            kwargs = request.files['kwargs']

            args = self.load_function(args)
            kwargs = self.load_function(kwargs)

            if '__call__' in self.batch:
                results = self.batched_query_algorithm('__call__', args, kwargs)
            else:
                results = self.obj(*args, **kwargs)

            io_results = io.BytesIO()
            self.dump_function(results, io_results)
            io_results.seek(0)

            return send_file(io_results, mimetype="text/plain")

    def query_algorithm(self, method):

        args = request.files['args']
        kwargs = request.files['kwargs']

        args = self.load_function(args)
        kwargs = self.load_function(kwargs)

        if method in self.batch:
            results = self.batched_query_algorithm(method, args, kwargs)
        else:
            method = getattr(self.obj, method)
            results = method(*args, **kwargs)

        io_results = io.BytesIO()
        self.dump_function(results, io_results)
        io_results.seek(0)

        return send_file(io_results, mimetype="text/plain")
