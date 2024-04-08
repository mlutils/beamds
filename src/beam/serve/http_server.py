import copy
from queue import Queue, Empty
from threading import Thread
from flask import Flask, request, jsonify, send_file

from ..logger import beam_logger as logger
from .server import BeamServer

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False


class HTTPServer(BeamServer):
    """
    Class representing a server for executing inference on an object or function.

    Args:
        obj: The object or function to perform inference on.
        use_torch (bool): Whether to use torch for serialization and deserialization. Default is True.
        batch: The methods to batch execute. Can be a boolean, string or list.
               If True, it will batch execute all the methods with '__call__' in the name.
               If a string, it will batch execute the method with the specified name.
               If a list, it will batch execute the methods with the names in the list.
        max_wait_time (float): The maximum time to wait for new inference tasks in a batch. Default is 1.0 second.
        max_batch_size (int): The maximum size of a batch for batch inference. Default is 10.
        tls (bool): Whether to use a secure TLS connection. Default is False.
        n_threads (int): The number of threads to use for the server. Default is 4.
        **kwargs: Additional keyword arguments.

    Attributes:
        app (Flask): The Flask application object.
        obj: The object or function being served.
        load_function: The function used for deserializing data.
        dump_function: The function used for serializing data.
        serialization_method (str): The serialization method used ('torch' or 'pickle').
        max_wait_time (float): The maximum time to wait for new inference tasks in a batch.
        max_batch_size (int): The maximum size of a batch for batch inference.
        tls (bool): Whether to use a secure TLS connection.
        n_threads (int): The number of threads used for the server.
        _request_queue (Queue): The queue for incoming inference tasks.
        _response_queue (defaultdict(Queue)): The queue for outgoing inference results.
        centralized_thread (Thread): The thread used for centralized batch inference.

    Methods:
        __init__: Initializes the BeamServer object.
        set_variable: Sets the value of a variable in the object being served.
        get_variable: Gets the value of a variable from the object being served.
        _cleanup: Cleans up resources used by the server.
        request_queue: Returns the request queue, creating it if necessary.
        response_queue: Return the response queue, creating it if necessary.
        build_algorithm_from_path: Creates a BeamServer object from an experiment file.
        run_non_blocking: Starts the server in a separate thread.
        run: Starts the server on the specified host and port.
        _centralized_batch_executor: Executes batch inference in a centralized manner.
        get_info: Gets information about the server and the object being served.
        batched_query_algorithm: Executes a method on the object being served in batch mode.
        call_function: Executes the '__call__' method on the function being served.
        query_algorithm: Executes a method on the object being served.
        run_uwsgi: Starts the server using uWSGI.
        run_waitress: Starts the server using Waitress.
        run_cherrypy: Starts the server using CherryPy.
        run_gunicorn: Starts the server using Gunicorn.
        run_wsgi: Starts the server using WSGI.
    """
    def __init__(self, *args, application=None, **kwargs):
        application = 'flask' or application
        super().__init__(*args, application=application, **kwargs)
        self.app = Flask(__name__)
        self.app.add_url_rule('/', view_func=self.get_info)

        if self.type == 'function':
            self.app.add_url_rule('/call/<client>', view_func=self.call, methods=['POST'])
        elif self.type == 'instance':
            if hasattr(self.obj, '__call__'):
                self.app.add_url_rule('/call/<client>', view_func=self.call, methods=['POST'])
            self.app.add_url_rule('/alg/<client>/<method>', view_func=self.query_algorithm, methods=['POST'])
            self.app.add_url_rule('/setvar/<client>/<name>', view_func=self.set_variable, methods=['POST'])
            self.app.add_url_rule('/getvar/<client>/<name>', view_func=self.get_variable)
        else:
            raise ValueError(f"Unknown type: {self.type}")

    def set_variable(self, client, name, *args, **kwargs):
        value = request.files['value']
        res = super().set_variable(client, name, value)
        return jsonify(res)

    def get_variable(self, client, name):
        io_results = super().get_variable(client, name)
        if client == 'beam':
            return send_file(io_results, mimetype="text/plain")
        else:
            return jsonify(io_results)

    def _run(self, host="0.0.0.0", port=None, server='waitress', use_reloader=True):

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
            raise ValueError(f"Unknown serve type: {server}")

    def get_info(self):
        d = super().get_info()
        return jsonify(d)

    @staticmethod
    def request_metadata(client='beam', method=None):

        metadata = {
            'method': request.method,
            'url': request.url,
            'base_url': request.base_url,
            'args': copy.deepcopy(request.args),
            'form': copy.deepcopy(request.form),
            'headers': copy.deepcopy(dict(request.headers)),
            'remote_addr': request.remote_addr,
            'user_agent': str(request.user_agent),
            'client': client,
            'alg_method': method,

        }

        return metadata

    def query_algorithm(self, client, method, *args, **kwargs):

        if client == 'beam':
            args = request.files['args']
            kwargs = request.files['kwargs']
        else:
            data = request.get_json()
            args = data.pop('args', [])
            kwargs = data.pop('kwargs', {})

        io_results = super().query_algorithm(client, method, args, kwargs)

        if client == 'beam':
            return send_file(io_results, mimetype="text/plain")
        else:
            return jsonify(io_results)

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
            'serve.socket_host': host,
            'serve.socket_port': port,
            'engine.autoreload.on': False,
            'serve.thread_pool': self.n_threads
        }
        if self.tls:
            config.update({
                'serve.ssl_module': 'builtin',
                'serve.ssl_certificate': 'path/to/certfile.pem',
                'serve.ssl_private_key': 'path/to/keyfile.pem'
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
            logger.warning("WSGI serve does not support multithreading, setting n_threads to 1")

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
