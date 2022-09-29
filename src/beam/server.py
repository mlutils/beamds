from flask import Flask, jsonify, request, send_file
from .experiment import beam_algorithm_generator, Experiment
import requests
import io
import torch
from .utils import logger, find_port
from gevent.pywsgi import WSGIServer


class BeamClient(object):

    def __init__(self, host):
        self.host = host

    def get(self, path):

        response = requests.get(f'http://{self.host}/{path}')
        return response

    def post(self, path, *args, **kwargs):

        io_args = io.BytesIO()
        torch.save(args, io_args)
        io_args.seek(0)

        io_kwargs = io.BytesIO()
        torch.save(kwargs, io_kwargs)
        io_kwargs.seek(0)

        response = requests.post(f'http://{self.host}/{path}', files={'args': io_args, 'kwargs': io_kwargs}, stream=True)

        if response.status_code == 200:
            response = torch.load(io.BytesIO(response.raw.data))

        return response


class BeamServer(object):

    def __init__(self, alg):

        self.alg = alg
        self.app = Flask(__name__)
        self.app.add_url_rule('/', view_func=self.get_info)
        self.app.add_url_rule('/alg/<method>', view_func=self.query_algorithm,  methods=['POST'])

    @staticmethod
    def build_algorithm_from_path(path, Alg, override_hparams=None, Dataset=None, alg_args=None, alg_kwargs=None,
                             dataset_args=None, dataset_kwargs=None, **argv):

        experiment = Experiment.reload_from_path(path, override_hparams=override_hparams, **argv)
        alg = experiment.algorithm_generator(Alg, Dataset=Dataset, alg_args=alg_args, alg_kwargs=alg_kwargs,
                                                  dataset_args=dataset_args, dataset_kwargs=dataset_kwargs)
        return BeamServer(alg)

    def run(self, host="0.0.0.0", port=None, debug=False):
        port = find_port(port=port, get_port_from_beam_port_range=True, application='flask')
        logger.info(f"Opening a flask inference server on port: {port}")

        # when debugging with pycharm set debug=False
        # if needed set use_reloader=False
        # see https://stackoverflow.com/questions/25504149/why-does-running-the-flask-dev-server-run-itself-twice
        # self.app.run(host=host, port=port, debug=debug)

        if port is not None:
            port = int(port)

        http_server = WSGIServer((host, port), self.app)
        http_server.serve_forever()

    def get_info(self):
        return jsonify(self.alg.experiment.vars_args)

    def query_algorithm(self, method):

        method = getattr(self.alg, method)

        args = request.files['args']
        kwargs = request.files['kwargs']

        args = torch.load(args)
        kwargs = torch.load(kwargs)

        results = method(*args, **kwargs)

        io_results = io.BytesIO()
        torch.save(results, io_results)
        io_results.seek(0)

        return send_file(io_results, mimetype="text/plain")
