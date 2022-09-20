from flask import Flask, jsonify, request, send_file
from .experiment import beam_algorithm_generator, Experiment
import requests
import io
import torch
from .utils import logger, find_port


class BeamClient(object):

    def __init__(self, url):
        self.url = url

    def get(self, path):

        response = requests.get(f'http://{self.url}/{path}')
        return response

    def post(self, path, *args, **kwargs):

        io_args = io.BytesIO()
        torch.save(args, io_args)
        io_args.seek(0)

        io_kwargs = io.BytesIO()
        torch.save(kwargs, io_kwargs)
        io_kwargs.seek(0)

        response = requests.post(f'http://{self.url}/{path}', files={'args': io_args, 'kwargs': io_kwargs}, stream=True)

        if response.status_code == 200:
            response = torch.load(io.BytesIO(response.raw.data))

        return response


class BeamServer(object):

    def __init__(self, path, Alg, override_hparams=None, Dataset=None, alg_args=None, alg_kwargs=None,
                             dataset_args=None, dataset_kwargs=None, **argv):

        self.experiment = Experiment.reload_from_path(path, override_hparams=override_hparams, **argv)
        self.alg = self.experiment.algorithm_generator(Alg, Dataset=Dataset, alg_args=alg_args, alg_kwargs=alg_kwargs,
                             dataset_args=dataset_args, dataset_kwargs=dataset_kwargs)

        self.app = Flask(self.experiment.root)
        self.app.add_url_rule('/', view_func=self.get_info)
        self.app.add_url_rule('/predict', view_func=self.predict,  methods=['POST'])

    def run(self, host="0.0.0.0", port=None, debug=False):
        port = find_port(port=port, get_port_from_beam_port_range=True, application='flask')
        logger.info(f"Opening a flask inference server on port: {port}")

        # when debugging with pycharm set debug=False
        # if needed set use_reloader=False
        # see https://stackoverflow.com/questions/25504149/why-does-running-the-flask-dev-server-run-itself-twice
        self.app.run(host=host, port=port, debug=debug)

    def get_info(self):
        # vars(self.experiment.hparams)
        return jsonify(self.experiment.vars_args)

    def predict(self):
        args = request.files['args']
        kwargs = request.files['kwargs']

        args = torch.load(args)
        kwargs = torch.load(kwargs)

        results = self.alg.predict(*args, **kwargs)

        io_results = io.BytesIO()
        torch.save(results, io_results)
        io_results.seek(0)

        return send_file(io_results, mimetype="text/plain")
