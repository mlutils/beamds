from flask import Flask, jsonify, request, send_file
from .experiment import Experiment
import requests
import io
import torch
from .utils import find_port, normalize_host
from gevent.pywsgi import WSGIServer
from .logger import beam_logger as logger
import types


def beam_remote(obj, host=None, port=None, debug=False):
    server = BeamServer(obj)
    server.run(host=host, port=port, debug=debug)


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

    def __call__(self, *args, **kwargs):
        return self.post('call', *args, **kwargs)

    def __getattr__(self, item):

        def method(*args, **kwargs):
            return self.post(f'alg/{item}', *args, **kwargs)

        return method


class BeamServer(object):

    def __init__(self, obj):

        self.app = Flask(__name__)
        self.app.add_url_rule('/', view_func=self.get_info)

        self.alg = obj
        self.func = obj

        if callable(obj):
            self.app.add_url_rule('/call', view_func=self.call_function, methods=['POST'])
        elif isinstance(obj, object):
            self.app.add_url_rule('/alg/<method>', view_func=self.query_algorithm, methods=['POST'])

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

        if self.alg is None:
            return jsonify(self.func.__code__.co_varnames)
        else:
            return jsonify(self.alg.experiment.vars_args)

    def call_function(self):

            args = request.files['args']
            kwargs = request.files['kwargs']

            args = torch.load(args)
            kwargs = torch.load(kwargs)

            results = self.func(*args, **kwargs)

            io_results = io.BytesIO()
            torch.save(results, io_results)
            io_results.seek(0)

            return send_file(io_results, mimetype="text/plain")

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

#
# # fastapi version
# #
# from fastapi import FastAPI, Depends, Request, File, UploadFile
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# import torch
# import io
#
# app = FastAPI()
#
# class AlgModel(BaseModel):
#     path: str
#     override_hparams: dict = None
#     Dataset: str = None
#     alg_args: list = []
#     alg_kwargs: dict = {}
#     dataset_args: list = []
#     dataset_kwargs: dict = {}
#
# class BeamServer:
#
#     def __init__(self, alg):
#         self.alg = alg
#
#     @classmethod
#     def build_algorithm_from_path(cls, alg_model: AlgModel):
#         experiment = Experiment.reload_from_path(alg_model.path, override_hparams=alg_model.override_hparams)
#         alg = experiment.algorithm_generator(Alg, Dataset=alg_model.Dataset,
#                                              alg_args=alg_model.alg_args,
#                                              alg_kwargs=alg_model.alg_kwargs,
#                                              dataset_args=alg_model.dataset_args,
#                                              dataset_kwargs=alg_model.dataset_kwargs)
#         return cls(alg)
#
#     def get_info(self):
#         return self.alg.experiment.vars_args
#
#     async def query_algorithm(self, method: str, args_file: UploadFile = File(...), kwargs_file: UploadFile = File(...)):
#         method = getattr(self.alg, method)
#         args = torch.load(await args_file.read())
#         kwargs = torch.load(await kwargs_file.read())
#
#         results = method(*args, **kwargs)
#
#         io_results = io.BytesIO()
#         torch.save(results, io_results)
#         io_results.seek(0)
#         return StreamingResponse(io_results, media_type="text/plain")
#
# beam_server_instance = BeamServer(alg) # You would initialize this with your actual algorithm
#
# @app.get("/")
# def get_info():
#     return beam_server_instance.get_info()
#
# @app.post("/alg/{method}")
# async def query_algorithm(method: str, args_file: UploadFile = File(...), kwargs_file: UploadFile = File(...)):
#     return await beam_server_instance.query_algorithm(method, args_file, kwargs_file)
#
# # In order to run the FastAPI app, you would typically use:
# # uvicorn your_module:app --host 0.0.0.0 --port 8000 --reload