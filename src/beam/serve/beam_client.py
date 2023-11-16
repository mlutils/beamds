import io
import pickle
from functools import partial
import requests

from .beam_server import has_torch
if has_torch:
    import torch


class BeamClient(object):

    def __init__(self, host):
        self.host = host
        self._info = None

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
        return self.info['serialization']

    @property
    def info(self):
        if self._info is None:
            self._info = requests.get(f'http://{self.host}/').json()
        return self._info

    @property
    def attributes(self):
        return self.info['attributes']

    def get(self, path):

        response = requests.get(f'http://{self.host}/{path}')
        if response.status_code == 200:
            response = self.load_function(io.BytesIO(response.raw.data))

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
            response = self.load_function(io.BytesIO(response.content))

        return response

    def __call__(self, *args, **kwargs):
        return self.post('call', *args, **kwargs)

    def __getattr__(self, item):

        if item in ['_lazy_cache']:
            return super(BeamClient, self).__getattr__(item)

        if item not in self.attributes:
            self._info = None

        attribute_type = self.attributes[item]
        if attribute_type == 'variable':
            return self.get(f'getvar/{item}')
        elif attribute_type == 'method':
            return partial(self.post, f'alg/{item}')
        raise ValueError(f"Unknown attribute type: {attribute_type}")

    def __setattr__(self, key, value):
        if key in ['host', '_info', '_lazy_cache']:
            super(BeamClient, self).__setattr__(key, value)
        else:
            self.post(f'setvar/{key}', value)
