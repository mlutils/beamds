import io
import pickle
from functools import partial
import requests
from ..core import Processor
from ..utils import lazy_property

from .beam_server import has_torch
if has_torch:
    import torch


class BeamClient(Processor):

    def __init__(self, host, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.host = host

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

    @lazy_property
    def info(self):
        raise NotImplementedError

    @property
    def attributes(self):
        return self.info['attributes']

    def get(self, path):

        raise NotImplementedError

    def post(self, path, *args, **kwargs):

        io_args = io.BytesIO()
        self.dump_function(args, io_args)
        io_args.seek(0)

        io_kwargs = io.BytesIO()
        self.dump_function(kwargs, io_kwargs)
        io_kwargs.seek(0)

        response = self._post(path, io_args, io_kwargs)

        return response

    def _post(self, path, io_args, io_kwargs):
            raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.post('call', *args, **kwargs)

    def __getattr__(self, item):

        if item.startswith('_'):
            return super(BeamClient, self).__getattr__(item)

        if item not in self.attributes:
            self.clear_cache('info')

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
