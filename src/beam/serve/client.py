import io
import pickle
from functools import partial
from ..path import normalize_host, BeamResource
from ..core import Processor

from .server import has_torch
if has_torch:
    import torch


class BeamClient(Processor, BeamResource):

    def __init__(self, *args, hostname=None, port=None, username=None, api_key=None, **kwargs):
        super().__init__(resource_type='client', hostname=hostname, port=port, username=username, **kwargs)
        self.host = normalize_host(hostname, port)
        self.api_key = api_key
        self.info = self.get_info()

    def get_info(self):
        raise NotImplementedError

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

    def _post(self, path, io_args, io_kwargs, **other_kwargs):
            raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.post('call/beam', *args, **kwargs)

    def getattr(self, item):
        if item.startswith('_') or item in ['info'] or not hasattr(self, 'info'):
            return super().__getattribute__(item)

        if item not in self.attributes:
            self.info = self.get_info()

        attribute_type = self.attributes[item]
        if attribute_type in ['variable', 'property']:
            return self.get(f'getvar/beam/{item}')
        elif attribute_type == 'method':
            return partial(self.post, f'alg/beam/{item}')
        raise ValueError(f"Unknown attribute type: {attribute_type}")

    def __setattr__(self, key, value):
        if key.startswith('_') or not hasattr(self, '_lazy_cache') or 'info' not in self._lazy_cache:
            super().__setattr__(key, value)
        else:
            if key in self.attributes and self.attributes[key] in ['property', 'method']:
                raise ValueError(f"Cannot set attribute: {key} (type: {self.attributes[key]})")
            self.post(f'setvar/beam/{key}', value)
