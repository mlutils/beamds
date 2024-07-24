import io
import pickle
from functools import partial, update_wrapper
from ..path import normalize_host, BeamResource
from ..base import BeamBase

from .server import has_torch
from ..utils import dict_to_signature

if has_torch:
    import torch


class BeamClient(BeamBase, BeamResource):

    def __init__(self, *args, hostname=None, port=None, username=None, api_key=None, **kwargs):

        BeamBase.__init__(self, **kwargs)
        BeamResource.__init__(self, resource_type='client', hostname=hostname, port=port, username=username, **kwargs)

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

        attribute_type = self.attributes[item]['type']
        if attribute_type in ['variable', 'property']:
            return self.get(f'getvar/beam/{item}')
        elif attribute_type == 'method':
            func = partial(self.post, f'alg/beam/{item}')
            func.__name__ = item
            func.__doc__ = self.attributes[item]['description']
            if self.attributes[item]['signature'] is not None:
                func = self._create_function_with_signature(func, dict_to_signature(self.attributes[item]['signature']))
            return func
        raise ValueError(f"Unknown attribute type: {attribute_type}")

    def _create_function_with_signature(self, func, signature):
        # Create a new function with the desired signature
        def new_func(*args, **kwargs):
            return func(*args, **kwargs)

        # Update the new function to have the same properties as the original
        update_wrapper(new_func, func)

        # Set the new function's signature
        new_func.__signature__ = signature

        return new_func

    def __setattr__(self, key, value):
        if key.startswith('_') or not hasattr(self, '_lazy_cache') or 'info' not in self._lazy_cache:
            super().__setattr__(key, value)
        else:
            if key in self.attributes and self.attributes[key] in ['property', 'method']:
                raise ValueError(f"Cannot set attribute: {key} (type: {self.attributes[key]})")
            self.post(f'setvar/beam/{key}', value)
