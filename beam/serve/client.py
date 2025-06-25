import io
import pickle
from functools import partial, update_wrapper
from ..path import normalize_host, BeamResource
from ..base import BeamBase

from ..importer import lazy_importer as lzi
from ..importer import torch
from ..utils import dict_to_signature
from ..logging import beam_logger as logger


class BeamClient(BeamBase, BeamResource):

    def __init__(self, *args, hostname=None, port=None, username=None, api_key=None, **kwargs):

        BeamBase.__init__(self, **kwargs)
        BeamResource.__init__(self, resource_type='client', hostname=hostname, port=port, username=username, **kwargs)

        self.host = normalize_host(hostname, port)
        self.api_key = api_key
        self.info = self.get_info()
        self._backwards_compatible = None
        if 'self' in self.info:
            self.__doc__ = self.info['self']['description']

    def __dir__(self):
        # d = list(super().__dir__()) + list(self.attributes.keys())
        d = list(self.attributes.keys())
        return sorted(d)

    @property
    def type(self):
        if 'type' in self.info:
            return self.info['type']
        else:
            logger.debug(f"Backcompatibility mode detected for {self.host}")
            return 'instance'

    def to_function(self):

        assert self.type == 'function', f"Cannot convert {self.type} to function"

        def _func(*args, **kwargs):
            return self(*args, **kwargs)

        _func.__name__ = self.info.get('name', 'function')

        signature = None
        if 'self' in self.info:
            _func.__doc__ = self.info['self']['description']
            signature = self.info['self'].get('signature', None)

        if signature is not None:
            _func = BeamClient._create_function_with_signature(_func, dict_to_signature(signature))

        return _func

    @classmethod
    def client(cls, *args, **kwargs):
        obj = cls(*args, **kwargs)
        if obj.type == 'function':
            return obj.to_function()
        return obj

    def get_info(self):
        raise NotImplementedError

    @property
    def load_function(self):
        if self.serialization == 'torch':
            if not lzi.has('torch'):
                raise ImportError('Cannot use torch serialization without torch installed')
            return torch.load
        else:
            return pickle.load

    @property
    def lf_kwargs(self):
        if self.serialization == 'torch':
            torch = lzi.torch
            if not torch:
                raise ImportError('Cannot use torch serialization without torch installed')
            return {'weights_only': True}
        else:
            return {}

    @property
    def dump_function(self):
        if self.serialization == 'torch':
            torch = lzi.torch
            if not torch:
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

    @property
    def backwards_compatible(self):
        if self._backwards_compatible is None:
            item, attribute = next(iter(self.attributes.items()))
            if type(attribute) is dict:
                backwards_compatible = False
            else:
                backwards_compatible = True
            self._backwards_compatible = backwards_compatible
        return self._backwards_compatible

    def getattr(self, item):
        if item.startswith('_') or item in ['info'] or not hasattr(self, 'info'):
            return super().__getattribute__(item)

        if item not in self.attributes:
            self.info = self.get_info()

        attribute = self.attributes[item]
        if self.backwards_compatible:
            attribute_type = attribute
        else:
            attribute_type = attribute['type']

        if attribute_type in ['variable', 'property']:
            return self.get(f'getvar/beam/{item}')
        elif attribute_type == 'method':
            func = partial(self.post, f'alg/beam/{item}')

            if not self.backwards_compatible:
                func.__name__ = item
                func.__doc__ = self.attributes[item]['description']
                if self.attributes[item]['signature'] is not None:
                    func = self._create_function_with_signature(func,
                                                                dict_to_signature(self.attributes[item]['signature']))
            return func

        raise ValueError(f"Unknown attribute type: {attribute_type}")

    @staticmethod
    def _create_function_with_signature(func, signature):
        # Create a new function with the desired signature
        def new_func(*args, **kwargs):
            return func(*args, **kwargs)

        # Update the new function to have the same properties as the original
        update_wrapper(new_func, func)

        # Set the new function's signature
        new_func.__signature__ = signature

        return new_func

    @staticmethod
    def _create_class_with_signature(cls, signature):
        # Create a new init method with the desired signature
        def new_init(self, *args, **kwargs):
            # Call the original init method
            cls.__init__(self, *args, **kwargs)

        # Update the init method with the original class __init__ properties
        update_wrapper(new_init, cls.__init__)

        # Set the new init method's signature
        new_init.__signature__ = signature

        # Replace the class __init__ method with the new one
        cls.__init__ = new_init

        return cls

    def __setattr__(self, key, value):
        if key.startswith('_') or not hasattr(self, '_lazy_cache') or 'info' not in self._lazy_cache:
            super().__setattr__(key, value)
        else:
            if key in self.attributes and self.attributes[key] in ['property', 'method']:
                raise ValueError(f"Cannot set attribute: {key} (type: {self.attributes[key]})")
            self.post(f'setvar/beam/{key}', value)
