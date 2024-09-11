import inspect
from typing import Dict

from dataclasses import dataclass

from ..base import BeamBase
from ..utils import safe_getmembers, cached_property, signature_to_dict
from ..config import to_dict


@dataclass
class ObjectAttribute:
    name: str
    type: str
    description: str = None
    signature: Dict[str, str] = None


@dataclass
class ObjectInfo:
    type: str
    type_name: str
    attributes: dict[str, ObjectAttribute]
    hparams: dict
    vars_args: list
    name: str = None
    serialization: str = None
    version: str = None
    self: ObjectAttribute = None


class MetaAsyncResult:

    def __init__(self, obj):
        self.obj = obj
        self._value = None
        self._is_ready = None
        self._is_success = None

    @classmethod
    def from_str(cls, value, **kwargs):
        raise NotImplementedError

    @property
    def value(self):
        raise NotImplementedError

    @property
    def get(self):
        return self.value

    def kill(self):
        raise NotImplementedError

    def wait(self, timeout=None):
        raise NotImplementedError

    @property
    def hex(self):
        raise NotImplementedError

    @property
    def str(self):
        return self.hex

    @property
    def is_ready(self):
        raise NotImplementedError

    @property
    def is_success(self):
        if self._is_success is None:
            try:
                if not self.is_ready:
                    return None
                _ = self.value
                self._is_success = True
            except Exception:
                self._is_success = False
        return self._is_success

    def __str__(self):
        return self.str

    def __repr__(self):
        raise NotImplementedError

    @property
    def state(self):
        raise NotImplementedError

    @property
    def args(self):
        return None

    @property
    def kwargs(self):
        return None


class MetaDispatcher(BeamBase):

    def __init__(self, obj, *routes, name=None, asynchronous=True, predefined_attributes=None, **kwargs):

        super().__init__(name=name, **kwargs)
        self.obj = obj
        self._routes = routes
        self.asynchronous = asynchronous
        self.call_function = None
        self._routes_methods = {}

        if predefined_attributes is None:
            predefined_attributes = {}
        self._predefined_attributes = predefined_attributes

    @property
    def real_object(self):
        return self.obj

    @property
    def routes(self):
        routes = self._routes
        if routes is None or len(routes) == 0:
            routes = [name for name, attr in safe_getmembers(self.real_object, predicate=inspect.isroutine)]
        return routes

    @cached_property
    def type(self):
        if inspect.isfunction(self.real_object):
            return "function"
        elif inspect.isclass(self.real_object):
            return "class"
        elif inspect.ismethod(self.real_object):
            return "method"
        else:
            return "instance" if isinstance(self.real_object, object) else "unknown"

    @cached_property
    def route_methods(self):
        return {route: getattr(self.real_object, route) for route in self.routes}

    def getattr(self, item):
        if item in self._routes_methods:
            return self._routes_methods[item]
        else:
            raise AttributeError(f"Attribute {item} not served with {self.__class__.__name__}")

    def get_info(self):
        obj = self.real_object
        attributes = {}
        from .._version import __version__
        if obj is None:
            attributes = {k: ObjectAttribute(name=k, type='method') for k in self._predefined_attributes}
            hparams = None
            vars_args = None
        elif self.type == 'function':
            vars_args = obj.__code__.co_varnames
            hparams = None
        else:
            # vars_args = obj.__init__.__code__.co_varnames
            vars_args = None
            if hasattr(obj, 'hparams'):
                hparams = to_dict(obj.hparams)
            else:
                hparams = None

            for name, attr in safe_getmembers(obj):
                if type(name) is not str:
                    continue
                if not name.startswith('_') and inspect.isroutine(attr):
                    # attributes[name] = ObjectAttribute(name=name, type='method')
                    # add docstring to the ObjectAttribute
                    attributes[name] = ObjectAttribute(name=name, type='method', description=attr.__doc__,
                                                       signature=signature_to_dict(inspect.signature(attr)))
                elif not name.startswith('_') and not inspect.isbuiltin(attr):
                    attributes[name] = ObjectAttribute(name=name, type='variable')

            properties = inspect.getmembers(type(obj), lambda m: isinstance(m, property))
            for name, attr in properties:
                if not name.startswith('_'):
                    attributes[name] = ObjectAttribute(name=name, type='property')

        if hasattr(obj, 'name'):
            name = obj.name
        elif hasattr(obj, '__name__'):
            name = obj.__name__
        elif hasattr(obj, '__class__'):
            name = obj.__class__.__name__
        else:
            name = None

        type_name = type(obj).__name__

        signature = None
        if callable(obj):
            signature = signature_to_dict(inspect.signature(obj))

        self_attr = ObjectAttribute(name='self', type=type_name, description=obj.__doc__, signature=signature)

        return ObjectInfo(attributes=attributes, hparams=hparams, vars_args=vars_args, name=name,
                          type=self.type, type_name=type_name, version=__version__, self=self_attr)
