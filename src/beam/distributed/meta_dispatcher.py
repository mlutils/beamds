import inspect

from ..core import Processor
from ..utils import lazy_property


class MetaAsyncResult:

    def __init__(self, obj):
        self.obj = obj
        self._value = None
        self._is_ready = None
        self._is_success = None

    @property
    def value(self):
        raise NotImplementedError

    @property
    def get(self):
        return self.value

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


class MetaDispatcher(Processor):

    def __init__(self, obj, *routes, name=None, asynchronous=True, **kwargs):

        super().__init__(name=name, **kwargs)
        self.obj = obj
        self._routes = routes
        self.asynchronous = asynchronous

        self.call_function = None
        self._routes_methods = {}


    @property
    def routes(self):
        routes = self._routes
        if routes is None or len(routes) == 0:
            routes = [name for name, attr in inspect.getmembers(self.obj)
                      if type(name) is str and not name.startswith('_') and
                      (inspect.ismethod(attr) or inspect.isfunction(attr))]
        return routes

    @lazy_property
    def type(self):
        if inspect.isfunction(self.obj):
            return "function"
        elif inspect.isclass(self.obj):
            return "class"
        elif inspect.ismethod(self.obj):
            return "method"
        else:
            return "instance" if isinstance(self.obj, object) else "unknown"

    def __getattr__(self, item):
        if item.startswith('_') or item == '_init_is_done' or not hasattr(self, '_init_is_done'):
            return super().__getattribute__(item)
        if item in self._routes_methods:
            return self._routes_methods[item]
        else:
            raise AttributeError(f"Attribute {item} not served with {self.__class__.__name__}")