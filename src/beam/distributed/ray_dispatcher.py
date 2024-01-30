import inspect

from ..core import Processor
from ..path import BeamURL
from ..utils import lazy_property, MetaInitIsDoneVerifier
import ray


class AsyncResult:

    def __init__(self, obj):
        self.obj = obj
        self._value = None
        self._is_ready = None
        self._is_success = None

    @property
    def value(self):
        if self._value is None:
            self._value = ray.get(self.obj)
        return self._value

    @property
    def get(self):
        return self.value

    def wait(self, timeout=None):
        ready, not_ready = ray.wait([self.obj], num_returns=1, timeout=timeout)
        return ready, not_ready

    @property
    def hex(self):
        return self.obj.hex()

    @property
    def str(self):
        return self.hex

    @property
    def is_ready(self):
        if not self._is_ready:
            ready, _ = self.wait(timeout=0)
            self._is_ready = len(ready) == 1
        return self._is_ready

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
        return f"AsyncResult({self.str}, is_ready={self.is_ready}, is_success={self.is_success})"


class RayCluster(Processor):

    def __init__(self, *args, name=None, address=None, host=None, port=None,
                    username=None, password=None, ray_kwargs=None, **kwargs):

        super().__init__(*args, name=name, **kwargs)

        if address is None:
            if host is None and port is None:
                address = 'auto'
            else:
                if host is None:
                    host = 'localhost'
                address = BeamURL(host=host, port=port, username=username, password=password)
                address = address.url

        ray_kwargs = ray_kwargs if ray_kwargs is not None else {}
        self.init_ray(address=address, ignore_reinit_error=True, **ray_kwargs)

    def wait(self, results, num_returns=1, timeout=None):
        results = [r.result if isinstance(r, AsyncResult) else r for r in results]
        return ray.wait(results, num_returns=num_returns, timeout=timeout)

    @staticmethod
    def init_ray(address=None, num_cpus=None, num_gpus=None, resources=None, labels=None, object_store_memory=None,
                 ignore_reinit_error=False, include_dashboard=True, dashboard_host='0.0.0.0',
                 dashboard_port=None, job_config=None, configure_logging=True, logging_level=None, logging_format=None,
                 log_to_driver=True, namespace=None, runtime_env=None, storage=None):
        kwargs = {}
        if logging_level is not None:
            kwargs['logging_level'] = logging_level

        if not ray.is_initialized():
            ray.init(address=address, num_cpus=num_cpus, num_gpus=num_gpus, resources=resources, labels=labels,
                     object_store_memory=object_store_memory, ignore_reinit_error=ignore_reinit_error,
                     job_config=job_config, configure_logging=configure_logging, logging_format=logging_format,
                     log_to_driver=log_to_driver, namespace=namespace, storage=storage,
                     runtime_env=runtime_env, dashboard_port=dashboard_port,
                     include_dashboard=include_dashboard, dashboard_host=dashboard_host, **kwargs)

    @staticmethod
    def shutdown():
        ray.shutdown()


class RemoteClass:

    def __init__(self, remote_class, asynchronous=False):
        self.remote_class = remote_class
        self.asynchronous = asynchronous

    def remote_wrapper(self, method):
        def wrapper(*args, **kwargs):
            res = method.remote(*args, **kwargs)
            if self.asynchronous:
                return AsyncResult(res)
            else:
                return ray.get(res)
        return wrapper

    def kill(self, no_restart=False):
        ray.kill(self.remote_class, no_restart=no_restart)

    def __getattr__(self, item):
        return self.remote_wrapper(getattr(self.remote_class, item))

    def __call__(self, *args, **kwargs):
        res = self.remote_class.__call__.remote(*args, **kwargs)
        if self.asynchronous:
            return AsyncResult(res)
        else:
            return ray.get(res)


class RayDispatcher(RayCluster, metaclass=MetaInitIsDoneVerifier):

    def __init__(self, obj, *routes, name=None, address=None, host=None, port=None,
                 username=None, password=None, remote_kwargs=None, ray_kwargs=None, asynchronous=False, **kwargs):

        super().__init__(name=name, address=address, host=host, port=port, username=username, password=password,
                         ray_kwargs=ray_kwargs, **kwargs)

        self.obj = obj
        self._routes = routes
        self.remote_kwargs = remote_kwargs if remote_kwargs is not None else {}
        self.asynchronous = asynchronous

        self.call_function = None
        self.routes_methods = {}

        if self.type == 'function':
            self.call_function = self.remote_function_wrapper(self.obj)
        elif self.type == 'instance':
            if hasattr(self.obj, '__call__'):
                self.call_function = self.remote_method_wrapper(self.obj, '__call__')
            for route in self.routes:
                if hasattr(self.obj, route):
                    self.routes_methods[route] = self.remote_function_wrapper(
                        self.remote_method_wrapper(self.obj, route))

        elif self.type == 'class':
            self.call_function = self.remote_class_wrapper(self.obj)
        else:
            raise ValueError(f"Unknown type: {self.type}")

        super().__init__(name=name, **kwargs)

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

    @property
    def ray_remote(self):
        return ray.remote(**self.remote_kwargs) if len(self.remote_kwargs) else ray.remote

    def remote_method_wrapper(self, obj, method_name):
        method = getattr(obj, method_name)
        def wrapper(*args, **kwargs):
            res = method(*args, **kwargs)
            return res

        return wrapper

    def remote_class_wrapper(self, cls):

        @self.ray_remote
        class RemoteClassWrapper(cls):
            pass

        def wrapper(*args, **kwargs):
            res = RemoteClassWrapper.remote(*args, **kwargs)
            return res

        return wrapper

    def remote_function_wrapper(self, func):

        func = self.ray_remote(func)
        def wrapper(*args, **kwargs):
            res = func.remote(*args, **kwargs)
            if self.asynchronous:
                return AsyncResult(res)
            else:
                return ray.get(res)
        return wrapper

    def __getattr__(self, item):
        if item == 'init_is_done' or not hasattr(self, 'init_is_done'):
            return super().__getattr__(item)
        if item in self.routes_methods:
            return self.routes_methods[item]
        else:
            raise AttributeError(f"Attribute {item} not served with ray")

    def __call__(self, *args, **kwargs):
        assert self.call_function is not None, "No function to call"
        res = self.call_function(*args, **kwargs)
        if self.type == 'class':
            return RemoteClass(res, asynchronous=self.asynchronous)
        else:
            return res
