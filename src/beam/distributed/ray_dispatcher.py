import inspect

from ..core import Processor
from ..path import BeamURL
from ..utils import lazy_property, mixin_dictionaries
import ray


class RayCluster:

    @staticmethod
    def init_ray(address=None, num_cpus=None, num_gpus=None, resources=None, labels=None, object_store_memory=None,
                 ignore_reinit_error=False, include_dashboard=True, dashboard_host='0.0.0.0',
                 dashboard_port=None, job_config=None, configure_logging=True, logging_level=None, logging_format=None,
                 log_to_driver=True, namespace=None, runtime_env=None, storage=None):
        kwargs = {}
        if logging_level is not None:
            kwargs['logging_level'] = logging_level

        ray.init(address=address, num_cpus=num_cpus, num_gpus=num_gpus, resources=resources, labels=labels,
                 object_store_memory=object_store_memory, ignore_reinit_error=ignore_reinit_error,
                 job_config=job_config, configure_logging=configure_logging, logging_format=logging_format,
                 log_to_driver=log_to_driver, namespace=namespace, storage=storage,
                 runtime_env=runtime_env, dashboard_port=dashboard_port,
                 include_dashboard=include_dashboard, dashboard_host=dashboard_host, **kwargs)

    @staticmethod
    def shutdown_ray():
        ray.shutdown()


class RemoteClass:

    def __init__(self, remote_class, asynchronous=True):
        self.remote_class = remote_class
        self.asynchronous = asynchronous

    def remote_wrapper(self, method):
        def wrapper(*args, **kwargs):
            res = method.remote(*args, **kwargs)
            if self.asynchronous:
                return res
            else:
                return ray.get(res)
        return wrapper

    def __getattr__(self, item):
        return self.remote_wrapper(getattr(self.remote_class, item))

    def __call__(self, *args, **kwargs):
        res = self.remote_class.remote(*args, **kwargs)
        if self.asynchronous:
            return res
        else:
            return ray.get(res)


class RayDispatcher(Processor, RayCluster):

    def __init__(self, obj, *routes, name=None, address=None, host=None, port=None,
                 username=None, password=None, remote_kwargs=None, ray_kwargs=None, asynchronous=True, **kwargs):

        if address is None:
            if host is None and port is None:
                address = 'auto'
            else:
                if host is None:
                    host = 'localhost'
                address = BeamURL(host=host, port=port, username=username, password=password)
                address = address.url

        self.obj = obj
        self._routes = routes
        remote_kwargs = remote_kwargs if remote_kwargs is not None else {}
        ray_kwargs = ray_kwargs if ray_kwargs is not None else {}
        self.init_ray(address=address, **ray_kwargs)
        self.asynchronous = asynchronous

        self.call_function = None
        self.routes_methods = {}
        if self.type == 'function':
            self.call_function = self.remote_wrapper(ray.remote(**remote_kwargs)(self.obj))
        elif self.type == 'instance':
            if hasattr(self.obj, '__call__'):
                self.call_function = self.remote_wrapper(ray.remote(**remote_kwargs)(self.obj.__call__))
            for route in self.routes:
                if hasattr(self.obj, route):
                    self.routes_methods[route] = self.remote_wrapper(ray.remote(**remote_kwargs)(getattr(self.obj, route)))

        elif self.type == 'class':
            self.call_function = self.remote_wrapper(ray.remote(**remote_kwargs)(self.obj))
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

    def remote_wrapper(self, method):
        def wrapper(*args, **kwargs):
            res = method.remote(*args, **kwargs)
            if self.asynchronous:
                return res
            else:
                return ray.get(res)
        return wrapper

    def __getattr__(self, item):
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
