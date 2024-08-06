import ray
from ..utils import cached_property

from ..utils import get_class_properties
from ..processor import Processor
from ..path import BeamURL
from ..processor import MetaAsyncResult, MetaDispatcher


class RayAsyncResult(MetaAsyncResult):

    @classmethod
    def from_str(cls, value, **kwargs):
        return cls(ray.ObjectRef(bytes.fromhex(value)))

    @property
    def value(self):
        if self._value is None:
            self._value = ray.get(self.obj)
        return self._value

    def wait(self, timeout=None):
        # ready, not_ready = ray.wait([self.obj], num_returns=1, timeout=timeout)
        # return ready, not_ready
        try:
            res = ray.get(self.obj, timeout=timeout)
            return res
        except ray.exceptions.GetTimeoutError:
            return None

    def kill(self, force=True, recursive=True):
        ray.cancel(self.obj, force=force, recursive=recursive)

    @property
    def hex(self):
        return self.obj.hex()

    @property
    def is_ready(self):
        if not self._is_ready:
            ready = self.wait(timeout=0)
            self._is_ready = ready is not None
        return self._is_ready

    def __repr__(self):
        return f"AsyncResult({self.str}, is_ready={self.is_ready}, is_success={self.is_success})"

    @property
    def state(self):
        if self.is_ready:
            return "SUCCESS" if self.is_success else "FAILURE"
        else:
            return "PENDING"


class RayClient(Processor):

    def __init__(self, *args, name=None, address=None, host=None, port=None,
                    username=None, password=None, ray_kwargs=None, init_ray=True, **kwargs):

        super().__init__(*args, name=name, **kwargs)

        if address is None:
            if host is None and port is None:
                address = 'auto'
            else:
                if host is None:
                    host = 'localhost'
                address = BeamURL(scheme='ray', hostname=host, port=port, username=username, password=password)
                address = address.url

        ray_kwargs = ray_kwargs if ray_kwargs is not None else {}
        if init_ray:
            self.init_ray(address=address, ignore_reinit_error=True, **ray_kwargs)

    def wait(self, results, num_returns=1, timeout=None):
        results = [r.result if isinstance(r, RayAsyncResult) else r for r in results]
        return ray.wait(results, num_returns=num_returns, timeout=timeout)

    @staticmethod
    def init_ray(address=None, num_cpus=None, num_gpus=None, resources=None, labels=None, object_store_memory=None,
                 ignore_reinit_error=False, include_dashboard=True, dashboard_host='0.0.0.0',
                 dashboard_port=None, job_config=None, configure_logging=True, logging_level=None, logging_format=None,
                 log_to_driver=True, namespace=None, runtime_env=None, storage=None, **kwargs):

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

    @property
    def head_node_ip(self):
        return ray.util.get_node_ip_address()


class RayRemoteClass:

    def __init__(self, remote_class, asynchronous=False, properties=None):
        self.remote_class = remote_class
        self.asynchronous = asynchronous
        self.properties = properties if properties is not None else []

    def remote_wrapper(self, method):
        def wrapper(*args, **kwargs):
            res = method.remote(*args, **kwargs)
            if self.asynchronous:
                return RayAsyncResult(res)
            else:
                return ray.get(res)
        return wrapper

    def kill(self, no_restart=False):
        ray.kill(self.remote_class, no_restart=no_restart)

    def __getattr__(self, item):

        if item in self.properties:
            res = self.remote_class._get_property.remote(item)
            if self.asynchronous:
                return RayAsyncResult(res)
            else:
                return ray.get(res)

        return self.remote_wrapper(getattr(self.remote_class, item))

    def __call__(self, *args, **kwargs):
        res = self.remote_class.__call__.remote(*args, **kwargs)
        if self.asynchronous:
            return RayAsyncResult(res)
        else:
            return ray.get(res)


class RayDispatcher(MetaDispatcher, RayClient):

    def __init__(self, obj, *routes, name=None, address=None, host=None, port=None,
                 username=None, password=None, remote_kwargs=None, ray_kwargs=None, asynchronous=True,
                 init_ray=True, **kwargs):

        MetaDispatcher.__init__(self, obj, *routes, name=name, ray_kwargs=ray_kwargs,
                                asynchronous=asynchronous, **kwargs)
        RayClient.__init__(self, name=name, address=address, host=host, port=port, username=username,
                           password=password, ray_kwargs=ray_kwargs, init_ray=init_ray, **kwargs)

        self.remote_kwargs = remote_kwargs if remote_kwargs is not None else {}

        if self.type == 'function':
            self.call_function = self.remote_function_wrapper(self.obj)
        elif self.type == 'instance':
            if hasattr(self.obj, '__call__'):
                self.call_function = self.remote_function_wrapper(self.remote_method_wrapper(self.obj,
                                                                                             '__call__'))
            for route in self.routes:
                if hasattr(self.obj, route):
                    self._routes_methods[route] = self.remote_function_wrapper(
                        self.remote_method_wrapper(self.obj, route))

        elif self.type == 'class':
            self.call_function = self.remote_class_wrapper(self.obj)
        else:
            raise ValueError(f"Unknown type: {self.type}")

    @cached_property
    def route_methods(self):
        return self._routes_methods

    def poll(self, task_id, timeout=0):
        async_res = RayAsyncResult.from_str(task_id)
        return async_res.wait(timeout=timeout)

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
            def _get_property(self, prop):
                return getattr(self, prop)

        def wrapper(*args, **kwargs):
            res = RemoteClassWrapper.remote(*args, **kwargs)
            return res

        return wrapper

    def remote_function_wrapper(self, func, asynchronous=None, bypass=False):

        if asynchronous is None:
            asynchronous = self.asynchronous

        func = self.ray_remote(func)

        def wrapper(*args, **kwargs):
            res = func.remote(*args, **kwargs)
            if bypass:
                return res
            elif asynchronous:
                return RayAsyncResult(res)
            else:
                return ray.get(res)
        return wrapper

    def __call__(self, *args, **kwargs):
        assert self.call_function is not None, "No function to call"
        res = self.call_function(*args, **kwargs)
        if self.type == 'class':
            properties = get_class_properties(self.obj)
            return RayRemoteClass(res, asynchronous=self.asynchronous, properties=properties)
        else:
            return res
