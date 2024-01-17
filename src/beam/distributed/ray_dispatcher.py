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


class RayDispatcher(Processor, RayCluster):

    def __init__(self, obj, *routes, name=None, address=None, host=None, port=None,
                 username=None, password=None, remote_kwargs=None, ray_kwargs=None, **kwargs):

        if address is None:
            if host is None and port is None:
                address = 'auto'
            else:
                if host is None:
                    host = 'localhost'
                address = BeamURL(host=host, port=port, username=username, password=password)
                address = address.url

        self.address = address
        self.obj = obj
        self.remote_kwargs = remote_kwargs if remote_kwargs is not None else {}
        self.ray_kwargs = ray_kwargs if ray_kwargs is not None else {}
        self.routes = routes
        self._ray_initialized = False

        super().__init__(name=name, **kwargs)

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

    @staticmethod
    def remote_wrapper(obj):
        def wrapper(*args, **kwargs):
            return obj.remote(*args, **kwargs)
        return wrapper

    def run(self, *routes, remote_kwargs=None):

        remote_kwargs = mixin_dictionaries(remote_kwargs, self.remote_kwargs)

        if not self._ray_initialized:
            self.init_ray(address=self.address, **self.ray_kwargs)

        if self.type == 'function':
            obj = ray.remote(**remote_kwargs)(self.obj)
            obj = self.remote_wrapper(obj)
        elif self.type == 'instance':
            if len(routes) == 0:
                routes = self.routes
            for route in routes:
                self.broker.task(name=route)(getattr(self.obj, route))

        if self.n_workers == 1 and not self.daemon:
            # Run in the main process
            self.start_worker()
        else:
            # Start multiple workers in separate processes
            processes = [Process(target=self.start_worker, daemon=self.daemon) for _ in range(self.n_workers)]
            for p in processes:
                p.start()