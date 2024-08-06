from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from functools import wraps
from ..utils import cached_property

from ..processor import MetaAsyncResult, MetaDispatcher
from ..utils import get_class_properties


class PoolAsyncResult(MetaAsyncResult):
    def __init__(self, future: Future):
        super().__init__(future)
        self.future = future

    @property
    def value(self):
        return self.future.result()

    def wait(self, timeout=None):
        return self.future.result(timeout=timeout)

    def kill(self):
        self.future.cancel()

    @property
    def is_ready(self):
        return self.future.done()

    @property
    def hex(self):
        return hex(id(self.future))

    def __repr__(self):
        return f"AsyncResult({self.str}, is_ready={self.is_ready}, is_success={self.is_success})"

    @property
    def state(self):
        if self.future.running():
            return "RUNNING"
        elif self.future.done():
            return "FINISHED"
        elif self.future.cancelled():
            return "CANCELLED"
        else:
            return "PENDING"


class PoolCluster:
    def __init__(self, max_workers=None, pool_type='thread'):
        PoolExecutor = ThreadPoolExecutor if pool_type == 'thread' else ProcessPoolExecutor
        self.executor = PoolExecutor(max_workers=max_workers)

    def submit(self, fn, *args, **kwargs):
        future = self.executor.submit(fn, *args, **kwargs)
        return future

    def map(self, fn, *iterables):
        return self.executor.map(fn, *iterables)

    def shutdown(self, wait=True):
        self.executor.shutdown(wait=wait)


class PoolRemoteClass:
    def __init__(self, target_class, asynchronous=False, executor=None, properties=None, pool_type='thread'):
        self._target_class = target_class
        self.properties = properties if properties is not None else []
        self.asynchronous = asynchronous
        self.pool_type = pool_type
        if executor is None:
            self.executor = PoolCluster(pool_type=pool_type)
        else:
            self.executor = executor

    @property
    def target_class(self):
        if isinstance(self._target_class, PoolAsyncResult):
            self._target_class = self._target_class.value
        return self._target_class

    def method_wrapper(self, method):
        def wrapper(*args, **kwargs):
            future = self.executor.submit(method, *args, **kwargs)
            if self.asynchronous:
                return PoolAsyncResult(future)
            else:
                return future.result()
        return wrapper

    def __getattr__(self, item):

        if item in self.properties:
            future = self.executor.submit(self.target_class._get_property, item)
            if self.asynchronous:
                return PoolAsyncResult(future)
            else:
                return future.result()

        attr = getattr(self.target_class, item)
        if callable(attr):
            return self.method_wrapper(attr)
        return attr

    def __call__(self, *args, **kwargs):
        return self.method_wrapper(self.target_class.__call__)(*args, **kwargs)


class PoolDispatcher(MetaDispatcher, PoolCluster):
    def __init__(self, obj, *routes, max_workers=None, asynchronous=True, pool_type='thread', **kwargs):
        MetaDispatcher.__init__(self, obj, *routes, asynchronous=asynchronous, **kwargs)
        PoolCluster.__init__(self, max_workers=max_workers, pool_type=pool_type)

        self.pool_type = pool_type

        if self.type == 'function':
            self.call_function = self.pooled_function_wrapper(self.obj)
        elif self.type == 'instance':
            if hasattr(self.obj, '__call__'):
                self.call_function = self.pooled_function_wrapper(self.obj.__call__)
            for route in self.routes:
                if hasattr(self.obj, route):
                    method = getattr(self.obj, route)
                    self._routes_methods[route] = self.pooled_function_wrapper(method)
        elif self.type == 'class':
            self.call_function = self.pooled_function_wrapper(self.factory_class_wrapper(self.obj))
        else:
            raise ValueError(f"Unknown type: {self.type}")

    @cached_property
    def route_methods(self):
        return self._routes_methods

    @staticmethod
    def factory_class_wrapper(cls):

        class RemoteClassWrapper(cls):
            def _get_property(self, prop):
                return getattr(self, prop)

        def wrapper(*args, **kwargs):
            res = RemoteClassWrapper(*args, **kwargs)
            return res

        return wrapper

    def pooled_function_wrapper(self, func, asynchronous=None, bypass=False):

        if asynchronous is None:
            asynchronous = self.asynchronous
        @wraps(func)
        def wrapper(*args, **kwargs):
            future = self.executor.submit(func, *args, **kwargs)
            if bypass:
                return future
            elif asynchronous:
                return PoolAsyncResult(future)
            else:
                return future.result()
        return wrapper

    def __call__(self, *args, **kwargs):
        res = self.call_function(*args, **kwargs)
        if self.type == 'class':
            properties = get_class_properties(self.obj)
            return PoolRemoteClass(res, asynchronous=self.asynchronous, properties=properties, pool_type=self.pool_type)
        else:
            return res


class ThreadedDispatcher(PoolDispatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pool_type='thread', **kwargs)


class ProcessDispatcher(PoolDispatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pool_type='process', **kwargs)
