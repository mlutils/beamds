from functools import partial, cached_property

from .utils import get_broker_url, get_backend_url
from ..processor import MetaAsyncResult, MetaDispatcher


class CeleryAsyncResult(MetaAsyncResult):

    @classmethod
    def from_str(cls, value, app=None):
        return cls(app.AsyncResult(value))

    @property
    def value(self):
        if self._value is None and self.is_ready:
            self._value = self.obj.get()  # Timeout can be adjusted
        return self._value

    def wait(self, timeout=None):

        try:
            return self.obj.get(timeout=timeout)
        except self.obj.TimeoutError:
            return None

    @property
    def hex(self):
        return self.obj.task_id

    @property
    def is_ready(self):
        if self._is_ready is None:
            self._is_ready = self.obj.ready()
        return self._is_ready

    @property
    def is_success(self):
        return self.obj.successful()

    @property
    def state(self):
        return self.obj.state

    @property
    def args(self):
        return self.obj.args

    @property
    def kwargs(self):
        return self.obj.kwargs

    def __repr__(self):
        return f"CeleryAsyncResult({self.hex}, is_ready={self.is_ready}, is_success={self.is_success})"


class CeleryDispatcher(MetaDispatcher):

    def __init__(self, *args, name=None, broker=None, backend=None,
                 broker_username=None, broker_password=None, broker_port=None, broker_scheme=None, broker_host=None,
                 backend_username=None, backend_password=None, backend_port=None, backend_scheme=None,
                 backend_host=None, asynchronous=True, log_level='INFO', **kwargs):

        # in celery obj is not used
        super().__init__(None, *args, name=name, asynchronous=asynchronous, **kwargs)

        self.broker_url = get_broker_url(broker=broker, broker_username=broker_username,
                                         broker_password=broker_password, broker_port=broker_port,
                                         broker_scheme=broker_scheme, broker_host=broker_host)

        self.backend_url = self.backend_url = get_backend_url(backend=backend, backend_username=backend_username,
                                           backend_password=backend_password, backend_port=backend_port,
                                           backend_scheme=backend_scheme, backend_host=backend_host)

        self.log_level = log_level

    @cached_property
    def broker(self):
        from celery import Celery
        app = Celery(self.name, broker=self.broker_url.url, backend=self.backend_url.url)
        app.conf.update(
            worker_log_level=self.log_level,
            broker_connection_retry_on_startup=True
        )
        return app

    def __call__(self, *args, **kwargs):
        return self.dispatch('function', *args, **kwargs)

    def poll(self, task_id, timeout=0):
        async_res = CeleryAsyncResult.from_str(task_id, app=self.broker)
        return async_res.wait(timeout=timeout)

    def metadata(self, task_id, *args, **kwargs):
        res = self.broker.AsyncResult(task_id)
        d = {'task_id': task_id, 'state': res.state, 'result': res.result,
             'traceback': res.traceback if res.state == 'FAILURE' else None, 'status': res.status,
             'children': res.children, 'retries': res.retries, "parent_id": res.parent.id if res.parent else None,
             'exception': str(res.result) if res.state == 'FAILURE' else None,
             'date_done': res.date_done if hasattr(res, 'date_done') else None,
             'runtime': res.runtime if hasattr(res, 'runtime') else None}

        return d

    def dispatch(self, attribute, *args, **kwargs):
        res = self.broker.send_task(attribute, args=args, kwargs=kwargs)
        res = CeleryAsyncResult(res)
        if self.asynchronous:
            return res
        else:
            return res.value

    def getattr(self, item):
        return partial(self.dispatch, item)

