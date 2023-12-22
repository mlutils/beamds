from functools import partial

from ..core import Processor
from ..path import BeamURL
from ..utils import lazy_property


class BeamDispatcher(Processor):

    def __init__(self, *args, name=None, broker=None, backend=None,
                 broker_username=None, broker_password=None, broker_port=None, broker_scheme=None, broker_host=None,
                 backend_username=None, backend_password=None, backend_port=None, backend_scheme=None, backend_host=None,
                 **kwargs):

        super().__init__(*args, name=name, **kwargs)

        if broker_scheme is None:
            broker_scheme = 'amqp'
        self.broker_url = BeamURL(url=broker, username=broker_username, password=broker_password, port=broker_port,
                           scheme=broker_scheme, host=broker_host)

        if backend_scheme is None:
            backend_scheme = 'redis'
        self.backend_url = BeamURL(url=backend, username=backend_username, password=backend_password, port=backend_port,
                                   scheme=backend_scheme, host=backend_host)

    @lazy_property
    def broker(self):
        from celery import Celery
        return Celery(self.name, broker=self.broker_url.url, backend=self.backend_url.url)

    def __call__(self, *args, **kwargs):
        return self.dispatch('function', *args, **kwargs)

    def dispatch(self, attribute, *args, **kwargs):
        return self.broker.send_task(attribute, args=args, kwargs=kwargs)

    def __getattr__(self, item):
        if item.startswith('_'):
            return super().__getattribute__(item)
        return partial(self.dispatch, item)

