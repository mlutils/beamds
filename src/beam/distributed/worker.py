import inspect

from ..core import Processor
from ..utils import lazy_property
from ..path import BeamURL


class BeamWorker(Processor):

    def __init__(self, obj, *args, name=None, broker=None, backend=None,
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

        self.obj = obj

    @lazy_property
    def type(self):
        if inspect.isfunction(self.obj):
            return 'function'
        return 'class'

    @lazy_property
    def celery(self):
        from celery import Celery
        return Celery(self.name, broker=self.broker_url.url, backend=self.backend_url.url)

    def run(self, *attributes):
        if self.type == 'function':
            self.celery.task(name='function')(self.obj)
        else:
            for at in attributes:
                self.celery.task(name=at)(getattr(self.obj, at))



