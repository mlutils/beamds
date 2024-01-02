from .dispatcher import BeamDispatcher
from ..core import Processor
from ..serve.http_server import HTTPServer
import websockets
from ..logger import beam_logger as logger


class BeamAsyncServer(Processor):

    def __init__(self, *routes, name=None, broker=None, backend=None,
                 broker_username=None, broker_password=None, broker_port=None, broker_scheme=None, broker_host=None,
                 backend_username=None, backend_password=None, backend_port=None, backend_scheme=None,
                 backend_host=None, use_torch=True, batch=None, max_wait_time=1.0, max_batch_size=10,
                 tls=False, n_threads=4, application=None, callback=None, **kwargs):

        super().__init__(name=name, **kwargs)

        predefined_attributes = {k: 'method' for k in routes}

        self.dispatcher = BeamDispatcher(name=name, broker=broker, backend=backend,
                                         broker_username=broker_username, broker_password=broker_password,
                                         broker_port=broker_port, broker_scheme=broker_scheme, broker_host=broker_host,
                                         backend_username=backend_username, backend_password=backend_password,
                                         backend_port=backend_port, backend_scheme=backend_scheme,
                                         backend_host=backend_host, **kwargs)

        self.http_server = HTTPServer(obj=self.dispatcher, use_torch=use_torch, batch=batch,
                                      max_wait_time=max_wait_time, max_batch_size=max_batch_size,
                                      tls=tls, n_threads=n_threads, application=application,
                                      predefined_attributes=predefined_attributes, **kwargs)

        from celery.signals import task_postrun
        if callback is None:
            self.callback = task_postrun.connect(self.job_done)
        else:
            self.callback = task_postrun.connect(callback)

    def job_done(self, task_id, state, result):
        logger.info(f"Task {task_id} finished with state {state}.")