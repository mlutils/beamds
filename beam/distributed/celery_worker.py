import inspect
from typing import Tuple, Dict, Any
from uuid import uuid4 as uuid
from multiprocessing import Process
from ..utils import cached_property
from dataclasses import dataclass, field
import warnings

from .utils import get_broker_url, get_backend_url
from ..processor import Processor
from ..logging import beam_logger as logger
from ..utils import safe_getmembers


class CeleryWorker(Processor):

    def __init__(self, obj, *routes, name=None, n_workers=1, daemon=False, broker=None, backend=None,
                 broker_username=None, broker_password=None, broker_port=None, broker_scheme=None, broker_host=None,
                 backend_username=None, backend_password=None, backend_port=None, backend_scheme=None,
                 backend_host=None, log_level='INFO', **kwargs):

        if name is None:
            try:
                import coolname
                name = coolname.generate_slug(2)
            except ImportError:
                name = str(uuid())

        super().__init__(name=name, n_workers=n_workers, daemon=daemon, **kwargs)

        self.broker_url = get_broker_url(broker=broker, broker_username=broker_username,
                                         broker_password=broker_password, broker_port=broker_port,
                                         broker_scheme=broker_scheme, broker_host=broker_host)

        self.backend_url = get_backend_url(backend=backend, backend_username=backend_username,
                                           backend_password=backend_password, backend_port=backend_port,
                                           backend_scheme=backend_scheme, backend_host=backend_host)

        self.obj = obj
        self.n_workers = self.get_hparam('n_workers')
        self.daemon = self.get_hparam('daemon')
        self._routes = routes
        self.log_level = log_level

        logger.info(f"Broker: {self.broker_url.url}, Backend: {self.backend_url.url}, "
                    f"n_workers: {self.n_workers}, daemon: {self.daemon}")

        logger.info(f"Setting up a Celery worker: app name: {self.name} broker: {self.broker_url.url} "
                    f"backend: {self.backend_url.url}")

    @cached_property
    def type(self):
        if inspect.isfunction(self.obj):
            return 'function'
        return 'class'

    @cached_property
    def broker(self):
        from celery import Celery
        from celery.exceptions import SecurityWarning
        warnings.filterwarnings("ignore", category=SecurityWarning)
        app = Celery(self.name, broker=self.broker_url.url, backend=self.backend_url.url)

        app.conf.update(
            worker_log_level=self.log_level,
            broker_connection_retry_on_startup=True
        )

        return app

    def start_worker(self):
        from celery.apps.worker import Worker
        worker = Worker(app=self.broker, loglevel='info', traceback=True)
        worker.start()

    @property
    def routes(self):
        routes = self._routes
        if routes is None or len(routes) == 0:
            routes = [name for name, attr in safe_getmembers(self.obj)
                      if type(name) is str and not name.startswith('_') and inspect.isroutine(attr)]

        return routes

    def run(self, *routes):

        if self.type == 'function':
            self.broker.task(name='function')(self.obj)
        else:
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


@dataclass
class Task:
    req_id: str
    args: Tuple = tuple()
    kwargs: Dict = field(default_factory=dict)
    done: bool = False
    in_progress: bool = False
    result: Any = None
    success: bool = False
    exception: Exception = None
    traceback: str = None
    start_time: float = None
    end_time: float = None

