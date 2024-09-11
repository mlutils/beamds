import json
from collections import defaultdict
from functools import partial

from .ray_dispatcher import RayClient, RayDispatcher
from .celery_worker import CeleryWorker
from .celery_dispatcher import CeleryDispatcher
from ..path import BeamURL
from ..serve import beam_server
from .async_client import AsyncClient


def beam_worker(obj, *routes, name=None, n_workers=1, daemon=False, broker=None, backend=None,
                broker_username=None, broker_password=None, broker_port=None, broker_scheme=None, broker_host=None,
                backend_username=None, backend_password=None, backend_port=None, backend_scheme=None, backend_host=None,
                **kwargs):

    worker = CeleryWorker(obj, *routes, name=name, n_workers=n_workers, daemon=daemon, broker=broker, backend=backend,
                          broker_username=broker_username, broker_password=broker_password, broker_port=broker_port,
                          broker_scheme=broker_scheme, broker_host=broker_host, backend_username=backend_username,
                          backend_password=backend_password, backend_port=backend_port, backend_scheme=backend_scheme,
                          backend_host=backend_host, **kwargs)
    worker.run()
    return worker


def beam_dispatcher(obj, *routes, name=None, n_workers=1, daemon=False, broker=None, backend=None,
                    broker_username=None, broker_password=None, broker_port=None, broker_scheme=None, broker_host=None,
                    backend_username=None, backend_password=None, backend_port=None, backend_scheme=None,
                    backend_host=None, **kwargs):

    worker = beam_worker(obj, *routes, name=name, n_workers=n_workers, daemon=daemon, broker=broker, backend=backend,
                         broker_username=broker_username, broker_password=broker_password, broker_port=broker_port,
                         broker_scheme=broker_scheme, broker_host=broker_host, backend_username=backend_username,
                         backend_password=backend_password, backend_port=backend_port, backend_scheme=backend_scheme,
                         backend_host=backend_host, log_level='CRITICAL', **kwargs)

    dispatcher = CeleryDispatcher(name=worker.name, broker=broker, backend=backend, broker_username=broker_username,
                                  broker_password=broker_password, broker_port=broker_port, broker_scheme=broker_scheme,
                                  broker_host=broker_host, backend_username=backend_username,
                                  backend_password=backend_password, backend_port=backend_port,
                                  backend_scheme=backend_scheme, backend_host=backend_host, **kwargs)
    return dispatcher


def beam_dispatcher_server(*routes, host=None, port=None, protocol='http', server_backend=None, non_blocking=False,
                           broker=None, backend=None,
                           name=None, broker_username=None, broker_password=None, broker_port=None, broker_scheme=None,
                           broker_host=None, backend_username=None, backend_password=None, backend_port=None,
                           backend_scheme=None, backend_host=None, **kwargs):

    predefined_attributes = {k: 'method' for k in routes}

    dispatcher = CeleryDispatcher(name=name, broker=broker, backend=backend,
                                  broker_username=broker_username, broker_password=broker_password,
                                  broker_port=broker_port, broker_scheme=broker_scheme, broker_host=broker_host,
                                  backend_username=backend_username, backend_password=backend_password,
                                  backend_port=backend_port, backend_scheme=backend_scheme, backend_host=backend_host,
                                  **kwargs)

    server = beam_server(obj=dispatcher, protocol=protocol, host=host, port=port, backend=server_backend,
                         predefined_attributes=predefined_attributes, non_blocking=non_blocking, **kwargs)
    return server


def async_client(uri, hostname=None, port=None, username=None, **kwargs):

    scheme = uri.split('://')[0]
    uri = uri.removeprefix('async-')

    uri = BeamURL.from_string(uri)

    if uri.hostname is not None:
        hostname = uri.hostname

    if uri.port is not None:
        port = uri.port

    if uri.username is not None:
        username = uri.username

    query = uri.query
    for k, v in query.items():
        kwargs[k.replace('-', '_')] = v

    if 'tls' not in kwargs:
        kwargs['tls'] = True if 'https' in scheme else False

    return AsyncClient(hostname=hostname, port=port, username=username, **kwargs)


def ray_client(uri, hostname=None, port=None, username=None, password=None, ray_kwargs=None, asynchronous=True,
               remote_kwargs=None):

    uri = BeamURL.from_string(uri)

    if uri.hostname is not None:
        hostname = uri.hostname

    address = None
    if hostname in ['localhost', '127.0.0.1', 'auto', None]:
        address = 'auto'
    else:
        if uri.port is not None:
            port = uri.port

        if uri.username is not None:
            username = uri.username

        if uri.password is not None:
            password = uri.password

    kwargs = defaultdict(lambda: None)
    query = uri.query
    for k, v in query.items():
        kwargs[k.replace('-', '_')] = v

    if kwargs['ray_kwargs'] is not None:
        ray_kwargs = json.loads(ray_kwargs['ray_kwargs'])

    if kwargs['remote_kwargs'] is not None:
        remote_kwargs = json.loads(remote_kwargs['remote_kwargs'])

    if kwargs['asynchronous'] is not None:
        asynchronous = kwargs['asynchronous']

    if type(asynchronous) is str:
        asynchronous = asynchronous.lower() == 'true'

    RayClient(address=address, host=hostname, port=port, username=username, password=password, ray_kwargs=ray_kwargs)

    kwargs = dict(init_ray=False)
    if remote_kwargs is not None:
        kwargs['remote_kwargs'] = remote_kwargs
    if asynchronous is not None:
        kwargs['asynchronous'] = asynchronous

    return partial(RayDispatcher, **kwargs)