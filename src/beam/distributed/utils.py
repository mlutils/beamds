from ..path import BeamURL
from ..utils import beam_base_port


def get_broker_url(broker=None, broker_username=None, broker_password=None, broker_port=None, broker_scheme=None,
                   broker_host=None):
    base_port = None
    if broker_scheme is None:
        broker_scheme = 'amqp'
    if broker_host is None:
        broker_host = 'localhost'
        base_port = beam_base_port()
    if broker_port is None:
        if base_port is None:
            broker_port = 5672 if broker_scheme == 'amqp' else 6379
        else:
            broker_port = base_port + 72 if broker_scheme == 'amqp' else base_port + 79

    broker_url = BeamURL(url=broker, username=broker_username, password=broker_password, port=broker_port,
                              scheme=broker_scheme, hostname=broker_host)

    return broker_url


def get_backend_url(backend=None, backend_username=None, backend_password=None, backend_port=None, backend_scheme=None,
                    backend_host=None):

    base_port = None
    if backend_scheme is None:
        backend_scheme = 'redis'
    if backend_host is None:
        backend_host = 'localhost'
        base_port = beam_base_port()
    if backend_port is None:
        if base_port is None:
            backend_port = 6379 if backend_scheme == 'redis' else None
        else:
            backend_port = base_port + 79 if backend_scheme == 'redis' else None

    backend_url = BeamURL(url=backend, username=backend_username, password=backend_password, port=backend_port,
                              scheme=backend_scheme, hostname=backend_host)
    return backend_url
