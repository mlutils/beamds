from ..path import BeamURL
from ..utils import beam_base_port, beam_service_port


def get_broker_url(broker=None, broker_username=None, broker_password=None, broker_port=None, broker_scheme=None,
                   broker_host=None):

    default_ports = {'amqp': 5672}
    if broker_scheme is None:
        broker_scheme = 'amqp'
    if broker_host is None:
        broker_host = 'localhost'
        if broker_port is None:
            beam_ports = {'amqp': beam_service_port('RABBITMQ_PORT')}
            broker_port = beam_ports.get(broker_scheme, None)

    if broker_port is None:
        broker_port = default_ports.get(broker_scheme, None)

    broker_url = BeamURL(url=broker, username=broker_username, password=broker_password, port=broker_port,
                              scheme=broker_scheme, hostname=broker_host)

    return broker_url


def get_backend_url(backend=None, backend_username=None, backend_password=None, backend_port=None, backend_scheme=None,
                    backend_host=None):

    default_ports = {'amqp': 5672, 'redis': 6379}
    if backend_scheme is None:
        backend_scheme = 'redis'
    if backend_host is None:
        backend_host = 'localhost'
        if backend_port is None:
            beam_ports = {'amqp': beam_service_port('RABBITMQ_PORT'), 'redis': beam_service_port('REDIS_PORT')}
            backend_port = beam_ports.get(backend_scheme, None)

    if backend_port is None:
        backend_port = default_ports.get(backend_scheme, None)

    backend_url = BeamURL(url=backend, username=backend_username, password=backend_password, port=backend_port,
                              scheme=backend_scheme, hostname=backend_host)
    return backend_url
