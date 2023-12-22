from .worker import BeamWorker
from .dispatcher import BeamDispatcher
from ..serve import beam_server


def beam_worker(obj, *args, name=None, n_workers=1, daemon=False, broker=None, backend=None,
                broker_username=None, broker_password=None, broker_port=None, broker_scheme=None, broker_host=None,
                backend_username=None, backend_password=None, backend_port=None, backend_scheme=None, backend_host=None,
                **kwargs):

    worker = BeamWorker(obj, *args, name=name, n_workers=n_workers, daemon=daemon, broker=broker, backend=backend,
                        broker_username=broker_username, broker_password=broker_password, broker_port=broker_port,
                        broker_scheme=broker_scheme, broker_host=broker_host, backend_username=backend_username,
                        backend_password=backend_password, backend_port=backend_port, backend_scheme=backend_scheme,
                        backend_host=backend_host, **kwargs)
    worker.run()


def beam_dispatcher_server(*routes, host=None, port=None, protocol='http', server_backend=None, non_blocking=False,
                           broker=None, backend=None,
                           name=None, broker_username=None, broker_password=None, broker_port=None, broker_scheme=None,
                           broker_host=None, backend_username=None, backend_password=None, backend_port=None,
                           backend_scheme=None, backend_host=None, **kwargs):

    predefined_attributes = {k: 'method' for k in routes}

    dispatcher = BeamDispatcher(name=name, broker=broker, backend=backend,
                                broker_username=broker_username, broker_password=broker_password,
                                broker_port=broker_port, broker_scheme=broker_scheme, broker_host=broker_host,
                                backend_username=backend_username, backend_password=backend_password,
                                backend_port=backend_port, backend_scheme=backend_scheme, backend_host=backend_host,
                                **kwargs)

    server = beam_server(obj=dispatcher, protocol=protocol, host=host, port=port, backend=server_backend,
                         predefined_attributes=predefined_attributes, non_blocking=non_blocking, **kwargs)
    return server
