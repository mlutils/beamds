from pathlib import PurePath

from ..path import BeamURL


def beam_server(obj, protocol='http', host=None, port=None, backend=None, non_blocking=False, **kwargs):

    run_kwargs = {}
    if 'http' in protocol:
        if 'tls' not in kwargs:
            kwargs['tls'] = True if protocol == 'https' else False
        if backend is not None:
            run_kwargs['server'] = backend
        from .http_server import HTTPServer
        server = HTTPServer(obj, **kwargs)
    elif 'grpc' in protocol:
        from .grpc_server import GRPCServer
        server = GRPCServer(obj, **kwargs)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    if non_blocking:
        server.run_non_blocking(host=host, port=port, **run_kwargs)
    else:
        server.run(host=host, port=port, **run_kwargs)

    return server


def beam_client(uri, hostname=None, port=None, username=None, api_key=None, **kwargs):

    uri = uri.removeprefix('beam-')

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

    if api_key is None and 'api_key' in kwargs:
        api_key = kwargs.pop('api_key')

    scheme = uri.scheme
    if 'http' in scheme:
        if 'tls' not in kwargs:
            kwargs['tls'] = True if scheme == 'https' else False
        from .http_client import HTTPClient
        return HTTPClient.client(hostname=hostname, port=port, username=username, api_key=api_key, **kwargs)
    elif 'grpc' in scheme:
        from .grpc_client import GRPCClient
        return GRPCClient.client(hostname=hostname, port=port, username=username, api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unknown protocol: {scheme}")


def triton_client(uri, hostname=None, port=None, model_name=None, model_version=None, **kwargs):

    uri = uri.removeprefix('triton-')

    uri = BeamURL.from_string(uri)

    if uri.hostname is not None:
        hostname = uri.hostname

    if uri.port is not None:
        port = uri.port

    if uri.path is not None:
        path = uri.path
        parts = PurePath(path).parts
        if len(parts) > 0:
            model_name = parts[0]
        if len(parts) > 1:
            model_version = parts[1]

    query = uri.query
    for k, v in query.items():
        kwargs[k.replace('-', '_')] = v

    from .triton import TritonClient
    return TritonClient(scheme=uri.scheme, hostname=hostname, port=port, model_name=model_name,
                        model_version=model_version, **kwargs)

