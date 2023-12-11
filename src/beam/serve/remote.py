from ..path import BeamURL


def beam_server(obj, protocol='http', host=None, port=None, backend=None, **kwargs):

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

    server.run(host=host, port=port, **run_kwargs)


def beam_client(uri, hostname=None, port=None, username=None, api_key=None, **kwargs):

    scheme = uri.split(':')[0]
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
        kwargs[k] = v

    if api_key is None and 'api_key' in kwargs:
        api_key = kwargs.pop('api_key')

    if 'http' in scheme:
        if 'tls' not in kwargs:
            kwargs['tls'] = True if scheme == 'https' else False
        from .http_client import HTTPClient
        return HTTPClient(hostname=hostname, port=port, username=username, api_key=api_key, **kwargs)
    elif scheme == 'grpc':
        from .grpc_client import GRPCClient
        return GRPCClient(hostname=hostname, port=port, username=username, api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unknown protocol: {scheme}")

