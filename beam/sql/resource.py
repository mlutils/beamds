from .core import BeamIbis
from ..path import BeamURL


def beam_ibis(path, username=None, hostname=None, port=None, private_key=None, access_key=None, secret_key=None,
              password=None, scheme=None, **kwargs):

    url = BeamURL.from_string(path)

    if url.hostname is not None:
        hostname = url.hostname

    if url.port is not None:
        port = url.port

    if url.username is not None:
        username = url.username

    if url.password is not None:
        password = url.password

    query = url.query
    for k, v in query.items():
        kwargs[k.replace('-', '_')] = v

    if access_key is None and 'access_key' in kwargs:
        access_key = kwargs.pop('access_key')

    path = url.path
    if path == '':
        path = '/'

    fragment = url.fragment

    backend = None
    if scheme is not None and '_' in scheme:
        backend = scheme.split('_')[1]

    return BeamIbis(path, hostname=hostname, backend=backend, port=port, username=username, password=password,
                       fragment=fragment, access_key=access_key, **kwargs)

