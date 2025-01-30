from .elastic import BeamElastic
from ..path import BeamURL


def beam_elastic(path, username=None, hostname=None, port=None, private_key=None, access_key=None, secret_key=None,
                 password=None, **kwargs):

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

    # if access_key is None and 'access_key' in kwargs:
    #     access_key = kwargs.pop('access_key')
    #
    # if private_key is None and 'private_key' in kwargs:
    #     private_key = kwargs.pop('private_key')
    #
    # if secret_key is None and 'secret_key' in kwargs:
    #     secret_key = kwargs.pop('secret_key')

    path = url.path
    if path == '':
        path = '/'

    fragment = url.fragment

    return BeamElastic(path, hostname=hostname, port=port, username=username, password=password,
                       fragment=fragment, **kwargs)

