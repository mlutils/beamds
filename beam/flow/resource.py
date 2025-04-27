from ..path import BeamURL, beam_key
from .client import AirflowClient


def airflow_client(path, username=None, hostname=None, port=None, password=None, **kwargs):

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

    path = url.path
    if path == '':
        path = '/'

    fragment = url.fragment

    username = beam_key('AIRFLOW_USERNAME', username)
    password = beam_key('AIRFLOW_PASSWORD', password)

    return AirflowClient(path, hostname=hostname, port=port, username=username, password=password,
                       fragment=fragment, **kwargs)