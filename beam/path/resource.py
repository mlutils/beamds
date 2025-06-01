from typing import Any

from .models import (BeamPath, S3Path, S3PAPath, HDFSPath, HDFSPAPath, SFTPPath, CometAsset,
                     RedisPath, SMBPath, MLFlowPath, GoogleStoragePath)
from .core import BeamKey, BeamURL, IOPath, DictPath
from pathlib import PurePath


beam_key = BeamKey()


def beam_path(path, username=None, hostname=None, port=None, private_key=None, access_key=None, secret_key=None,
              password=None, project_id=None, **kwargs) -> BeamPath | Any:
    """

    @param port:
    @param hostname:
    @param username:
    @param protocol:
    @param private_key:
    @param secret_key: AWS secret key
    @param access_key: AWS access key
    @param path: URI syntax: [protocol://][username@][hostname][:port][/path/to/file]
    @return: BeamPath object
    """

    if isinstance(path, BeamPath):
        return path

    if isinstance(path, BeamURL) or isinstance(path, PurePath):
        path = str(path)

    if type(path) != str:
        return path

    if len(path) > 1 and path[1] == ':':  # windows path
        path = path.replace('\\', '/')
        path = path.lstrip('/')
        return BeamPath(path, scheme='nt')
    elif '://' not in path:
        return BeamPath(path, scheme='file')

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

    if private_key is None and 'private_key' in kwargs:
        private_key = kwargs.pop('private_key')

    if secret_key is None and 'secret_key' in kwargs:
        secret_key = kwargs.pop('secret_key')

    if project_id is None and 'project_id' in kwargs:
        project_id = kwargs.pop('project_id')

    path = url.path

    if url.protocol is None or (url.protocol == 'file'):
        return BeamPath(path)

    if path == '':
        path = '/'

    username = beam_key('BEAM_USERNAME', username)
    password = beam_key('BEAM_PASSWORD', password)

    if 's3' in url.protocol:

        access_key = beam_key('AWS_ACCESS_KEY_ID', access_key)
        secret_key = beam_key('AWS_SECRET_ACCESS_KEY', secret_key)

        if url.protocol == 's3-pa':
            return S3PAPath(path, hostname=hostname, port=port, access_key=access_key, secret_key=secret_key, **kwargs)
        else:
            return S3Path(path, hostname=hostname, port=port, access_key=access_key, secret_key=secret_key,  **kwargs)

    elif url.protocol == 'hdfs':
        return HDFSPath(path, hostname=hostname, port=port, username=username, password=password, **kwargs)

    elif url.protocol == 'hdfs-pa':
        return HDFSPAPath(path, hostname=hostname, port=port, username=username, password=password, **kwargs)

    elif url.protocol == 'redis':
        return RedisPath(path, hostname=hostname, port=port, username=username, password=password, **kwargs)

    elif url.protocol == 'smb':
        return SMBPath(path, hostname=hostname, port=port, username=username, password=password, **kwargs)

    elif url.protocol == 'comet':

        access_key = beam_key('COMET_API_KEY', access_key)
        return CometAsset(path, access_key=access_key, **kwargs)

    elif url.protocol == 'mlflow':
        return MLFlowPath(path, hostname=hostname, port=port, username=username, password=password,
                          **kwargs)

    elif url.protocol == 'io':
        return IOPath(path, **kwargs)

    elif url.protocol == 'dict':
        return DictPath(path, **kwargs)

    elif url.protocol in ['gs', 'gcs']:
        if url.protocol == 'gs':
            path = f'/{hostname}/{path.lstrip("/")}'  # Google Storage requires a leading slash
            hostname = None
        return GoogleStoragePath(path, hostname=hostname, port=port, 
                                 access_key=access_key, project_id=project_id, **kwargs)
    elif url.protocol == 'http':
        raise NotImplementedError
    elif url.protocol == 'https':
        raise NotImplementedError
    elif url.protocol == 'ftp':
        raise NotImplementedError
    elif url.protocol == 'ftps':
        raise NotImplementedError
    elif url.protocol == 'nt':
        path = path.replace('\\', '/')
        return BeamPath(path)
    elif url.protocol == 'sftp':

        private_key = beam_key('SSH_PRIVATE_KEY', private_key)
        return SFTPPath(path, hostname=hostname, username=username, port=port,
                        private_key=private_key, password=password, **kwargs)
    else:
        raise Exception(f'Unknown protocol: {url.protocol}')


def in_memory_storage(mode=None, data=None):
    if mode == 'file':
        return beam_path('io:///', data=data)
    return beam_path('dict:///', data=data)
