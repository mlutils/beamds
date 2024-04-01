from .models import (BeamPath, S3Path, S3PAPath, HDFSPath, HDFSPAPath, SFTPPath, CometAsset,
                     RedisPath, SMBPath)
from .core import BeamKey, BeamURL, IOPath, DictPath


beam_key = BeamKey()


def beam_path(path, username=None, hostname=None, port=None, private_key=None, access_key=None, secret_key=None,
              password=None, **kwargs):
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
    if type(path) != str:
        return path

    if ':' not in path:
        return BeamPath(path, scheme='file')
    elif path[1] == ':':  # windows path
        path = path.replace('\\', '/')
        path = path.lstrip('/')
        return BeamPath(path, scheme='nt')

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
        kwargs[k] = v

    if access_key is None and 'access_key' in kwargs:
        access_key = kwargs.pop('access_key')

    if private_key is None and 'private_key' in kwargs:
        private_key = kwargs.pop('private_key')

    if secret_key is None and 'secret_key' in kwargs:
        secret_key = kwargs.pop('secret_key')

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

    elif url.protocol == 'io':
        return IOPath(path, **kwargs)

    elif url.protocol == 'dict':
        return DictPath(path, **kwargs)

    elif url.protocol == 'gs':
        raise NotImplementedError
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
        raise NotImplementedError


def in_memory_storage(mode=None, data=None):
    if mode == 'file':
        return beam_path('io:///', data=data)
    return beam_path('dict:///', data=data)
