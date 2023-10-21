

def resource(uri, **kwargs):
    if type(uri) != str:
        return uri
    if ':' not in uri:
        from .path import BeamPath
        return BeamPath(uri, **kwargs)

    scheme = uri.split(':')[0]
    if scheme == 'file':
        from .path import BeamPath
        return BeamPath(uri, **kwargs)
    elif scheme == 's3':
        from .path import S3Path
        return S3Path(uri, **kwargs)
    elif scheme == 'gs':
        raise NotImplementedError
    elif scheme == 'hdfs':
        from .path import HDFSPath
        return HDFSPath(uri, **kwargs)
    elif scheme == 'beam-server':
        from .server import BeamClient
        return BeamClient(uri, **kwargs)
    elif scheme == 'hdfs-pa':
        pass
