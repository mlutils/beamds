

def resource(uri, **kwargs):
    if type(uri) != str:
        return uri
    if ':' not in uri:
        from .path import beam_path
        return beam_path(uri, **kwargs)

    scheme = uri.split(':')[0]
    if scheme in ['file', 's3', 's3-pa', 'hdfs', 'hdfs-pa', 'sftp']:
        from .path import beam_path
        return beam_path(uri, **kwargs)
    elif scheme == 'beam-serve':
        from .serve.beam_client import BeamClient
        return BeamClient(uri, **kwargs)
    elif scheme in ['openai', 'tgi', 'fastchat', 'huggingface', 'fastapi']:
        from .llm import beam_llm
        return beam_llm(uri, **kwargs)
    else:
        raise Exception(f'Unknown resource scheme: {scheme}')
