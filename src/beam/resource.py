
def resource(uri, **kwargs):
    if type(uri) != str:
        return uri
    if ':' not in uri:
        from .path import beam_path
        return beam_path(uri, **kwargs)

    scheme = uri.split(':')[0]
    if scheme in ['file', 's3', 's3-pa', 'hdfs', 'hdfs-pa', 'sftp', 'comet', 'io', 'dict', 'redis', 'smb', 'nt']:
        from .path import beam_path
        return beam_path(uri, **kwargs)
    elif scheme in ['beam-http', 'beam-https', 'beam-grpc']:
        from .serve import beam_client
        return beam_client(uri, **kwargs)
    elif scheme in ['async-http', 'async-https']:
        from .distributed import async_client
        return async_client(uri, **kwargs)
    elif scheme in ['openai', 'tgi', 'fastchat', 'huggingface', 'fastapi', 'fastapi-dp']:
        from .llm import beam_llm
        return beam_llm(uri, **kwargs)
    elif scheme in ['triton', 'triton-http', 'triton-grpc', 'triton-https', 'triton-grpcs']:
        from .serve import triton_client
        return triton_client(uri, **kwargs)
    else:
        raise Exception(f'Unknown resource scheme: {scheme}')
