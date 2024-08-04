from typing import Union, Any

from .base import BeamResource, resource_names


def resource(uri, **kwargs) -> Union[BeamResource, Any]:
    if type(uri) != str:
        return uri
    if ':' not in uri:
        from .path import beam_path
        return beam_path(uri, **kwargs)
    elif uri[1] == ':':  # windows path
        from .path import beam_path
        return beam_path(uri, **kwargs)

    scheme = uri.split(':')[0]
    if scheme in resource_names['path']:
        from .path import beam_path
        return beam_path(uri, **kwargs)
    elif scheme in resource_names['serve']:
        from .serve import beam_client
        return beam_client(uri, **kwargs)
    elif scheme in resource_names['distributed']:
        from .distributed import async_client
        return async_client(uri, **kwargs)
    elif scheme in resource_names['llm']:
        from .llm import beam_llm
        return beam_llm(uri, **kwargs)
    elif scheme in resource_names['triton']:
        from .serve import triton_client
        return triton_client(uri, **kwargs)
    elif scheme in resource_names['ray']:
        from .distributed import ray_client
        return ray_client(uri, **kwargs)
    elif scheme in resource_names['embedding']:
        from .embedding import beam_embedding
        return beam_embedding(uri, **kwargs)
    else:
        raise Exception(f'Unknown resource scheme: {scheme}')


def this_file():
    import inspect
    # Get the current call stack
    stack = inspect.stack()

    # Iterate over the stack frames to find the first one outside the current module
    for frame in stack:
        caller_file_path = frame.filename
        if not caller_file_path.endswith('/resources.py'):
            return resource(caller_file_path).resolve()

    # If no such frame is found (very unlikely), return the first frame
    return resource(stack[0].filename).resolve()


def this_dir():
    return this_file().parent
