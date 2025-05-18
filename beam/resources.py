import os
from typing import Union, Any
import sys

from .base import BeamResource, resource_names

dynamic_resources = {}


def register_resource(scheme, generator):
    dynamic_resources[scheme] = generator


def resource(uri, **kwargs) -> Union[BeamResource, Any]:
    if type(uri) != str:
        return uri

    if '://' not in uri:
        from .path import beam_path
        return beam_path(uri, **kwargs)

    scheme = uri.split('://')[0]
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
    elif scheme in resource_names['elastic']:
        from .docs import beam_elastic
        return beam_elastic(uri, **kwargs)
    elif scheme in resource_names['airflow']:
        from .flow import airflow_client
        return airflow_client(uri, **kwargs)
    elif scheme in resource_names['ibis']:
        from .sql import beam_ibis
        return beam_ibis(uri, **kwargs)
    elif scheme in dynamic_resources:
        return dynamic_resources[scheme](uri, **kwargs)
    else:
        raise Exception(f'Unknown resource scheme: {scheme}')


def this_file():

    from .utils import is_notebook
    import inspect
    # Get the current call stack
    stack = inspect.stack()

    p = None
    # Iterate over the stack frames to find the first one outside the current module
    for frame in stack:
        caller_file_path = frame.filename
        if not caller_file_path.endswith('/resources.py'):
            p = resource(caller_file_path).resolve()
            break
    if p is None:
        # If no such frame is found (very unlikely), return the first frame
        p = resource(stack[0].filename).resolve()

    if 'ipykernel' in p.str and is_notebook():

        try:
            r = resource(sys.argv[2]).read()
            return resource(r['jupyter_session'])
        except Exception as e:
            from .logging import beam_logger as logger
            logger.error(f"Notebook detected, could not get the current file path from ths sys.argv configuration: {e}")
            return None

    return p


def this_dir():
    p = this_file()
    if p is None:
        from .logging import beam_logger as logger
        logger.warning('Could not get the current file (probably jupyter settings), '
                       'returning the current working directory instead')
        return cwd()
    return p.parent


def cwd():
    return resource('.').resolve()


def chdir(path):
    p = str(resource(path).resolve())
    os.chdir(p)
