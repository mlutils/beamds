import os, torch, copy, sys
from collections import defaultdict
import numpy as np
import torch.distributed as dist
from fnmatch import fnmatch, filter
from tqdm import *
import random
import torch
import pandas as pd
import multiprocessing as mp
import socket
from contextlib import closing
from random import randint
from collections import namedtuple

from loguru import logger

# logger.remove(handler_id=0)
logger.remove()
logger.add(sys.stdout, colorize=True,
           format='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>')

def is_boolean(x):

    x_type = check_type(x)
    if x_type.minor in ['numpy', 'pandas', 'tensor'] and 'bool' in str(x.dtype).lower():
        return True
    if x_type.minor == 'list' and type(x[0]) is bool:
        return True

    return False


def as_tensor(x, device=None):
    if type(x) is not torch.Tensor:
        x = torch.tensor(x)

    if device is not None:
        x = x.to(device)

    return x


def slice_to_array(s, l=None, arr_type='tensor'):
    f = torch.arange if arr_type == 'tensor' else np.arange
    if type(s) is slice:

        step = s.step
        if step is None:
            step = 1

        start = s.start
        if start is None:
            start = 0

        stop = s.stop
        if stop is None:
            stop = l

        return f(start, stop, step)
    return s


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        p = str(s.getsockname()[1])
    return p


def check_if_port_is_available(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    return sock.connect_ex(('127.0.0.1', int(port))) == 0


def get_notebook_name():
    """Execute JS code to save Jupyter notebook name to variable `notebook_name`"""
    from IPython.core.display import Javascript, display_javascript
    js = Javascript("""IPython.notebook.kernel.execute('notebook_name = "' + IPython.notebook.notebook_name + '"');""")

    return display_javascript(js)


def process_async(func, args, mp_context='spawn', num_workers=10):
    ctx = mp.get_context(mp_context)
    with ctx.Pool(num_workers) as pool:
        res = [pool.apply_async(func, (args,)) for arg in args]
        results = []
        for r in tqdm_beam(res):
            results.append(r.get())

    return results


def check_element_type(x):
    t = str(type(x)).lower()

    if pd.isna(x):
        return 'none'
    if 'int' in t:
        return 'int'
    if 'float' in t:
        return 'float'
    if 'str' in t:
        return 'str'
    return 'object'


def check_minor_type(x):
    t = type(x)

    if issubclass(t, dict):
        return 'dict'
    if isinstance(x, torch.Tensor):
        return 'tensor'
    if issubclass(t, list):
        return 'list'
    if isinstance(x, np.ndarray):
        return 'numpy'
    if isinstance(x, pd.core.base.PandasObject):
        return 'pandas'
    if issubclass(t, tuple):
        return 'tuple'
    else:
        return 'other'


def check_type(x):
    '''

    returns:

    <major type>, <minor type>, <elements type>

    major type: array, scalar, dict, none, other
    minor type: tensor, numpy, pandas, native, list, tuple, none
    elements type: int, float, str, object, none, unknown

    '''

    type_tuple = namedtuple('Type', 'major minor element')

    t = type(x)

    mit = check_minor_type(x)
    if np.isscalar(x) or (torch.is_tensor(x) and (not len(x.shape))):
        mjt = 'scalar'
        if type(x) in [int, float, str]:
            mit = 'native'
        elt = check_element_type(x)

    elif issubclass(t, dict):
        mjt = 'dict'
        mit = 'dict'
        elt = check_element_type(next(iter(x.values())))

    elif x is None:
        mjt = 'none'
        mit = 'none'
        elt = 'none'

    elif mit != 'other':
        mjt = 'array'
        if mit in ['list', 'tuple']:
            elt = check_element_type(x[0])
        elif mit in ['numpy', 'tensor', 'pandas']:
            if mit == 'pandas':
                dt = str(x.values.dtype)
            else:
                dt = str(x.dtype)
            if 'float' in dt:
                elt = 'float'
            elif 'int' in dt:
                elt = 'int'
            else:
                elt = 'object'
        else:
            elt = 'unknown'

    else:
        mjt = 'other'
        mit = 'other'
        elt = 'other'

    return type_tuple(major=mjt, minor=mit, element=elt)


def include_patterns(*patterns):
    """Factory function that can be used with copytree() ignore parameter.
    Arguments define a sequence of glob-style patterns
    that are used to specify what files to NOT ignore.
    Creates and returns a function that determines this for each directory
    in the file hierarchy rooted at the source directory when used with
    shutil.copytree().
    """

    def _ignore_patterns(path, names):
        keep = set(name for pattern in patterns
                   for name in filter(names, pattern))
        ignore = set(name for name in names
                     if name not in keep and not os.path.isdir(os.path.join(path, name)))
        return ignore

    return _ignore_patterns


def is_notebook():
    return '_' in os.environ and 'jupyter' in os.environ['_']


def setup(rank, world_size, port='7463'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup(rank, world_size):
    dist.destroy_process_group()


def set_seed(seed=-1, constant=0, increment=False, deterministic=False):
    '''
    :param seed: set -1 to avoid change, set 0 to randomly select seed, set [1, 2**32) to get new seed
    :param constant: a constant to be added to the seed
    :param increment: whether to generate incremental seeds
    :param deterministic: whether to set torch to be deterministic
    :return: None
    '''

    if 'cnt' not in set_seed.__dict__:
        set_seed.cnt = 0
    set_seed.cnt += 1

    if increment:
        constant += set_seed.cnt

    if seed == 0:
        seed = np.random.randint(1, 2 ** 32 - constant) + constant

    if seed > 0:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def to_device(data, device='cuda', half=False):
    if type(data) is dict:
        return {k: to_device(v, device=device, half=half) for k, v in data.items()}
    elif (type(data) is list) or (type(data) is tuple):
        return [to_device(s, device=device, half=half) for s in data]
    elif issubclass(type(data), torch.Tensor):
        if half and data.dtype in [torch.float32, torch.float64]:
            data = data.half()
        return data.to(device)
    else:
        return data


def finite_iterations(iterator, n):
    for i, out in enumerate(iterator):

        if i + 1 < n:
            yield out
        else:
            return out


def tqdm_beam(x, *args, enable=True, notebook=True, **argv):
    if not enable:
        return x
    else:
        my_tqdm = tqdm_notebook if (is_notebook() and notebook) else tqdm
        return my_tqdm(x, *args, **argv)

