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
from timeit import default_timer as timer
from loguru import logger

# logger.remove(handler_id=0)
logger.remove()
logger.add(sys.stdout, level='INFO', colorize=True,
           format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>')


def rate_string_format(n, t):
    if n / t > 1:
        return f"{n / t: .4} [iter/sec]"
    return f"{t / n: .4} [sec/iter]"


def beam_logger():
    return logger


def is_boolean(x):

    x_type = check_type(x)
    if x_type.minor in ['numpy', 'pandas', 'tensor'] and 'bool' in str(x.dtype).lower():
        return True
    if x_type.minor == 'list' and type(x[0]) is bool:
        return True

    return False


def as_tensor(x, device=None, return_vector=False):
    if type(x) is not torch.Tensor:
        x = torch.tensor(x)

    if return_vector:
        if not len(x.shape):
            x = x.unsqueeze(0)

    if device is not None:
        x = x.to(device)

    return x


def slice_to_index(s, l=None, arr_type='tensor', sliced=None):

    if type(s) is slice:

        f = torch.arange if arr_type == 'tensor' else np.arange

        if s == slice(None):
            if sliced is not None:
                return sliced
            elif l is not None:
                return f(l)
            else:
                return ValueError(f"Cannot slice: {s} without length info")

        step = s.step
        if step is None:
            step = 1

        start = s.start
        if start is None:
            start = 0 if step > 0 else l-1
        elif start < 0:
            start = l + start

        stop = s.stop
        if stop is None:
            stop = l if step > 0 else -1
        elif stop < 0:
            stop = l + stop

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
    return sock.connect_ex(('127.0.0.1', int(port))) != 0


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


def beam_device(device):
    if type(device) is torch.device:
        return device
    device = str(device)
    return torch.device(int(device) if device.isnumeric() else device)


def check_element_type(x):
    t = str(type(x)).lower()

    if not np.isscalar(x) and (not (torch.is_tensor(x) and (not len(x.shape)))):
        return 'array'
    elif pd.isna(x):
        return 'none'
    if 'int' in t:
        return 'int'
    if 'bool' in t:
        return 'bool'
    if 'float' in t:
        return 'float'
    if 'str' in t:
        return 'str'
    return 'object'


def check_minor_type(x):
    t = type(x)

    if isinstance(x, torch.Tensor):
        return 'tensor'
    if isinstance(x, np.ndarray):
        return 'numpy'
    if isinstance(x, pd.core.base.PandasObject):
        return 'pandas'
    if issubclass(t, dict):
        return 'dict'
    if issubclass(t, list):
        return 'list'
    if issubclass(t, tuple):
        return 'tuple'
    else:
        return 'other'


type_tuple = namedtuple('Type', 'major minor element')


def check_type(x, check_minor=True, check_element=True):
    '''

    returns:

    <major type>, <minor type>, <elements type>

    major type: array, scalar, dict, none, other
    minor type: tensor, numpy, pandas, native, list, tuple, none
    elements type: int, float, str, object, none, unknown

    '''

    t = type(x)

    if np.isscalar(x) or (torch.is_tensor(x) and (not len(x.shape))):
        mjt = 'scalar'
        if type(x) in [int, float, str]:
            mit = 'native'
        else:
            mit = check_minor_type(x) if check_minor else 'na'
        elt = check_element_type(x) if check_element else 'na'

    elif issubclass(t, dict):
        mjt = 'dict'
        mit = 'dict'
        elt = check_element_type(next(iter(x.values()))) if check_element else 'na'

    elif x is None:
        mjt = 'none'
        mit = 'none'
        elt = 'none'

    else:

        mit = check_minor_type(x) if check_minor else 'na'
        if mit != 'other':
            mjt = 'array'
            if mit in ['list', 'tuple']:
                elt = check_element_type(x[0]) if check_element else 'na'
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
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True


def to_device(data, device='cuda', half=False):

    if issubclass(type(data), dict):
        return {k: to_device(v, device=device, half=half) for k, v in data.items()}
    elif (type(data) is list) or (type(data) is tuple):
        return [to_device(s, device=device, half=half) for s in data]
    elif issubclass(type(data), torch.Tensor):
        if half and data.dtype in [torch.float32, torch.float64]:
            data = data.half()
        return data.to(device)
    else:
        return data


def concat_data(data):

    d0 = data[0]
    if issubclass(type(d0), dict):
        return {k: concat_data([di[k] for di in data]) for k in d0.keys()}
    elif (type(d0) is list) or (type(d0) is tuple):
        return [concat_data([di[n] for di in data]) for n in range(len(d0))]
    elif issubclass(type(d0), torch.Tensor):
        return torch.cat(data)
    else:
        return data


def finite_iterations(iterator, n):
    for i, out in enumerate(iterator):
        yield out
        if i + 1 == n:
            break


def tqdm_beam(x, *args, threshold=10, stats_period=1, message_func=None, enable=None, notebook=True, **argv):

    my_tqdm = tqdm_notebook if (is_notebook() and notebook) else tqdm

    if enable is False:
        return x

    elif enable is True:

        pb = my_tqdm(x, *args, **argv)
        for xi in pb:
            if message_func is not None:
                pb.set_description(message_func(xi))
            yield xi

    else:

        iter_x = iter(x)

        if 'total' in argv:
            l = argv['total']
            argv.pop('total')
        else:
            try:
                l = len(x)
            except TypeError:
                l = None

        t0 = timer()

        stats_period = stats_period if l is not None else threshold
        n = 0
        while (te := timer()) - t0 <= stats_period:
            n += 1
            try:
                yield next(iter_x)
            except StopIteration:
                return

        long_iter = None
        if l is not None:
            long_iter = (te - t0) / n * l > threshold

        if l is None or long_iter:
            pb = my_tqdm(iter_x, *args, initial=n, total=l, **argv)
            for xi in pb:
                if message_func is not None:
                    pb.set_description(message_func(xi))
                yield xi
        else:
            for xi in iter_x:
                yield xi

