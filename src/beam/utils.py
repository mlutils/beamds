import os, torch, copy, sys
from collections import defaultdict
import numpy as np
import torch.distributed as dist
from fnmatch import fnmatch, filter
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm
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
from torchvision import transforms
import hashlib
from functools import partial

# logger.remove(handler_id=0)
logger.remove()
logger.add(sys.stdout, level='INFO', colorize=True,
           format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>')


def stack_train_results(results, batch_size=None):

    stacked_results = defaultdict(dict)

    for k_type in results.keys():
        for k_name, v in results[k_type].items():
            stacked_results[k_type][k_name] = v

    stacked_results = stack_inference_results(stacked_results, batch_size=batch_size)

    return stacked_results


def stack_inference_results(results, batch_size=None):
    for n, res in results.items():
        for k, v in res.items():
            v_type = check_type(v)
            if v_type.major == 'array' and v_type.minor == 'list' and v_type.element != 'array':
                vi_type = check_type(v[0])
                if vi_type.minor == 'numpy':
                    results[n][k] = np.stack(results[n][k])
                elif vi_type.minor == 'tensor':
                    results[n][k] = torch.stack(results[n][k])
            elif v_type.major == 'array' and v_type.element == 'array':
                if v_type.minor in ['tensor', 'numpy', 'list']:

                    if v_type.minor == 'tensor':
                        oprs = {'cat': torch.cat, 'stack': torch.stack}
                    elif v_type.minor == 'numpy':
                        oprs = {'cat': np.concatenate, 'stack': np.stack}
                    else:
                        vi_type = check_type(v[0])
                        if vi_type.minor == 'tensor':
                            oprs = {'cat': torch.cat, 'stack': torch.stack}
                        elif vi_type.minor == 'numpy':
                            oprs = {'cat': np.concatenate, 'stack': np.stack}
                        else:
                            break

                    opr = oprs['cat']
                    if batch_size is not None and v[0].shape != batch_size:
                        opr = oprs['stack']

                    results[n][k] = opr(results[n][k])

    return results


def rate_string_format(n, t):
    if n / t > 1:
        return f"{n / t: .4} [iter/sec]"
    return f"{t / n: .4} [sec/iter]"


def beam_logger():
    return logger


def print_beam_hyperparameters(args, debug_only=False):

    if debug_only:
        log_func = logger.debug
    else:
        log_func = logger.info

    log_func(f"beam project: {args.project_name}")
    log_func('Experiment Hyperparameters')
    log_func('----------------------------------------------------------'
             '---------------------------------------------------------------------')

    hparams_list = args.hparams
    var_args_sorted = dict(sorted(vars(args).items()))

    for k, v in var_args_sorted.items():
        if k == 'hparams':
            continue
        elif k in hparams_list:
            log_func(k + ': ' + str(v))
        else:
            logger.debug(k + ': ' + str(v))

    log_func('----------------------------------------------------------'
             '---------------------------------------------------------------------')

def find_port(port=None, get_port_from_beam_port_range=True, application='tensorboard'):

    if application == 'tensorboard':
        first_beam_range = 66
        first_global_range = 26006
    elif application == 'flask':
        first_beam_range = 50
        first_global_range = 25000
    else:
        raise NotImplementedError

    if port is None:

        port_range = None

        if get_port_from_beam_port_range:

            base_range = None
            if 'JUPYTER_PORT' in os.environ:

                base_range = int(os.environ['JUPYTER_PORT']) // 100

            elif os.path.isfile('/workspace/configuration/config.csv'):
                conf = pd.read_csv('/workspace/configuration/config.csv')
                base_range = int(conf.set_index('parameters').loc['initials'])

            if base_range is not None:

                port_range = range(base_range * 100, (base_range + 1) * 100)
                port_range = np.roll(np.array(port_range), -first_beam_range)

        if port_range is None:
            port_range = np.roll(np.array(range(10000, 2 ** 16)), -first_global_range)

        for p in port_range:
            if check_if_port_is_available(p):
                port = str(p)
                break

        if port is None:
            logger.error("Cannot find free port in the specified range")
            return

    else:
        if not check_if_port_is_available(port):
            logger.error(f"Port {port} is not available")
            return

    return port


def is_boolean(x):

    x_type = check_type(x)
    if x_type.minor in ['numpy', 'pandas', 'tensor'] and 'bool' in str(x.dtype).lower():
        return True
    if x_type.minor == 'list' and len(x) and isinstance(x[0], bool):
        return True

    return False


def slice_to_index(s, l=None, arr_type='tensor', sliced=None):

    if isinstance(s, slice):

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
        res = [pool.apply_async(func, (arg,)) for arg in args]
        results = []
        for r in tqdm_beam(res):
            results.append(r.get())

    return results


def pretty_format_number(x):

    if x is None or np.isinf(x) or np.isnan(x):
        return f'{x}'.ljust(10)
    if int(x) == x and np.abs(x) < 10000:
        return f'{int(x)}'.ljust(10)
    if np.abs(x) >= 10000 or np.abs(x) < 0.0001:
        return f'{float(x):.4}'.ljust(10)
    if np.abs(x) >= 1000:
        return f'{x:.1f}'.ljust(10)
    if np.abs(x) < 10000 and np.abs(x) >= 0.0001:
        nl = int(np.log10(np.abs(x)))
        return f'{np.sign(x) * int(np.abs(x) * (10 ** (4 - nl))) * float(10 ** (nl - 4))}'.ljust(8)[:8].ljust(10)

    return f'{x}:NoFormat'


def beam_device(device):
    if isinstance(device, torch.device) or device is None:
        return device
    device = str(device)
    return torch.device(int(device) if device.isnumeric() else device)


def check_element_type(x):

    if not np.isscalar(x) and (not (torch.is_tensor(x) and (not len(x.shape)))):
        return 'array'
    if pd.isna(x):
        return 'none'

    if hasattr(x, 'dtype'):
        t = str(x.dtype).lower()
    else:
        t = str(type(x)).lower()

    if 'int' in t:
        return 'int'
    if 'bool' in t:
        return 'bool'
    if 'float' in t:
        if '16' in t:
            return 'float16'
        else:
            return 'float'
    if 'str' in t:
        return 'str'

    return 'object'


def check_minor_type(x):

    if isinstance(x, torch.Tensor):
        return 'tensor'
    if isinstance(x, np.ndarray):
        return 'numpy'
    if isinstance(x, pd.core.base.PandasObject):
        return 'pandas'
    if isinstance(x, dict):
        return 'dict'
    if isinstance(x, list):
        return 'list'
    if isinstance(x, tuple):
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
    elements type: int, float, str, object, empty, none, unknown

    '''

    if np.isscalar(x) or (torch.is_tensor(x) and (not len(x.shape))):
        mjt = 'scalar'
        if type(x) in [int, float, str]:
            mit = 'native'
        else:
            mit = check_minor_type(x) if check_minor else 'na'
        elt = check_element_type(x) if check_element else 'na'

    elif isinstance(x, dict):
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
                if check_element:
                    if len(x):
                        elt = check_element_type(x[0])
                    else:
                        elt = 'empty'
                else:
                    elt = 'na'
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

    if isinstance(data, dict):
        return {k: to_device(v, device=device, half=half) for k, v in data.items()}
    elif isinstance(data, list) or isinstance(data, tuple):
        return [to_device(s, device=device, half=half) for s in data]
    elif isinstance(data, torch.Tensor):
        if half and data.dtype in [torch.float32, torch.float64]:
            data = data.half()
        return data.to(device)
    else:
        return data


def recursive_batch(x, index):

    if isinstance(x, dict):
        return {k: recursive_batch(v, index) for k, v in x.items()}
    elif isinstance(x, list) or isinstance(x, tuple):
        return [recursive_batch(s, index) for s in x]
    elif x is None:
        return None
    else:
        return x[index]


def recursive_device(x):

    if isinstance(x, dict):
        for xi in x.values():
            try:
                return recursive_device(xi)
            except AttributeError:
                # case of None
                pass
    elif isinstance(x, list) or isinstance(x, tuple):
        for xi in x:
            try:
                return recursive_device(xi)
            except AttributeError:
                # case of None
                pass
    return x.device


def recursive_len(x):

    if isinstance(x, dict):
        for xi in x.values():
            try:
                return recursive_len(xi)
            except TypeError:
                # case of None
                pass

    elif isinstance(x, list) or isinstance(x, tuple):
        for xi in x:
            try:
                return recursive_len(xi)
            except TypeError:
                # case of None
                pass

    return len(x)


def as_numpy(x):

    if isinstance(x, dict):
        return {k: as_numpy(v) for k, v in x.items()}
    elif isinstance(x, list) or isinstance(x, tuple):
        return [as_numpy(s) for s in x]

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.array(x)

    if x.size == 1:
        if 'int' in str(x.dtype):
            x = int(x)
        else:
            x = float(x)

    return x


def as_tensor(x, device=None, dtype=None, return_vector=False):

    if isinstance(x, dict):
        return {k: as_tensor(v, device=device, return_vector=return_vector) for k, v in x.items()}
    elif isinstance(x, list) or isinstance(x, tuple):
        return [as_tensor(s, device=device, return_vector=return_vector) for s in x]
    elif x is None:
        return None

    x = torch.as_tensor(x, device=device, dtype=dtype)
    if return_vector:
        if not len(x.shape):
            x = x.unsqueeze(0)

    return x


def concat_data(data):

    d0 = data[0]
    if isinstance(d0, dict):
        return {k: concat_data([di[k] for di in data]) for k in d0.keys()}
    elif isinstance(d0, list) or isinstance(d0, tuple):
        return [concat_data([di[n] for di in data]) for n in range(len(d0))]
    elif isinstance(d0, torch.Tensor):
        return torch.cat(data)
    else:
        return data


def batch_augmentation_(x, augmentations):
    return torch.stack([augmentations(xi) for xi in x])


def batch_augmentation(augmentations):

    ba = partial(batch_augmentation_, augmentations=augmentations)
    return transforms.Lambda(ba)


def finite_iterations(iterator, n):
    for i, out in enumerate(iterator):
        yield out
        if i + 1 == n:
            break


def hash_tensor(x, fast=False, coarse=False):
    """
    This  function returns a deterministic hash of the tensor content
    @param x: the tensor to hash
    @param fast: whether to consider only the first and last elements of the tensor for hashing
    @param coarse: whether to apply coarse hashing where the tensor is quantized into low resolution (16bit) tensor
    @return: an integer representing the hash value
    """
    if torch.numel(x) < 10000:
        fast = False

    if coarse and 'float' in str(x.dtype):
        x = (x / x.max() * (2 ** 15)).half()

    x = as_numpy(x)

    if fast:
        x = str(x).encode('utf-8')
    else:
        x.flags.writeable = False
        x = x.data

    return int(hashlib.sha1(x).hexdigest(), 16)


def tqdm_beam(x, *args, threshold=10, stats_period=1, message_func=None, enable=None, notebook=True, **argv):

    """
    Beam's wrapper for the tqdm progress bar. It features a universal interface for both jupyter notebooks and .py files.
    In addition, it provides a "lazy progress bar initialization". The progress bar is initialized only if its estimated
    duration is longer than a threshold.

    Parameters
    ----------
        x:
        threshold : float
            The smallest expected duration (in Seconds) to generate a progress bar. This feature is used only if enable
            is set to None.
        stats_period: float
            The initial time period (in seconds) to calculate the ineration statistics (iters/sec). This statistics is used to estimate the expected duction of the entire iteration.
        message_func: func
            A dynamic message to add to the progress bar. For example, this message can plot the instantaneous loss.
        enable: boolean/None
            Whether to enable the progress bar, disable it or when set to None, use lazy progress bar.
        notebook: boolean
            A boolean that overrides the internal calculation of is_notebook. Set to False when you want to avoid printing notebook styled tqdm bars (for example, due to multiprocessing).
    """

    my_tqdm = tqdm_notebook if (is_notebook() and notebook) else tqdm

    if enable is False:
        for xi in x:
            yield xi

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

