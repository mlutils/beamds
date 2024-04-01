import itertools
import os, sys
from collections import defaultdict
import random
import numpy as np
from fnmatch import filter
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm

import pandas as pd
import __main__
from datetime import timedelta
import time
import re
import threading

import traceback
import linecache
from concurrent.futures import ThreadPoolExecutor, TimeoutError


try:
    import modin.pandas as mpd

    has_modin = True
except ImportError:
    mpd = None
    has_modin = False

try:
    import torch

    has_torch = True
except ImportError:
    has_torch = False

try:
    import scipy

    has_scipy = True
except ImportError:
    has_scipy = False

try:
    import polars as pl
    has_polars = True
except ImportError:
    has_polars = False

# try:
#     import cudf
#     has_cudf = True
# except ImportError:
#     has_cudf = False

import socket
from contextlib import closing
from collections import namedtuple
from timeit import default_timer as timer

from pathlib import Path, PurePath, PurePosixPath
from collections import Counter
import inspect
from argparse import Namespace
from functools import wraps, partial
from ..path import PureBeamPath


TypeTuple = namedtuple('TypeTuple', 'major minor element')
DataBatch = namedtuple("DataBatch", "index label data")


# class Beamdantic(BaseModel):
#     # To be used with pydantic classes and lazy_property
#     _lazy_cache: Any = PrivateAttr(default=None)


class BeamDict(dict, Namespace):
    def __init__(self, initial_data=None, **kwargs):
        if isinstance(initial_data, dict):
            self.__dict__.update(initial_data)
        elif isinstance(initial_data, BeamDict):
            self.__dict__.update(initial_data.__dict__)
        elif hasattr(initial_data, '__dict__'):  # This will check for Namespace or any other object with attributes
            self.__dict__.update(initial_data.__dict__)
        elif initial_data is not None:
            raise TypeError(
                "initial_data should be either a dictionary, an instance of DictNamespace, or a Namespace object")

            # Handle additional kwargs
        for key, value in kwargs.items():
            self.__dict__[key] = value

        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = BeamDict(value)
            if isinstance(value, list):
                self.__dict__[key] = [BeamDict(v) if isinstance(v, dict) else v for v in value]

    def __getattr__(self, key):
        try:
            return self.__dict__[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def pop(self, key, default=None):
        try:
            return self.__dict__.pop(key)
        except KeyError:
            return default

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __repr__(self):
        return repr(self.__dict__)

    def __contains__(self, key):
        return key in self.__dict__


def retrieve_name(var):
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]

# def retrieve_name(var):
#     name = None
#     # Start by checking the global scope of the caller.
#     for fi in reversed(inspect.stack()):
#         names = [var_name for var_name, var_val in fi.frame.f_globals.items() if var_val is var]
#         if names:
#             name = names[0]
#             break  # Exit on the first match in global scope.
#
#     # If not found in global scope, check the local scope from inner to outer.
#     if not name:
#         for fi in inspect.stack():
#             names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
#             if names:
#                 name = names[0]
#                 break  # Exit on the first match in local scope.
#
#     return name


def has_kwargs(func):
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in inspect.signature(func).parameters.values())


def strip_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def strip_suffix(text, suffix):
    if text.endswith(suffix):
        return text[:-len(suffix)]
    return text


class nested_defaultdict(defaultdict):

    @staticmethod
    def default_factory_list():
        return defaultdict(list)

    @staticmethod
    def default_factory_dict():
        return defaultdict(dict)

    def __init__(self, default_factory=None, **kwargs):
        if default_factory is list:
            default_factory = self.default_factory_list
        elif default_factory is dict:
            default_factory = self.default_factory_dict
        super().__init__(default_factory, **kwargs)


def lazy_property(fn):

    @property
    def _lazy_property(self):
        try:
            cache = getattr(self, '_lazy_cache')
            return cache[fn.__name__]
        except KeyError:
            v = fn(self)
            cache[fn.__name__] = v
            return v
        except AttributeError:
            v = fn(self)
            setattr(self, '_lazy_cache', {fn.__name__: v})
            return v

    @_lazy_property.setter
    def _lazy_property(self, value):
        try:
            cache = getattr(self, '_lazy_cache')
            cache[fn.__name__] = value
        except AttributeError:
            setattr(self, '_lazy_cache', {fn.__name__: value})

    return _lazy_property


def get_public_ip():
    import requests
    try:
        response = requests.get("https://api64.ipify.org?format=json")
        ip = response.json().get("ip")
        return ip
    except requests.RequestException:
        return None


def rate_string_format(n, t):
    if n / t > 1:
        return f"{n / t: .4} [iter/sec]"
    return f"{t / n: .4} [sec/iter]"


def beam_base_port():
    base_range = None
    if 'JUPYTER_PORT' in os.environ:
        base_range = 100 * (int(os.environ['JUPYTER_PORT']) // 100)
    elif os.path.isfile('/workspace/configuration/config.csv'):
        try:
            conf = pd.read_csv('/workspace/configuration/config.csv', index_col=0)
            base_range = 100 * int(conf.loc['initials'])
        except:
            pass
    return base_range


def beam_service_port(service):
    port = None
    try:
        conf = pd.read_csv('/workspace/configuration/config.csv', index_col=0)
        port = int(conf.drop_duplicates().loc[service.lower()])
    except:
        pass

    return port


def find_port(port=None, get_port_from_beam_port_range=True, application='none', blacklist=None, whitelist=None):
    from ..logger import beam_logger as logger

    if blacklist is None:
        blacklist = []

    if whitelist is None:
        whitelist = []

    blacklist = [int(p) for p in blacklist]
    whitelist = [int(p) for p in whitelist]

    if application == 'tensorboard':
        first_beam_range = 66
        first_global_range = 26006
    elif application == 'flask':
        first_beam_range = 50
        first_global_range = 25000
    elif application == 'ray':
        first_beam_range = 65
        first_global_range = 28265
    elif application == 'distributed':
        first_beam_range = 64
        first_global_range = 28264
    else:
        first_beam_range = 2
        first_global_range = 30000

    if port is None:

        port_range = None

        if get_port_from_beam_port_range:

            base_range = None
            if os.path.isfile('/workspace/configuration/config.csv'):
                conf = pd.read_csv('/workspace/configuration/config.csv')
                base_range = int(conf.set_index('parameters').drop_duplicates().loc['initials'])

            if base_range is not None:
                port_range = range(base_range * 100, (base_range + 1) * 100)
                port_range = np.roll(np.array(port_range), -first_beam_range)

        if port_range is None:
            port_range = np.roll(np.array(range(10000, 2 ** 16)), -first_global_range)

        for p in port_range:

            if p in blacklist:
                continue

            if whitelist and p not in whitelist:
                continue

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


def pretty_format_number(x, short=False):

    just = 4 if short else 10
    trim = 4 if short else 8
    exp = 2 if short else 4

    if x is None or np.isinf(x) or np.isnan(x):
        return f'{x}'.ljust(just)
    if int(x) == x and np.abs(x) < 10000:
        return f'{int(x)}'.ljust(just)
    if np.abs(x) >= 10000 or np.abs(x) < 0.0001:
        return f'{float(x):.4}'.ljust(just)
    if np.abs(x) >= 1000:
        return f'{x:.1f}'.ljust(just)
    if np.abs(x) < 10000 and np.abs(x) >= 0.0001:
        nl = int(np.log10(np.abs(x)))
        return f'{np.sign(x) * int(np.abs(x) * (10 ** (exp - nl))) * float(10 ** (nl - exp))}'.ljust(trim)[:trim].ljust(just)

    return f'{x}:NoFormat'


def pretty_print_timedelta(seconds):
    # Convert seconds into timedelta
    t_delta = timedelta(seconds=seconds)

    # Extract days, hours, minutes and seconds
    days = t_delta.days
    if days > 0:
        seconds = t_delta.seconds
        frac_days = days + seconds / (3600 * 24)
        return f"{pretty_format_number(frac_days, short=True)} days"

    hours = t_delta.seconds // 3600
    if hours > 0:
        seconds = t_delta.seconds % 3600
        frac_hours = hours + seconds / 3600
        return f"{pretty_format_number(frac_hours, short=True)} hours"

    minutes = t_delta.seconds // 60
    if minutes > 0:
        seconds = t_delta.seconds % 60
        frac_minutes = minutes + seconds / 60
        return f"{pretty_format_number(frac_minutes, short=True)} minutes"

    return f"{pretty_format_number(t_delta.seconds, short=True)} seconds"


def check_element_type(x):
    unknown = (check_minor_type(x) == 'other')

    if not unknown and not np.isscalar(x) and (has_torch and not (torch.is_tensor(x) and (not len(x.shape)))):
        return 'array'

    if pd.isna(x):
        return 'none'

    if hasattr(x, 'dtype'):
        # this case happens in custom classes that have a dtype attribute
        if unknown:
            return 'other'

        t = str(x.dtype).lower()
    else:
        t = str(type(x)).lower()

    if 'int' in t:
        return 'int'
    if 'bool' in t:
        return 'bool'
    if 'float' in t:
        return 'float'
    if 'str' in t:
        return 'str'
    if 'complex' in t:
        return 'complex'

    return 'object'


def check_minor_type(x):
    if has_torch and isinstance(x, torch.Tensor):
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
    if isinstance(x, set):
        return 'set'
    if has_modin and isinstance(x, mpd.base.BasePandasDataset):
        return 'modin'
    if has_scipy and scipy.sparse.issparse(x):
        return 'scipy_sparse'
    if has_polars and isinstance(x, pl.DataFrame):
        return 'polars'
    if isinstance(x, PurePath) or isinstance(x, PureBeamPath):
        return 'path'
    else:
        return 'other'

def elt_of_list(x):
    if len(x) < 100:
        sampled_indices = range(len(x))
    else:
        sampled_indices = np.random.randint(len(x), size=(100,))

    elt0 = None
    for i in sampled_indices:
        elt = check_element_type(x[i])

        if elt0 is None:
            elt0 = elt

        if elt != elt0:
            return 'object'

    return elt0


def check_type(x, check_minor=True, check_element=True):
    '''

    returns:

    <major type>, <minor type>, <elements type>

    major type: container, array, scalar, none, other
    minor type: dict, list, tuple, set, tensor, numpy, pandas, scipy_sparse, native, none
    elements type: array, int, float, complex, bool, str, object, empty, none, unknown

    '''

    if np.isscalar(x) or (has_torch and torch.is_tensor(x) and (not len(x.shape))):
        mjt = 'scalar'
        if check_minor:
            if type(x) in [int, float, str, complex, bool]:
                mit = 'native'
            else:
                mit = check_minor_type(x)
        else:
            mit = 'na'
        elt = check_element_type(x) if check_element else 'na'

    elif isinstance(x, dict):
        mjt = 'container'
        mit = 'dict'

        if check_element:
            if len(x):
                elt = check_element_type(next(iter(x.values())))
            else:
                elt = 'empty'
        else:
            elt = 'na'

    elif x is None:
        mjt = 'none'
        mit = 'none'
        elt = 'none'

    elif isinstance(x, slice):
        mjt = 'slice'
        mit = 'slice'
        elt = 'slice'

    elif isinstance(x, Counter):
        mjt = 'counter'
        mit = 'counter'
        elt = 'counter'

    else:

        elt = 'unknown'

        if hasattr(x, '__len__'):
            mjt = 'array'
        else:
            mjt = 'other'
        if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, set):
            if not len(x):
                elt = 'empty'
            else:

                if len(x) < 20:
                    elts = [check_element_type(xi) for xi in x]

                else:

                    sample_size = 20
                    try:
                        ind = np.random.randint(len(x), size=(sample_size,))
                        elts = [check_element_type(x[i]) for i in ind]
                    except TypeError:
                        # assuming we are in the case of a set
                        random.sample(list(x), sample_size)

                set_elts = set(elts)
                if len(set_elts) == 1:
                    elt = elts[0]
                elif set_elts == {'int', 'float'}:
                    elt = 'float'
                else:
                    elt = 'object'

            if elt in ['array', 'object', 'none']:
                mjt = 'container'

        mit = check_minor_type(x) if check_minor else 'na'

        if elt:
            if mit in ['numpy', 'tensor', 'pandas', 'scipy_sparse']:
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

        if mit == 'other':
            mjt = 'other'
            elt = 'other'

    return TypeTuple(major=mjt, minor=mit, element=elt)


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


def running_platform() -> str:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return 'notebook'  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return 'ipython'  # Terminal running IPython
        else:
            return 'other'  # Other type (?)
    except NameError:
        if hasattr(__main__, '__file__'):
            return 'script'
        else:
            return 'console'


def is_notebook() -> bool:
    return running_platform() == 'notebook'


def recursive_func(x, func, *args, **kwargs):
    if isinstance(x, dict):
        return {k: recursive_func(v, func, *args, **kwargs) for k, v in x.items()}
    elif isinstance(x, list):
        return [recursive_func(s, func, *args, **kwargs) for s in x]
    elif x is None:
        return None
    else:
        return func(x, *args, **kwargs)


def squeeze_scalar(x, x_type=None):

    if x_type is None:
        x_type = check_type(x)

    if x_type.minor == 'list':
        if len(x) == 1:
            x = x[0]
            x_type = check_type(x)

    if x_type.major == 'scalar':
        if x_type.element == 'int':
            return int(x)
        elif x_type.element == 'float':
            return float(x)
        elif x_type.element == 'complex':
            return complex(x)
        elif x_type.element == 'bool':
            return bool(x)
        elif x_type.element == 'str':
            return str(x)

    return x


def dictionary_iterator(d):

    d = {k: iter(v) for k, v in d.items()}
    for _ in itertools.count():
        try:
            yield {k: next(v) for k, v in d.items()}
        except StopIteration:
            return


def get_item_with_tuple_key(x, key):
    if x is None:
        return None

    if isinstance(key, tuple):
        for k in key:
            if x is None:
                return None
            x = x[k]
        return x
    else:
        return x[key]


def get_closest_item_with_tuple_key(x, key):
    if not isinstance(key, tuple):
        key = (key,)

    for k in key:
        x_type = check_type(x)
        if x_type.minor == 'dict' and k in x:
            x = x[k]
        elif x_type.minor == 'list' and k < len(x):
            x = x[k]
        elif x_type.major == 'container':
            return None
        else:
            return x
    return x


def set_item_with_tuple_key(x, key, value):
    if isinstance(key, tuple):
        for k in key[:-1]:
            x = x[k]
        x[key[-1]] = value
    else:
        x[key] = value


def new_container(k):
    if type(k) is int:
        x = []
    else:
        x = {}

    return x


def insert_tupled_key(x, k, v, default=None):
    if x is None and default is None:
        x = new_container(k[0])
    elif x is None:
        x = default

    xi = x

    for ki, kip1 in zip(k[:-1], k[1:]):

        if isinstance(xi, list):
            assert type(ki) is int and len(xi) == ki, 'Invalid key'
            xi.append(new_container(kip1))

        elif ki not in xi:
            xi[ki] = new_container(kip1)

        xi = xi[ki]

    ki = k[-1]
    if isinstance(xi, list):
        assert type(ki) is int and len(xi) == ki, 'Invalid key'
        xi.append(v)
    else:
        xi[ki] = v

    return x


def build_container_from_tupled_keys(keys, values):
    keys = sorted(keys)

    x = None
    for ki, vi in zip(keys, values):
        x = insert_tupled_key(x, ki, vi)

    return x


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


def get_edit_ratio(s1, s2):
    import Levenshtein as lev
    return lev.ratio(s1, s2)


def get_edit_distance(s1, s2):
    import Levenshtein as lev
    return lev.distance(s1, s2)


def filter_dict(d, keys):
    if keys is True:
        return d

    if keys is False:
        return {}

    keys_type = check_type(keys)

    if keys_type.major == 'scalar':
        keys = [keys]

    elif keys_type.minor in ['list', 'tuple']:
        keys = set(keys)
    else:
        raise ValueError(f"keys must be a scalar, list or tuple. Got {keys_type}")

    return {k: v for k, v in d.items() if k in keys}


def none_function(*args, **kwargs):
    return None


class NoneClass:
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, item):
        return none_function


class NullClass:
    pass


def beam_traceback(exc_type=None, exc_value=None, tb=None, context=3):

    if exc_type is None:
        exc_type, exc_value, tb = sys.exc_info()

    if exc_type is None:
        print("No exception found, Printing stack only:")
        return f"{traceback.print_stack()}"
    else:
        return jupyter_like_traceback(exc_type=exc_type, exc_value=exc_value, tb=tb, context=context)


def jupyter_like_traceback(exc_type=None, exc_value=None, tb=None, context=3):

    if exc_type is None:
        exc_type, exc_value, tb = sys.exc_info()

    # Extract regular traceback
    tb_list = traceback.extract_tb(tb)

    # Generate context for each traceback line
    extended_tb = []
    for frame in tb_list:
        filename, lineno, name, _ = frame
        start_line = max(1, lineno - context)
        lines = linecache.getlines(filename)[start_line - 1: lineno + context]
        for offset, line in enumerate(lines, start_line):
            marker = '---->' if offset == lineno else ''
            extended_tb.append(f"{filename}({offset}): {marker} {line.strip()}")

    # Combine the context with the error message
    traceback_text = '\n'.join(extended_tb)
    return f"{traceback_text}\n{exc_type.__name__}: {exc_value}"


def retry(func=None, retrials=3, logger=None, name=None, verbose=False, sleep=1):
    if func is None:
        return partial(retry, retrials=retrials, sleep=sleep)

    name = name if name is not None else func.__name__
    @wraps(func)
    def wrapper(*args, **kwargs):
        local_retrials = retrials
        last_exception = None
        while local_retrials > 0:
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                last_exception = e
                local_retrials -= 1
                if logger is not None:

                    if local_retrials == np.inf:
                        retry_message = f"Retrying {name}..."
                    else:
                        retry_message = f"Retries {local_retrials}/{retrials} left."

                    logger.warning(f"Exception occurred in {name}. {retry_message}")

                    if verbose:
                        logger.warning(jupyter_like_traceback())

                if local_retrials > 0:
                    time.sleep(sleep * (1 + np.random.rand()))
        if last_exception:
            raise last_exception

    return wrapper


def run_forever(func=None, *args, sleep=1, name=None, logger=None, **kwargs):
    return retry(func=func, *args, retrials=np.inf, logger=logger, name=name, sleep=sleep, **kwargs)


def parse_text_to_protocol(text, protocol='json'):

    if protocol == 'json':
        import json
        res = json.loads(text)
    elif protocol == 'html':
        from bs4 import BeautifulSoup

        res = BeautifulSoup(text, 'html.parser')
    elif protocol == 'xml':
        from lxml import etree

        res = etree.fromstring(text)
    elif protocol == 'csv':
        import pandas as pd
        from io import StringIO

        res = pd.read_csv(StringIO(text))
    elif protocol == 'yaml':
        import yaml
        res = yaml.load(text, Loader=yaml.FullLoader)

    elif protocol == 'toml':
        import toml
        res = toml.loads(text)

    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    return res


class Slicer:
    def __init__(self, x, x_type=None, wrap_object=False):
        self.x = x
        if x_type is None:
            x_type = check_type(x)
        self.x_type = x_type
        self.wrap_object = wrap_object

    def __getitem__(self, item):
        return slice_array(self.x, item, x_type=self.x_type, wrap_object=self.wrap_object)


class DataObject:
    def __init__(self, data, data_type=None):
        self.data = data
        self._data_type = data_type

    @property
    def data_type(self):
        if self._data_type is None:
            self._data_type = check_type(self.data)
        return self._data_type


def slice_array(x, index, x_type=None, indices_type=None, wrap_object=False):

    if x_type is None:
        x_type = check_minor_type(x)
    else:
        x_type = x_type.minor

    if indices_type is None:
        indices_type = check_minor_type(index)
    else:
        indices_type = indices_type.minor

    if indices_type == 'pandas':
        index = index.values
    if indices_type == 'other': # the case where there is a scalar value with a dtype attribute
        index = int(index)
    if x_type == 'numpy':
        return x[index]
    elif x_type == 'pandas':
        return x.iloc[index]
    elif x_type == 'tensor':
        if x.is_sparse:
            x = x.to_dense()
        return x[index]
    elif x_type == 'list':
        return [x[i] for i in index]
    else:
        try:
            xi = x[index]
            if wrap_object:
                xi = DataObject(xi)
            return xi
        except:
            raise TypeError(f"Cannot slice object of type {x_type}")


def is_arange(x, convert_str=True):
    x_type = check_type(x)

    if x_type.element in ['array', 'object', 'empty', 'none', 'unknown']:
        return False

    if convert_str and x_type.element == 'str':
        pattern = re.compile(r'^(?P<prefix>.*?)(?P<number>\d+)(?P<suffix>.*?)$')
        df = []
        for xi in x:
            match = pattern.match(xi)
            if match:
                df.append(match.groupdict())
            else:
                return None, False
        df = pd.DataFrame(df)
        if not df['prefix'].nunique() == 1 or not df['suffix'].nunique() == 1:
            return None, False

        arr_x = df['number'].astype(int).values
    else:
        arr_x = np.array(x)

    try:
        arr_x = arr_x.astype(int)
        argsort = np.argsort(arr_x)
        arr_x = arr_x[argsort]
    except (ValueError, TypeError):
        return None, False

    isa = np.issubdtype(arr_x.dtype, np.number) and (np.abs(np.arange(len(x)) - arr_x).sum() == 0)

    if not isa:
        argsort = None

    return argsort, isa


# convert a dict to list if is_arange is True
def dict_to_list(x, convert_str=True):
    x_type = check_type(x)

    if not x:
        return []

    if x_type.minor != 'dict':
        return x

    keys = np.array(list(x.keys()))
    argsort, isa = is_arange(keys, convert_str=convert_str)

    if isa:
        return [x[k] for k in keys[argsort]]
    else:
        return x


class Timer(object):
    def __init__(self, logger, name='', silence=False, timeout=None, task=None, task_args=None, task_kwargs=None):
        self.name = name
        self.logger = logger
        self.silence = silence
        self.timeout = timeout
        self.task = task
        self.task_args = task_args or ()
        self.task_kwargs = task_kwargs or {}
        self._elapsed = 0
        self.paused = True
        self.t0 = None
        self.executor = None
        self.future = None

    def __enter__(self):
        if not self.silence:
            self.logger.info(f"Starting timer: {self.name}")
        self.run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = self.pause()
        if not self.silence:
            self.logger.info(f"Timer {self.name} paused. Elapsed time: {pretty_format_number(elapsed)} Sec")

    def reset(self):
        self._elapsed = 0
        self.paused = True
        self.t0 = None
        return self

    @property
    def elapsed(self):
        if self.paused:
            return self._elapsed
        return self._elapsed + time.time() - self.t0

    def pause(self):
        if self.paused:
            return self._elapsed
        self._elapsed = self._elapsed + time.time() - self.t0
        self.paused = True
        return self._elapsed

    def run(self):
        self.paused = False
        self.t0 = time.time()

        if self.task is not None:

            self.logger.info(f"Starting task with timeout of {self.timeout} seconds.")
            self.executor = ThreadPoolExecutor(max_workers=1)
            self.future = self.executor.submit(self.task, *self.task_args, **self.task_kwargs)

            try:
                if self.future:
                    self.future.result(timeout=self.timeout)
            except TimeoutError:
                self.logger.info(f"Timer {self.name} exceeded timeout of {self.timeout} seconds.")
            finally:
                elapsed = self.pause()
                if not self.silence:
                    self.logger.info(f"Timer {self.name} paused. Elapsed time: {elapsed} Sec")
                if self.executor:
                    self.executor.shutdown()

    def __str__(self):
        return f"Timer {self.name}: state: {'paused' if self.paused else 'running'}, elapsed: {self.elapsed}"

    def __repr__(self):
        return str(self)


class ThreadSafeDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()

    def __setitem__(self, key, value):
        with self.lock:
            super().__setitem__(key, value)

    def __getitem__(self, key):
        with self.lock:
            return super().__getitem__(key)

    def __delitem__(self, key):
        with self.lock:
            super().__delitem__(key)

    def update(self, *args, **kwargs):
        with self.lock:
            super().update(*args, **kwargs)

    def pop(self, key, *args):
        with self.lock:
            return super().pop(key, *args)

    def clear(self):
        with self.lock:
            super().clear()

    def setdefault(self, key, default=None):
        with self.lock:
            return super().setdefault(key, default)

    def popitem(self):
        with self.lock:
            return super().popitem()

    def copy(self):
        with self.lock:
            return super().copy()

    def keys(self):
        with self.lock:
            return list(super().keys())

    def values(self):
        with self.lock:
            return list(super().values())

    def items(self):
        with self.lock:
            return list(super().items())


def mixin_dictionaries(*dicts):
    res = {}
    for d in dicts[::-1]:
        if d is not None:
            res.update(d)
    return res


def get_class_properties(cls):
    properties = []
    for attr_name in dir(cls):
        attr_value = getattr(cls, attr_name)
        if isinstance(attr_value, property):
            properties.append(attr_name)
    return properties


def pretty_print_dict(d, name):

    # Convert each key-value pair to 'k=v' format and join them with commas
    formatted_str = ", ".join(f"{k}={v}" for k, v in d.items())

    # Enclose in parentheses
    formatted_str = f"{name}({formatted_str})"
    return formatted_str