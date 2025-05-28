import itertools
import os
import sys
from collections import defaultdict
import types

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
from inspect import isclass, getmro

import socket
from contextlib import closing
from collections import namedtuple
from timeit import default_timer as timer

import inspect
from argparse import Namespace
from functools import wraps, partial, cached_property as native_cached_property
from collections import OrderedDict
import yaml
import json
from dataclasses import asdict, is_dataclass


# do not delete this import (it is required as some modules import the following imported functions from this file)
from ..type import check_type, check_minor_type, check_element_type, is_scalar, is_container, is_cached_property, Types, \
    is_beam_path

DataBatch = namedtuple("DataBatch", "index label data")


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


def get_public_ip():
    import requests
    try:
        response = requests.get("https://api64.ipify.org?format=json")
        ip = response.json().get("ip")
        return ip
    except requests.RequestException:
        return "127.0.0.1"


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
    from ..logging import beam_logger as logger

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
    elif application == 'debugpy':
        first_beam_range = 63
        first_global_range = 28263
    else:
        first_beam_range = 2
        first_global_range = 30000

    if port is None:

        port_range = None

        if get_port_from_beam_port_range:

            base_range = None
            if os.path.isfile('/workspace/configuration/config.csv'):
                conf = pd.read_csv('/workspace/configuration/config.csv')
                base_range = int(conf.set_index('parameters').drop_duplicates().loc['initials'].iloc[0])

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
    if x_type.minor in [Types.numpy, Types.pandas, Types.tensor, Types.cudf] and 'bool' in str(x.dtype).lower():
        return True
    elif x_type.minor == Types.polars and 'Boolean' == str(next(iter(x.schema.values()))):
        return True
    if x_type.minor == Types.list and len(x) and isinstance(x[0], bool):
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

    big_num = 1000 if short else 10000
    normal_num = 100 if short else 1000
    small_num = 0.1 if short else 0.001

    if x is None or np.isinf(x) or np.isnan(x):
        return f'{x}'.ljust(just)
    if int(x) == x and np.abs(x) < big_num:
        return f'{int(x)}'.ljust(just)
    if np.abs(x) >= big_num or np.abs(x) < small_num:
        if short:
            return f'{x:.1e}'.ljust(just)
        else:
            return f'{float(x):.4}'.ljust(just)
    if np.abs(x) >= normal_num:
        return f'{x:.1f}'.ljust(just)
    if np.abs(x) < big_num and np.abs(x) >= small_num:
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


def parse_string_number(x, time_units=None, unit_prefixes=None, timedelta_format=True, return_units=False):
    v, u = _parse_string_number(x, time_units=time_units, unit_prefixes=unit_prefixes, timedelta_format=timedelta_format)
    if return_units:
        return v, u
    return v


def int_or_float(x):
    try:
        int_x = int(x)
        if int_x == float(x):
            return int_x
    except:
        pass
    try:
        float_x = float(x)
        return float_x
    except:
        raise ValueError(f"Cannot convert {x} to int or float")


def _parse_string_number(x, time_units=None, unit_prefixes=None, timedelta_format=True):

    try:
        return int_or_float(x), ''
    except:
        pass

    # check for unit prefix or time format
    match = re.match(r'^(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?P<unit>[a-zA-Z]+)$', x)
    if match:
        unit = match.group('unit')
        val = int_or_float(match.group('value'))

        # if time format return timedelta
        if time_units is None:
            time_units = {'s': ('seconds', 1), 'm': ('minutes', 60), 'h': ('hours', 60*60), 'd': ('days', 60*60*24),
                          'ms': ('milliseconds', 1e-3), 'us': ('microseconds', 1e-6), 'ns': ('nanoseconds', 1e-9),
                          'y': ('years', 1), 'mo': ('months', 1), 'w': ('weeks', 60*60*24*7),
                          'sec': ('seconds', 1), 'min': ('minutes', 60), 'hours': ('hours', 60*60), 'days': ('days', 60*60*24),
                          'weeks': ('weeks', 60*60*24*7), 'months': ('months', 1), 'minutes': ('minutes', 60),
                          'seconds': ('seconds', 1), 'years': ('years', 1)}

        if unit in time_units.keys():
            if timedelta_format:
                    return timedelta(**{time_units[unit][0]: val}), unit
            return val * time_units[unit][1], unit

        if unit_prefixes is None:
            # if in unit prefix return the value in the unit
            unit_prefixes = {'k': int(1e3), 'M': int(1e6), 'Gi': int(1e9), 'T': int(1e12),
                             'K': int(1e3), 'm': int(1e-3), 'u': int(1e-6), 'n': int(1e-9),
                             'G': int(1e9), 'Mi': int(1e6), 'p': int(1e-12), 'f': int(1e-15)}

        if unit in unit_prefixes.keys():
            return val * unit_prefixes[unit], unit

    return x, ''


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
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return 'notebook'  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return 'ipython'  # Terminal running IPython
        else:
            return 'other'  # Other type (?)
    except (NameError, ModuleNotFoundError):
        if hasattr(__main__, '__file__'):
            return 'script'
        else:
            return 'console'


def is_notebook() -> bool:
    return running_platform() == 'notebook'


def recursive_func(x, func, *args, _ignore_none=False, **kwargs):
    if isinstance(x, dict):
        return {k: recursive_func(v, func, *args, _ignore_none=_ignore_none, **kwargs) for k, v in x.items()}
    elif isinstance(x, list):
        return [recursive_func(s, func, *args, _ignore_none=_ignore_none, **kwargs) for s in x]
    elif _ignore_none and x is None:
        return None
    else:
        return func(x, *args, **kwargs)


def squeeze_scalar(x, x_type=None):

    if x_type is None:
        x_type = check_type(x)

    if x_type.minor == Types.list:
        if len(x) == 1:
            x = x[0]
            x_type = check_type(x)

    if x_type.major == Types.scalar:
        if x_type.element == Types.int:
            return int(x)
        elif x_type.element == Types.float:
            return float(x)
        elif x_type.element == Types.complex:
            return complex(x)
        elif x_type.element == Types.bool:
            return bool(x)
        elif x_type.element == Types.str:
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
        if x_type.minor == Types.dict and k in x:
            x = x[k]
        elif x_type.minor == Types.list and k < len(x):
            x = x[k]
        elif x_type.major == Types.container:
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


def insert_tupled_key(x, k, v, default=None, keys=None):
    if x is None and default is None:
        if keys is None:
            x = new_container(k[0])
        else:
            keys = [ki[0] for ki in keys]
            if is_arange(keys, sort=False):
                x = []
            else:
                x = {}

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


def build_container_from_tupled_keys(keys, values, sorted_keys=None):

    if sorted_keys is None:
        sorted_keys = sorted(keys)
    kv_map = {k: v for k, v in zip(keys, values)}
    sorted_values = [kv_map[k] for k in sorted_keys]

    x = None
    for ki, vi in zip(sorted_keys, sorted_values):
        x = insert_tupled_key(x, ki, vi, keys=sorted_keys)

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

    # check if x has len and its len equals 1, in this case we don't need to show the progress bar
    elif hasattr(x, '__len__') and len(x) == 1:
        yield from x

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

    if keys_type.major == Types.scalar:
        keys = [keys]

    elif keys_type.minor in [Types.list, Types.tuple]:
        keys = set(keys)
    else:
        raise ValueError(f"keys must be a scalar, list or tuple. Got {keys_type}")

    return {k: v for k, v in d.items() if k in keys}


def none_function(*args, **kwargs):
    return None


def identity_function(x, **kwargs):
    return x


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


def retry(func=None, retries=3, logger=None, name=None, verbose=False, sleep=1, timeout=None):
    if func is None:
        return partial(retry, retries=retries, sleep=sleep)

    name = name if name is not None else func.__name__
    @wraps(func)
    def wrapper(*args, **kwargs):
        local_retries = retries
        last_exception = None
        while local_retries > 0:
            try:
                if timeout is not None:
                    # Use Timer to run the task with timeout
                    timer = Timer(logger=logger, name=name, silent=not verbose, timeout=timeout, task=func,
                                  task_args=args, task_kwargs=kwargs)
                    return timer.run()  # Run the task within the Timer and return the result
                else:
                    # Run the task without timeout
                    return func(*args, **kwargs)
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                last_exception = e
                local_retries -= 1
                if logger is not None:

                    if local_retries == np.inf:
                        retry_message = f"Retrying {name}..."
                    else:
                        retry_message = f"Retries {local_retries}/{retries} left."

                    logger.warning(f"Exception occurred in {name}. {retry_message}")

                    if verbose:
                        logger.warning(jupyter_like_traceback())

                if local_retries > 0:
                    time.sleep(sleep * (1 + np.random.rand()))
        if last_exception:
            raise last_exception

    return wrapper


def run_forever(func=None, *args, sleep=1, name=None, logger=None, **kwargs):
    return retry(func=func, *args, retries=np.inf, logger=logger, name=name, sleep=sleep, **kwargs)


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

    if indices_type in [Types.pandas, Types.polars]:
        index = index.values
    if indices_type == Types.other:  # the case where there is a scalar value with a dtype attribute
        index = int(index)
    if x_type in [Types.numpy, Types.polars]:
        return x[index]
    elif x_type in [Types.pandas, Types.cudf]:
        return x.iloc[index]
    elif x_type == Types.tensor:
        if x.is_sparse:
            x = x.to_dense()
        return x[index]
    elif x_type == Types.list:
        return [x[i] for i in index]
    else:
        try:
            xi = x[index]
            if wrap_object:
                xi = DataObject(xi)
            return xi
        except:
            raise TypeError(f"Cannot slice object of type {x_type}")


def is_arange(x, convert_str=True, sort=True):
    x_type = check_type(x)

    if x_type.element in [Types.array, Types.object, Types.empty, Types.none, Types.unknown]:
        return None, False

    if convert_str and x_type.element == Types.str:
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
        if sort:
            argsort = np.argsort(arr_x)
            arr_x = arr_x[argsort]
    except (ValueError, TypeError):
        return None, False

    isa = np.issubdtype(arr_x.dtype, np.number) and (np.abs(np.arange(len(x)) - arr_x).sum() == 0)

    if not isa:
        argsort = None

    if sort:
        return argsort, isa
    else:
        return isa


# convert a dict to list if is_arange is True
def dict_to_list(x, convert_str=True):
    x_type = check_type(x)

    if not x:
        return []

    if x_type.minor != Types.dict:
        return x

    keys = np.array(list(x.keys()))
    argsort, isa = is_arange(keys, convert_str=convert_str)

    if isa:
        return [x[k] for k in keys[argsort]]
    else:
        return x


class Timer(object):
    def __init__(self, logger, name='', silent=False, timeout=None, task=None, task_args=None, task_kwargs=None,
                 graceful=False):
        self.name = name
        self.logger = logger
        self.silent = silent
        self.timeout = timeout
        self.task = task
        self.task_args = task_args or ()
        self.task_kwargs = task_kwargs or {}
        self._elapsed = 0
        self.paused = True
        self.t0 = None
        self.executor = None
        self.future = None
        self.graceful = graceful

    def __enter__(self):
        if not self.silent:
            self.logger.info(f"Starting timer: {self.name}")
        self.run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = self.pause()
        if not self.silent:
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

    def run(self, *args, **kwargs):
        self.paused = False
        self.t0 = time.time()

        args = [*args, *self.task_args]
        kwargs = {**kwargs, **self.task_kwargs}

        if self.task is not None:

            self.logger.info(f"Starting task with timeout of {self.timeout} seconds.")
            self.executor = ThreadPoolExecutor(max_workers=1)
            self.future = self.executor.submit(self.task, *args, **kwargs)

            res = None
            try:
                res = self.future.result(timeout=self.timeout)
            except TimeoutError:
                self.logger.info(f"Timer {self.name} exceeded timeout of {self.timeout} seconds.")
                if self.graceful:
                    self.future.cancel()
                else:
                    raise TimeoutError(f"Timer {self.name} exceeded timeout of {self.timeout} seconds.")
            finally:
                elapsed = self.pause()
                if not self.silent:
                    self.logger.info(f"Timer {self.name} paused. Elapsed time: {elapsed} Sec")
                if self.executor:
                    self.executor.shutdown()

            return res

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


def get_cached_properties(obj):
    cached_props = {}
    # Inspect the class dictionary for cached_property instances
    for name, prop in inspect.getmembers(type(obj), lambda member: isinstance(member, native_cached_property)):
        # Check if the instance dictionary has a cached value for this property
        if name in obj.__dict__:
            cached_props[name] = getattr(obj, name)
    return cached_props


def pretty_print_dict(d, name=None, dense=True):

    # Convert each key-value pair to 'k=v' format and join them with commas
    if dense:
        separator = ', '
    else:
        separator = ',\n'
    formatted_str = f"{separator}".join(f"{k}={v}" for k, v in d.items())

    # Enclose in parentheses
    if name is None:
        formatted_str = f"{formatted_str}"
    else:
        formatted_str = f"{name}:({formatted_str})"

    return formatted_str


def pprint(obj, logger=None, level='info'):
    # print with dump to yaml
    if logger is None:
        print(yaml.dump(obj))
    else:
        getattr(logger, level)(yaml.dump(obj))


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


class LimitedSizeDict(OrderedDict):
    def __init__(self, size_limit=None, on_removal=None):
        super().__init__()
        self.size_limit = size_limit
        self.on_removal = on_removal  # Callback function to call on removal

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        OrderedDict.__setitem__(self, key, value)
        if self.size_limit is not None and len(self) > self.size_limit:
            oldest_key, oldest_value = self.popitem(last=False)
            if self.on_removal:  # Check if a callback function is provided
                self.on_removal(oldest_key, oldest_value)  # Call the callback with the removed key and value


class LimitedSizeDictFactory:
    def __init__(self, size_limit=None, on_removal=None):
        self.size_limit = size_limit
        self.on_removal = on_removal

    def __call__(self):
        return LimitedSizeDict(size_limit=self.size_limit, on_removal=self.on_removal)


class CachedAttributeException(Exception):
    """Custom exception to be raised instead of AttributeError in cached properties."""
    pass


class cached_property(native_cached_property):
    def __get__(self, instance, owner=None):
        try:
            # Use super() to call the __get__ method of the parent cached_property class
            return super().__get__(instance, owner)
        except AttributeError as e:
            # Change the AttributeError to BeamAttributeException
            raise CachedAttributeException(f"An AttributeError occurred in cached_property: {self.attrname}") from e


def getmembers(object, predicate=None):
    """Return all members of an object as (name, value) pairs sorted by name.
    Optionally, only return members that satisfy a given predicate."""
    if isclass(object):
        mro = (object,) + getmro(object)
    else:
        mro = ()
    results = []
    processed = set()
    names = dir(object)
    # :dd any DynamicClassAttributes to the list of names if object is a class;
    # this may result in duplicate entries if, for example, a virtual
    # attribute with the same name as a DynamicClassAttribute exists
    try:
        for base in object.__bases__:
            for k, v in base.__dict__.items():
                if isinstance(v, types.DynamicClassAttribute):
                    names.append(k)
    except AttributeError:
        pass
    for key in names:
        # First try to get the value via getattr.  Some descriptors don't
        # like calling their __get__ (see bug #1785), so fall back to
        # looking in the __dict__.
        try:
            if is_cached_property(object, key) and key not in object.__dict__.keys():
                continue
            value = getattr(object, key)
            # handle the duplicate key
            if key in processed:
                raise AttributeError
        except AttributeError:
            for base in mro:
                if key in base.__dict__:
                    value = base.__dict__[key]
                    break
            else:
                # could be a (currently) missing slot member, or a buggy
                # __dir__; discard and move on
                continue
        if not predicate or predicate(key, value):
            results.append((key, value))
        processed.add(key)
    results.sort(key=lambda pair: pair[0])
    return results


def safe_getmembers(obj, predicate=None):

    if predicate is None:
        predicate = lambda value: True

    return getmembers(obj, predicate=lambda key, value: predicate(value) and not key.startswith('_'))


class BeamJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        if is_beam_path(obj):
            return obj.as_uri()
        return json.JSONEncoder.default(self, obj)


def serialize_annotation(annotation):
    """Convert annotation to a serializable format."""
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, type):
        return annotation.__name__
    if isinstance(annotation, str):
        return annotation
    return repr(annotation)

def deserialize_annotation(annotation_str, global_ns=None):
    """Convert serialized annotation back to its original format."""
    if annotation_str is None:
        return inspect.Parameter.empty
    if global_ns is None:
        global_ns = globals()
    try:
        return eval(annotation_str, global_ns)
    except Exception:
        return annotation_str


def signature_to_dict(signature):
    """Convert a Signature object to a dictionary representation."""

    def default_param(param):
        param_type = check_type(param)
        if not param_type.is_scalar:
            return str(param)
        return param

    return {
        'parameters': [
            {
                'name': param.name,
                'kind': param.kind.name,
                'default': default_param(param.default) if param.default is not inspect.Parameter.empty else None,
                'annotation': serialize_annotation(param.annotation)
            }
            for param in signature.parameters.values()
        ],
        'return_annotation': serialize_annotation(signature.return_annotation)
    }


def dict_to_signature(d, global_ns=None):
    """Convert a dictionary representation back to a Signature object."""
    parameters = [
        inspect.Parameter(
            name=param['name'],
            kind=getattr(inspect.Parameter, param['kind']),
            default=param['default'] if param['default'] is not None else inspect.Parameter.empty,
            annotation=deserialize_annotation(param['annotation'], global_ns)
        )
        for param in d['parameters']
    ]
    return_annotation = deserialize_annotation(d['return_annotation'], global_ns)
    return inspect.Signature(parameters, return_annotation=return_annotation)


def return_constant(a):
    def _return_constant(*args, **kwargs):
        return a
    return _return_constant


def get_number_of_cores():
    try:
        import multiprocessing as mp
        return os.cpu_count() or mp.cpu_count()
    except (AttributeError, NotImplementedError):
        return 1  # Fallback to 1 if the number of cores cannot be determined



def remote_debugger(port=None, logger=None):
    if port is None:
        port = int(find_port(application='debugpy'))
    import debugpy
    debugpy.listen(port)
    print(f"Waiting for debugger to attach on port {port}...")
    debugpy.wait_for_client()
    print("Debugger attached")


