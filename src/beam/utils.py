import os, sys
from collections import defaultdict
import numpy as np
import torch.distributed as dist
from fnmatch import filter
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm
import random
import torch
import pandas as pd
import pyarrow.feather as feather

try:
    import modin.pandas as mpd
    has_modin = True
except ImportError:
    mpd = None
    has_modin = False

import socket
from contextlib import closing
from collections import namedtuple
from timeit import default_timer as timer
from loguru import logger
from torchvision import transforms
import hashlib
from functools import partial
import itertools
import scipy
from pathlib import Path
import re
from collections import Counter
import time
import inspect
from pathlib import PurePath
from argparse import Namespace
import pyarrow as pa

# logger.remove(handler_id=0)
logger.remove()
logger.add(sys.stdout, level='INFO', colorize=True, format=
           '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | BeamLog | <level>{level}</level> | <level>{message}</level>')


def retrieve_name(var):
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


class PureBeamPath:

    feather_index_mark = "feather_index:"

    def __init__(self, *pathsegments, configuration=None, info=None, client=None, **kwargs):
        super().__init__()

        if len(pathsegments) == 1 and isinstance(pathsegments[0], PureBeamPath):
            pathsegments = pathsegments[0].parts

        self.path = PurePath(*pathsegments)
        self.configuration = Namespace(**kwargs)
        if configuration is not None:

            if type(configuration) == Namespace:
                configuration = configuration.__dict__
            for k, v in configuration.items():
                setattr(self.configuration, k, v)

        self.info = info if info is not None else {}
        self.mode = "rb"
        self.file_object = None
        self.client = client

    def not_empty(self):

        if self.is_dir():
            for p in self.iterdir():
                if p.not_empty():
                    return True
                if p.is_file():
                    return True
        return False

    def rmtree(self):
        if self.is_file():
            self.unlink()
        elif self.is_dir():
            for item in self.iterdir():
                if item.is_dir():
                    rmtree(item)
                else:
                    item.unlink()
            self.rmdir()

    def clean(self):

        if self.exists():
            rmtree(self)
        else:
            if self.parent.exists():
                for p in self.parent.iterdir():
                    if p.stem == self.name:
                        rmtree(p)

        self.mkdir(parents=True)
        self.rmdir()

    def rmdir(self):
        raise NotImplementedError

    def unlink(self):
        raise NotImplementedError

    def __truediv__(self, other):
        return self.joinpath(str(other))

    def __call__(self, mode="rb"):
        self.mode = mode
        return self

    def __str__(self):
        return str(self.path)

    def __repr__(self):
        return self.path.as_uri()

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

    def gen(self, path):
        PathType = type(self)
        return PathType(path, configuration=self.configuration, info=self.info, client=self.client)

    @property
    def parts(self):
        return self.path.parts

    @property
    def drive(self):
        return self.path.drive

    @property
    def root(self):
        return self.path.root

    @property
    def anchor(self):
        return self.path.anchor

    @property
    def parents(self):
        return tuple([self.gen(p) for p in self.path.parents])

    @property
    def parent(self):
        return self.gen(self.path.parent)

    @property
    def name(self):
        return self.path.name

    @property
    def suffix(self):
        return self.path.suffix

    @property
    def suffixes(self):
        return self.path.suffixes

    @property
    def stem(self):
        return self.path.stem

    def as_posix(self):
        return self.path.as_posix()

    def as_uri(self):
        return self.path.as_uri()

    def is_absolute(self):
        return self.path.is_absolute()

    def is_relative_to(self, *other):
        return self.path.is_relative_to(*other)

    def is_reserved(self):
        return self.path.is_reserved()

    def joinpath(self, *other):
        return self.gen(self.path.joinpath(*other))

    def match(self, pattern):
        return self.path.match(pattern)

    def relative_to(self, *other):
        return self.gen(self.path.relative_to(*other))

    def with_name(self, name):
        return self.gen(self.path.with_name(name))

    def with_stem(self, stem):
        return self.gen(self.path.with_stem(stem))

    def with_suffix(self, suffix):
        return self.gen(self.path.with_suffix(suffix))

    def samefile(self, other):
        raise NotImplementedError

    def iterdir(self):
        raise NotImplementedError

    def is_file(self):
        raise NotImplementedError

    def is_dir(self):
        raise NotImplementedError

    def mkdir(self, *args, **kwargs):
        raise NotImplementedError

    def exists(self):
        raise NotImplementedError

    def glob(self, *args, **kwargs):
        raise NotImplementedError

    def rename(self, target):
        return NotImplementedError

    def read(self, ext=None, **kwargs):

        if ext is None:
            ext = self.suffix

        with self(mode=PureBeamPath.mode('read', ext)) as fo:

            if ext == '.fea':

                import pyarrow as pa
                x = feather.read_feather(pa.BufferReader(fo.read()), **kwargs)
                # x = pd.read_feather(fo, **kwargs)

                c = x.columns
                for ci in c:
                    if PureBeamPath.feather_index_mark in ci:
                        index_name = ci.lstrip(PureBeamPath.feather_index_mark)
                        x = x.rename(columns={ci: index_name})
                        x = x.set_index(index_name)
                        break

            elif ext == '.csv':
                x = pd.read_csv(fo, **kwargs)
            elif ext in ['.pkl', '.pickle']:
                x = pd.read_pickle(fo, **kwargs)
            elif ext in ['.npy', '.npz']:
                x = np.load(fo, allow_pickle=True, **kwargs)
            elif ext == '.scipy_npz':
                x = scipy.sparse.load_npz(fo, **kwargs)
            elif ext == '.parquet':
                x = pd.read_parquet(fo, **kwargs)
            elif ext == '.pt':
                x = torch.load(fo, **kwargs)
            elif ext in ['.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt']:
                x = pd.read_excel(fo, **kwargs)
            elif ext == '.avro':
                x = []
                with open(fo, 'rb') as fo:
                    import fastavro
                    for record in fastavro.reader(fo):
                        x.append(record)
            elif ext == '.json':

                #TODO: add json read with fastavro and shcema
                # x = []
                # with open(path, 'r') as fo:
                #     for record in fastavro.json_reader(fo):
                #         x.append(record)

                x = pd.read_json(fo, **kwargs)

            elif ext == '.orc':
                import pyarrow as pa
                x = pa.orc.read(fo, **kwargs)

            else:
                raise ValueError("Unknown extension type.")

        return x

    @staticmethod
    def mode(op, ext):
        if op == 'write':
            m = 'w'
        else:
            m = 'r'

        if ext not in ['.avro', '.json', '.orc']:
            m = f"{m}b"

        return m

    def write(self, x, ext=None, **kwargs):

        if ext is None:
            ext = self.suffix

        path = str(self)

        with self(mode=PureBeamPath.mode('write', ext)) as fo:

            if ext == '.fea':

                if len(x.shape) == 1:
                    x = pd.Series(x)
                    if x.name is None:
                        x.name = 'val'

                x = pd.DataFrame(x)
                x = x.rename({c: str(c) for c in x.columns}, axis=1)

                index_name = x.index.name if x.index.name is not None else 'index'
                df = x.reset_index()
                new_name = PureBeamPath.feather_index_mark + index_name
                x = df.rename(columns={index_name: new_name})
                x.to_feather(fo, **kwargs)
            elif ext == '.csv':
                x = pd.DataFrame(x)
                x.to_csv(fo, **kwargs)
            elif ext in ['.pkl', '.pickle']:
                pd.to_pickle(x, fo, **kwargs)
            elif ext == '.npy':
                np.save(fo, x, **kwargs)
            elif ext == '.npz':
                np.savez(fo, x, **kwargs)
            elif ext == '.scipy_npz':
                scipy.sparse.save_npz(fo, x, **kwargs)
                self.rename(f'{path}.npz', path)
            elif ext == '.parquet':
                x = pd.DataFrame(x)
                x.to_parquet(fo, **kwargs)
            elif ext == '.pt':
                torch.save(x, fo, **kwargs)
            else:
                raise ValueError(f"Unsupported extension type: {ext} for file {x}.")

        return self

    def resolve(self, strict=False):
        return self


class Timer(object):

    def __init__(self, silence=False):
        self.silence = silence
        self._elapsed = 0
        self.t0 = time.time()

    def __enter__(self):
        if not self.silence:
            logger.info(f"Starting timer")
        self.t0 = time.time()
        return self

    @property
    def elapsed(self):
        return self._elapsed + time.time() - self.t0

    def pause(self):
        self._elapsed = self._elapsed + time.time() - self.t0
        return self._elapsed

    def run(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        td = time.time() - self.t0
        if not self.silence:
            logger.info(f"Time elapsed: {pretty_format_number(td)} Sec")


def stack_train_results(results, batch_size=None):

    stacked_results = defaultdict(dict)

    for k_type in results.keys():
        for k_name, v in results[k_type].items():
            stacked_results[k_type][k_name] = v

    stacked_results = stack_batched_results(stacked_results, batch_size=batch_size)

    return stacked_results


def stack_batched_results(results, batch_size=None):
    for n, res in results.items():
        for k, v in res.items():
            v_type = check_type(v)
            if v_type.major == 'container':
                v = recursive_flatten(v)
                v_type = check_type(v)

            if v_type.major == 'container' and v_type.element == 'array':

                if v_type.minor == 'tensor':
                    oprs = {'cat': torch.cat, 'stack': torch.stack}
                elif v_type.minor == 'numpy':
                    oprs = {'cat': np.concatenate, 'stack': np.stack}
                elif v_type.minor == 'pandas':
                    oprs = {'cat': pd.concat, 'stack': pd.concat}
                else:
                    break

                opr = oprs['cat']
                if batch_size is not None and v[0].shape != batch_size:
                    opr = oprs['stack']

                results[n][k] = opr(results[n][k])

            elif v_type.major == 'array' and v_type.minor in ['list', 'tuple']:
                vi_type = check_type(v[0])
                if vi_type.minor in ['numpy', 'native']:
                    results[n][k] = np.stack(results[n][k])
                elif vi_type.minor == 'tensor':
                    results[n][k] = torch.stack(results[n][k])

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


def get_chunks(x, chunksize=None, n_chunks=None, partition=None, dim=0):

    keys = []
    values = []
    for k, v in recursive_chunks(x, chunksize=chunksize, n_chunks=n_chunks, partition=partition, dim=dim):
        keys.append(k)
        values.append(v)

    if not is_arange(keys):
        values = dict(zip(keys, values))
    else:
        values = [values[i] for i in np.argsort(keys)]

    return values


def is_arange(x):

    x_type = check_type(x)

    if x_type.element in ['array', 'object', 'empty', 'none', 'unknown']:
        return False

    arr_x = np.array(x)
    try:
        arr_x = arr_x.astype(int)
        arr_x.sort()
    except (ValueError, TypeError):
        return False

    return np.issubdtype(arr_x.dtype, np.number) and (np.abs(np.arange(len(x)) - arr_x).sum() == 0)


def recursive_chunks(x, chunksize=None, n_chunks=None, partition=None, squeeze=False, dim=0):

    x_type = check_type(x)

    try:
        if (x_type.major == 'container') and (x_type.minor == 'dict'):
            gen = {k: recursive_chunks(v, chunksize=chunksize, n_chunks=n_chunks,
                                       partition=partition, squeeze=squeeze, dim=dim) for k, v in x.items()}

            for i in itertools.count():
                d = {}
                for k, v in gen.items():
                    i, v = next(v)
                    d[k] = v

                yield i, d

        elif x_type.major == 'container':

            gen = [recursive_chunks(s, chunksize=chunksize, n_chunks=n_chunks, partition=partition,
                                    squeeze=squeeze, dim=dim) for s in x]
            for i in itertools.count():
                # yield [next(s) for s in gen]
                l = []
                for k, s in enumerate(gen):
                    i, s = next(s)
                    l.append(s)

                yield i, l

        elif x is None:
            for i in itertools.count():
                yield i, None
        else:
            for k, c in divide_chunks(x, chunksize=chunksize, n_chunks=n_chunks, partition=partition,
                                      squeeze=squeeze, dim=dim):
                yield k, c

    except StopIteration:
        return


def recursive_size(x):

    x_type = check_type(x)
    if x_type.major == 'container':

        keys = []
        values = []

        for k, v in iter_container(x):
            keys.append(k)
            values.append(recursive_size(v))

        if x_type.minor == 'dict':
            values = dict(zip(keys, values))

        return values

    else:

        if x_type.minor == 'tensor':
            return x.element_size() * x.nelement()
        elif x_type.minor in ['numpy', 'scipy_sparse']:
            return x.size * x.dtype.itemsize
        elif x_type.minor == 'pandas':
            try:
                return np.sum(x.memory_usage(index=True, deep=True))
            except:
                return x.size * x.dtype.itemsize
        elif x_type.minor == 'list':
            if len(x) <= 1000:
                return np.sum([sys.getsizeof(i) for i in x])
            ind = np.random.randint(len(x), size=(1000,))
            return len(x) * np.mean([sys.getsizeof(x[i]) for i in ind])
        else:
            return sys.getsizeof(x)


def recursive(func):

    def apply_recursively(x, *args, **kwargs):

        x_type = check_type(x)
        if x_type.major == 'container':

            keys = []
            values = []

            for k, v in iter_container(x):
                keys.append(k)
                values.append(apply_recursively(v, *args, **kwargs))

            if x_type.minor == 'dict':
                values = dict(zip(keys, values))

            return values

        else:

            return func(x, *args, **kwargs)

    return apply_recursively


def recursive_keys(x):

    x_type = check_type(x)
    if x_type.major == 'container':

        keys = []
        values = []

        for k, v in iter_container(x):
            keys.append(k)
            values.append(recursive_keys(v))

        if all([v is None for v in values]):
            return keys

        if x_type.minor == 'dict':
            if not is_arange(values):
                values = dict(zip(keys, values))
            else:
                values = [values[i] for i in np.argsort(keys)]

        return values

    return None


def recursive_size_summary(x, mode='sum'):

    x_type = check_type(x)

    if x_type.minor == 'dict':

        if mode == 'sum':
            return sum([recursive_size_summary(v, mode=mode) for v in x.values()])
        elif mode == 'max':
            return max([recursive_size_summary(v, mode=mode) for v in x.values()])
        else:
            raise NotImplementedError

    elif (x_type.minor in ['list', 'tuple']) and x_type.element in ['object', 'unknown', 'other']:

        if mode == 'sum':
            return sum([recursive_size_summary(s, mode=mode) for s in x])
        elif mode == 'max':
            return max([recursive_size_summary(s, mode=mode) for s in x])
        else:
            raise NotImplementedError

    elif x is None:
        return 0
    else:
        if x_type.minor == 'tensor':
            return x.element_size() * x.nelement()
        elif x_type.minor in ['numpy', 'scipy_sparse']:
            return x.size * x.dtype.itemsize
        elif x_type.minor == 'pandas':
            return np.sum(x.memory_usage(index=True, deep=True))
        else:
            return sys.getsizeof(x)


def divide_chunks(x, chunksize=None, n_chunks=None, partition=None, squeeze=False, dim=0):

    assert ((chunksize is None) != (n_chunks is None)), "divide_chunks requires only one of chunksize|n_chunks"
    x_type = check_type(x, check_element=False)

    assert x_type.major in ['array', 'other'], "divide_chunks supports only array types"

    if n_chunks is not None and hasattr(x, '__len__'):
        n_chunks = min(len(x), n_chunks)

    if x_type.major == 'array':

        l = len(x)

        if chunksize is None:
            chunksize = l // n_chunks

        if n_chunks is None:
            n_chunks = int(np.round(l / chunksize))

        if x_type.minor == 'tensor':
            for i, c in enumerate(torch.tensor_split(x, n_chunks, dim=dim)):
                if squeeze and len(c) == 1:
                    c = c.squeeze()

                yield i, c

        elif x_type.minor == 'pandas' and partition != None:

            grouped = x.groupby(partition, sort=True)
            for k, g in grouped:
                yield k, g

        elif x_type.minor in ['numpy', 'pandas']:

            for i, c in enumerate(np.array_split(x, n_chunks, axis=dim)):
                if squeeze and len(c) == 1:
                    c = c.squeeze()
                yield i, c

        else:
            for j, i in enumerate(np.array_split(np.arange(l), n_chunks)):

                v = x[i[0]:i[-1]+1]
                if squeeze and len(v) == 1:
                    v = v[0]
                yield j, v

    else:

        c = []
        i = 0
        for i, xi in enumerate(iter(x)):

            c.append(xi)
            if len(c) == chunksize:

                if squeeze and len(c) == 1:
                    c = c[0]
                yield i, c

                c = []

        i = i+1 if i > 0 else i
        yield i, c


def recursive_merge(dfs, method='tree', **kwargs):
    if len(dfs) == 1:
        return dfs[0]
    if len(dfs) == 2:
        return pd.merge(dfs[0], dfs[1], **kwargs)
    if method == 'series':
        return recursive_merge([dfs[0], recursive_merge(dfs[1:], method='series', **kwargs)], method='series', **kwargs)
    if method == 'tree':
        return recursive_merge([recursive_merge(dfs[:len(dfs)//2], method='tree', **kwargs),
                                recursive_merge(dfs[len(dfs)//2:], method='tree', **kwargs)], method='tree', **kwargs)
    raise ValueError('Unknown method type')

def is_chunk(path, chunk_pattern='_chunk'):
    return path.is_file() and bool(re.search(rf'\d{6}{chunk_pattern}\.', str(path.name)))


def iter_container(x):
    if hasattr(x, 'items'):
        return x.items()
    return enumerate(x)


def rmtree(path):

    if type(path) is str:
        path = Path(path)

    if path.is_file():
        path.unlink()
    elif path.is_dir():
        for item in path.iterdir():
            if item.is_dir():
                rmtree(item)
            else:
                item.unlink()
        path.rmdir()


def recursive_collate_chunks(*xs, dim=0, on='index', how='outer', method='tree'):

    x_type = check_type(xs[0])
    if x_type.major == 'container':

        values = []
        keys = []

        for k, _ in iter_container(xs[0]):
            values.append(recursive_collate_chunks(*[xi[k] for xi in xs], dim=dim, on=on, how=how, method=method))
            keys.append(k)

        if x_type.minor == 'dict':
            values = dict(zip(keys, values))

        return values

    else:
        return collate_chunks(*xs, dim=dim, on=on, how=how, method=method)


def collate_chunks(*xs, keys=None, dim=0, on='index', how='outer', method='tree'):

    x = list(xs)

    if not len(x):
        return x

    x_type = check_type(x[0], check_element=False)
    if (x_type.major not in ['array', 'other']) or (dim == 1 and x_type.minor not in ['tensor', 'numpy', 'pandas']):
        return x

    if x_type.minor == 'tensor':
        return torch.cat(x, dim=dim)

    elif x_type.minor == 'numpy':
        return np.concatenate(x, axis=dim)

    elif x_type.minor == 'scipy_sparse':

        if dim == 0:
            return scipy.sparse.vstack(x)
        return scipy.sparse.hstack(x)

    elif x_type.minor == 'pandas':
        if on is None or dim == 0:
            return pd.concat(x, axis=dim)
        elif on == 'index':
            return recursive_merge(x, method=method, how=how, left_index=True, right_index=True)
        else:
            return recursive_merge(x, method=method, how=how, on=on)
    else:

        xc = []
        for xi in iter(x):
            xc.extend(xi)
        return xc


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

    unknown = (check_minor_type(x) == 'other')

    if not unknown and not np.isscalar(x) and (not (torch.is_tensor(x) and (not len(x.shape)))):
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

    if unknown:
        return 'other'

    return 'object'


def check_minor_type(x):

    if isinstance(x, torch.Tensor):
        return 'tensor'
    if isinstance(x, np.ndarray):
        return 'numpy'
    if isinstance(x, pd.core.base.PandasObject):
        return 'pandas'
    if has_modin and isinstance(x, mpd.base.BasePandasDataset):
        return 'modin'
    if scipy.sparse.issparse(x):
        return 'scipy_sparse'
    if isinstance(x, dict):
        return 'dict'
    if isinstance(x, list):
        return 'list'
    if isinstance(x, tuple):
        return 'tuple'
    if isinstance(x, Path) or isinstance(x, PureBeamPath):
        return 'path'
    else:
        return 'other'


type_tuple = namedtuple('Type', 'major minor element')


def check_type(x, check_minor=True, check_element=True):
    '''

    returns:

    <major type>, <minor type>, <elements type>

    major type: container, array, scalar, none, other
    minor type: dict, list, tuple, tensor, numpy, pandas, scipy_sparse, native, none
    elements type: array, int, float, str, object, empty, none, unknown

    '''

    if np.isscalar(x) or (torch.is_tensor(x) and (not len(x.shape))):
        mjt = 'scalar'
        if check_minor:
            if type(x) in [int, float, str]:
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
        if isinstance(x, list) or isinstance(x, tuple):
            if not len(x):
                elt = 'empty'
            else:

                if len(x) < 100:
                    elts = [check_element_type(xi) for xi in x]

                else:
                    ind = np.random.randint(len(x), size=(100, ))
                    elts = [check_element_type(x[i]) for i in ind]

                if len(set(elts)) == 1:
                    elt = elts[0]
                else:
                    elt = 'object'

            if elt in ['array', 'object']:
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

# def is_notebook():
#     return '_' in os.environ and 'jupyter' in os.environ['_']


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


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


def recursive_func(x, func, *args, **kwargs):

    if isinstance(x, dict):
        return {k: recursive_func(v, func, *args, **kwargs) for k, v in x.items()}
    elif isinstance(x, list) or isinstance(x, tuple):
        return [recursive_func(s, func, *args, **kwargs) for s in x]
    elif x is None:
        return None
    else:
        return func(x, *args, **kwargs)


def recursive_flat_array(x):

    x_type = check_type(x)

    if x_type.minor == 'numpy':
        return x.flatten().tolist()
    elif x_type.minor == 'tensor':
        return x.flatten().tolist()
    elif x_type.minor == 'pandas':
        return x.values.flatten().tolist()
    elif x_type.minor == 'scipy_sparse':
        return x.toarray().flatten().tolist()
    elif x_type.minor in ['list', 'tuple']:
        if x_type.element != 'array':
            return list(x)

        l = []
        for xi in x:
            l.extend(recursive_flat_array(xi))
        return l

    elif x_type.minor == 'native':
        return [x]

    else:
        return [x]


def recursive_flatten(x, flat_array=False):

    x_type = check_type(x)

    if x_type.major == 'container':
        l = []
        for i, xi in iter_container(x):
            l.extend(recursive_flatten(xi, flat_array=flat_array))
        return l
    else:
        if not flat_array or x_type.major == 'scalar':
            return [x]
        else:
            return recursive_flat_array(x)


def recursive_flatten_with_keys(x):

    x_type = check_type(x)

    if x_type.major == 'container':
        d = {}
        for i, xi in iter_container(x):
            di = recursive_flatten_with_keys(xi)
            di = {(i, *k): v for k, v in di.items()}
            d.update(di)
        return d
    else:
        return {tuple(): x}


def get_item_with_tuple_key(x, key):

    if isinstance(key, tuple):
        for k in key:
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


def insert_tupled_key(x, k, v):
    if x is None:
        x = new_container(k[0])

    xi = x

    for ki, kip1 in zip(k[:-1], k[1:]):

        if type(ki) is int and len(xi) == ki:
            xi.append(new_container(kip1))

        elif ki not in xi:
            xi[ki] = new_container(kip1)

        xi = xi[ki]

    ki = k[-1]
    if type(ki) is int and len(xi) == ki:
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


@recursive
def recursive_batch(x, index):

    if x is None:
        return None
    elif hasattr(x, 'iloc'):
        return x.iloc[index]
    else:
        return x[index]


@recursive
def recursive_len(x):

    x_type = check_type(x)

    if x_type.minor == 'scipy_sparse':
        return x.shape[0]

    if hasattr(x, '__len__'):
        return len(x)

    if x is None:
        return None

    return 1


@recursive
def recursive_types(x):

    x_type = check_type(x)
    return f'{x_type.major}.{x_type.minor}.{x_type.element}'


@recursive
def recursive_shape(x):
    if hasattr(x, 'shape'):
        return x.shape
    if hasattr(x, '__len__'):
        return len(x)
    return None


@recursive
def recursive_slice(x, s):
    if x is None:
        return None
    return x.__getitem__(s)


@recursive
def recursive_slice_columns(x, columns):

    x_type = check_type(x)

    if x is None:
        return None
    elif x_type.minor == 'pandas':
        return x[columns]
    else:
        return x[:, columns]

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


def container_len(x):

    if isinstance(x, dict):
        for xi in x.values():
            try:
                return container_len(xi)
            except TypeError:
                # case of None
                pass

    elif isinstance(x, list) or isinstance(x, tuple):
        for xi in x:
            try:
                return container_len(xi)
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

    if dtype is None and hasattr(x, 'dtype'):
        if 'int' in str(x.dtype):
            dtype = torch.int64
        else:
            dtype = torch.float32

    if check_type(x, check_element=False).minor == 'pandas':
        x = x.values

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