import os, sys
from collections import defaultdict
import numpy as np
from fnmatch import filter
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm

import pandas as pd
import json
import __main__
from datetime import timedelta

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

import socket
from contextlib import closing
from collections import namedtuple
from timeit import default_timer as timer

from pathlib import Path
from collections import Counter
import inspect
from pathlib import PurePath
from argparse import Namespace
from urllib.parse import urlparse, urlunparse, parse_qsl, ParseResult

TypeTuple = namedtuple('TypeTuple', 'major minor element')
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


class BeamURL:

    def __init__(self, url=None, scheme=None, hostname=None, port=None, username=None, password=None, path=None,
                 fragment=None, params=None, **query):

        self._url = url
        self._parsed_url = None
        if url is None:
            netloc = BeamURL.to_netloc(hostname=hostname, port=port, username=username, password=password)
            query = BeamURL.dict_to_query(**query)
            if scheme is None:
                scheme = 'file'
            if path is None:
                path = ''
            if netloc is None:
                netloc = ''
            if query is None:
                query = ''
            if fragment is None:
                fragment = ''
            if params is None:
                params = ''
            self._parsed_url = ParseResult(scheme=scheme, netloc=netloc, path=path, params=params, query=query,
                                           fragment=fragment)

        assert self._url is not None or self._parsed_url is not None, 'Either url or parsed_url must be provided'

    @property
    def parsed_url(self):
        if self._parsed_url is not None:
            return self._parsed_url
        self._parsed_url = urlparse(self._url)
        return self._parsed_url

    @property
    def url(self):
        if self._url is not None:
            return self._url
        self._url = urlunparse(self._parsed_url)
        return self._url

    def __repr__(self):
        return self.url

    def __str__(self):

        netloc = BeamURL.to_netloc(hostname=self.hostname, port=self.port, username=self.username)
        parsed_url = ParseResult(scheme=self.scheme, netloc=netloc, path=self.path, params=None, query=None,
                                 fragment=None)
        return urlunparse(parsed_url)

    @property
    def scheme(self):
        return self.parsed_url.scheme

    @property
    def protocol(self):
        return self.scheme

    @property
    def username(self):
        if self.parsed_url.netloc is None:
            return None
        return self.parsed_url.username

    @property
    def hostname(self):
        if self.parsed_url.netloc is None:
            return None
        return self.parsed_url.hostname

    @property
    def password(self):
        if self.parsed_url.netloc is None:
            return None
        return self.parsed_url.password

    @property
    def port(self):
        if self.parsed_url.netloc is None:
            return None
        return self.parsed_url.port

    @property
    def path(self):
        return self.parsed_url.path

    @property
    def query_string(self):
        return self.parsed_url.query

    @property
    def query(self):
        return dict(parse_qsl(self.parsed_url.query))

    @property
    def fragment(self):
        return self.parsed_url.fragment

    @property
    def params(self):
        return self.parsed_url.params

    @staticmethod
    def to_netloc(hostname=None, port=None, username=None, password=None):

        if not hostname:
            return None

        netloc = hostname
        if username:
            if password:
                username = f"{username}:{password}"
            netloc = f"{username}@{netloc}"
        if port:
            netloc = f"{netloc}:{port}"
        return netloc

    @staticmethod
    def to_path(path):
        return PurePath(path).as_posix()

    @staticmethod
    def query_to_dict(query):
        return dict(parse_qsl(query))

    @staticmethod
    def dict_to_query(**query):
        return '&'.join([f'{k}={v}' for k, v in query.items() if v is not None])

    @classmethod
    def from_string(cls, url):
        parsed_url = urlparse(url)
        return cls(url, parsed_url)


def retrieve_name(var):
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


class PureBeamPath:
    feather_index_mark = "feather_index:"

    text_extensions = ['.txt', '.text', '.py', '.sh', '.c', '.cpp', '.h', '.hpp', '.java', '.js', '.css', '.html']
    textual_extensions = text_extensions + ['.avro', '.json', '.orc', '.ndjson']

    def __init__(self, *pathsegments, url=None, scheme=None, hostname=None, port=None, username=None, password=None,
                 fragment=None, params=None, client=None, **kwargs):
        super().__init__()

        if len(pathsegments) == 1 and isinstance(pathsegments[0], PureBeamPath):
            pathsegments = pathsegments[0].parts

        self.path = PurePath(*pathsegments)

        if url is not None:
            scheme = url.scheme
            hostname = url.hostname
            port = url.port
            username = url.username
            password = url.password
            fragment = url.fragment
            params = url.params
            kwargs = url.query

        self.url = BeamURL(scheme=scheme, hostname=hostname, port=port, username=username, fragment=fragment,
                           params=params, password=password, path=str(self.path), **kwargs)

        self.mode = "rb"
        self.file_object = None
        self.client = client

    def __getstate__(self):
        return self.as_uri()

    def __setstate__(self, state):

        url = BeamURL.from_string(state)

        self.__init__(url.path, hostname=url.hostname, port=url.port, username=url.username,
                      password=url.password, fragment=url.fragment, params=url.params, client=None, **url.query)

    def __iter__(self):
        for p in self.iterdir():
            yield p

    def __getitem__(self, name):
        return self.joinpath(name)

    def __setitem__(self, key, value):
        p = self.joinpath(key)
        p.write(value)

    def not_empty(self):

        if self.is_dir():
            for p in self.iterdir():
                if p.not_empty():
                    return True
                if p.is_file():
                    return True
        return False

    def copy(self, dst):

        if self.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            for p in self.iterdir():
                p.copy(dst / p.name)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            with self.open("rb") as f:
                with dst.open("wb") as g:
                    g.write(f.read())

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

    def walk(self):
        dirs = []
        files = []
        for p in self.iterdir():
            if p.is_dir():
                dirs.append(p.name)
            else:
                files.append(p.name)

        yield self, dirs, files

        for dir in dirs:
            yield from self.joinpath(dir).walk()

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

    def getmtime(self):
        return None

    def stat(self):
        raise NotImplementedError

    def rmdir(self):
        raise NotImplementedError

    def unlink(self, **kwargs):
        raise NotImplementedError

    def __truediv__(self, other):
        return self.joinpath(str(other))

    def __fspath__(self, mode="rb"):
        raise TypeError("For BeamPath (named bp), use bp.open(mode) instead of open(bp, mode)")

    def __call__(self, mode="rb"):
        self.mode = mode
        return self

    def open(self, mode="rb"):
        self.mode = mode
        return self

    def close(self):
        if self.file_object is not None:
            self.file_object.close()
            self.file_object = None

    def __str__(self):
        return str(self.path)

    def __repr__(self):
        if self.is_absolute():
            return str(self.url)
        return str(self.path)

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

    @property
    def hostname(self):
        return self.url.hostname

    @property
    def port(self):
        return self.url.port

    @property
    def username(self):
        return self.url.username

    @property
    def password(self):
        return self.url.password

    @property
    def fragment(self):
        return self.url.fragment

    @property
    def params(self):
        return self.url.params

    @property
    def query(self):
        return self.url.query

    def gen(self, path):

        PathType = type(self)

        return PathType(path, client=self.client, hostname=self.hostname, port=self.port, username=self.username,
                        password=self.password, fragment=self.fragment, params=self.params, **self.query)

    @property
    def parts(self):
        return self.path.parts

    @property
    def drive(self):
        return self.path.drive

    @property
    def root(self):
        return self.path.root

    def is_root(self):
        return str(self.path) == '/'

    @property
    def anchor(self):
        return self.gen(self.path.anchor)

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
        return self.url.url

    def is_absolute(self):
        return self.path.is_absolute()

    def is_relative_to(self, *other):
        if len(other) == 1 and isinstance(other[0], PureBeamPath):
            other = str(other[0])
        else:
            other = str(PureBeamPath(*other))
        return self.path.is_relative_to(other)

    def is_reserved(self):
        return self.path.is_reserved()

    def joinpath(self, *other):
        return self.gen(self.path.joinpath(*[str(o) for o in other]))

    def match(self, pattern):
        return self.path.match(pattern)

    def relative_to(self, *other):
        if len(other) == 1 and isinstance(other[0], PureBeamPath):
            other = str(other[0])
        else:
            other = str(PureBeamPath(*other))
        return PureBeamPath(self.path.relative_to(other))

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

    def replace(self, target):
        return NotImplementedError

    def read_io(self, mode=None, **kwargs):

        if mode is None:
            mode = self.mode

        if mode == 'rb':
            return self.file_object.read_bytes()
        elif mode == 'r':
            return self.file_object.read_text(**kwargs)
        else:
            raise NotImplementedError

    def read(self, ext=None, **kwargs):

        if ext is None:
            ext = self.suffix

        with self(mode=PureBeamPath.mode('read', ext)) as fo:

            if ext == '.fea':

                import pyarrow as pa
                # x = feather.read_feather(pa.BufferReader(fo.read()), **kwargs)
                x = pd.read_feather(fo, **kwargs)

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
            elif ext in PureBeamPath.text_extensions:
                if 'readlines' in kwargs and kwargs['readlines']:
                    x = fo.readlines()
                else:
                    x = fo.read()
            elif ext == '.scipy_npz':
                import scipy
                x = scipy.sparse.load_npz(fo, **kwargs)
            elif ext == '.flac':
                import soundfile
                x = soundfile.read(fo, **kwargs)
            elif ext == '.parquet':
                x = pd.read_parquet(fo, **kwargs)
            elif ext == '.pt':
                import torch
                x = torch.load(fo, **kwargs)
            elif ext in ['.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt']:
                x = pd.read_excel(fo, **kwargs)
            elif ext == '.avro':
                x = []
                with open(fo, 'rb') as fo:
                    import fastavro
                    for record in fastavro.reader(fo):
                        x.append(record)
            elif ext in ['.json', '.ndjson']:

                # TODO: add json read with fastavro and shcema
                # x = []
                # with open(path, 'r') as fo:
                #     for record in fastavro.json_reader(fo):
                #         x.append(record)

                nd = ext == '.ndjson'
                try:
                    x = pd.read_json(fo, lines=nd, **kwargs)
                except:
                    fo.seek(0)
                    if nd:
                        x = []
                        for line in fo:
                            x.append(json.loads(line))
                    else:
                        x = json.load(fo)

            elif ext == '.orc':
                import pyarrow as pa
                x = pa.orc.read(fo, **kwargs)

            else:
                with open(fo, 'rb') as fo:
                    x = fo.read()

        return x

    @staticmethod
    def mode(op, ext):
        if op == 'write':
            m = 'w'
        else:
            m = 'r'

        if ext not in PureBeamPath.textual_extensions:
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

                if isinstance(x.index, pd.MultiIndex):
                    raise TypeError("MultiIndex not supported with feather extension.")

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
            elif ext == '.json':
                json.dump(x, fo, **kwargs)
            elif ext == '.ndjson':
                for xi in x:
                    json.dump(xi, fo, **kwargs)
                    fo.write("\n")
            elif ext == '.txt':
                fo.write(str(x))
            elif ext == '.npz':
                np.savez(fo, x, **kwargs)
            elif ext == '.scipy_npz':
                import scipy
                scipy.sparse.save_npz(fo, x, **kwargs)
                self.rename(f'{path}.npz', path)
            elif ext == '.parquet':
                x = pd.DataFrame(x)
                x.to_parquet(fo, **kwargs)
            elif ext == '.pt':
                import torch
                torch.save(x, fo, **kwargs)
            else:
                raise ValueError(f"Unsupported extension type: {ext} for file {x}.")

        return self

    def resolve(self, strict=False):
        return self


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


def rate_string_format(n, t):
    if n / t > 1:
        return f"{n / t: .4} [iter/sec]"
    return f"{t / n: .4} [sec/iter]"


def print_beam_hyperparameters(args, debug_only=False):
    from ..logger import beam_logger as logger

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
    from ..logger import beam_logger as logger

    if application == 'tensorboard':
        first_beam_range = 66
        first_global_range = 26006
    elif application == 'flask':
        first_beam_range = 50
        first_global_range = 25000
    elif application == 'ray':
        first_beam_range = 65
        first_global_range = 28265
    else:
        first_beam_range = 2
        first_global_range = 30000

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
    if has_modin and isinstance(x, mpd.base.BasePandasDataset):
        return 'modin'
    if has_scipy and scipy.sparse.issparse(x):
        return 'scipy_sparse'
    if isinstance(x, dict):
        return 'dict'
    if isinstance(x, list):
        return 'list'
    if isinstance(x, tuple):
        return 'tuple'
    if isinstance(x, set):
        return 'set'
    if isinstance(x, Path) or isinstance(x, PureBeamPath):
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
    elements type: array, int, float, complex, str, object, empty, none, unknown

    '''

    if np.isscalar(x) or (has_torch and torch.is_tensor(x) and (not len(x.shape))):
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
        if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, set):
            if not len(x):
                elt = 'empty'
            else:

                if len(x) < 20:
                    elts = [check_element_type(xi) for xi in x]

                else:
                    ind = np.random.randint(len(x), size=(20,))
                    elts = [check_element_type(x[i]) for i in ind]

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


def normalize_host(hostname, port=None, default='localhost'):
    if hostname is None:
        hostname = default
    if port is None:
        host = f"{hostname}"
    else:
        host = f"{hostname}:{port}"

    return host


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
