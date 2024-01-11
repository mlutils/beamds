import json
import os
from pathlib import PurePosixPath, PureWindowsPath, Path
from urllib.parse import urlparse, urlunparse, parse_qsl, ParseResult
import pandas as pd
import numpy as np
import re


class PureBeamPath:
    feather_index_mark = "feather_index:"

    text_extensions = ['.txt', '.text', '.py', '.sh', '.c', '.cpp', '.h', '.hpp', '.java', '.js', '.css', '.html',
                       '.yaml', '.yml']
    textual_extensions = text_extensions + ['.json', '.orc', '.ndjson']

    def __init__(self, *pathsegments, url=None, scheme=None, hostname=None, port=None, username=None, password=None,
                 fragment=None, params=None, client=None, **kwargs):
        super().__init__()

        if len(pathsegments) == 1 and isinstance(pathsegments[0], PureBeamPath):
            pathsegments = pathsegments[0].parts

        if scheme == 'windows':
            self.path = PureWindowsPath(*pathsegments)
        else:
            self.path = PurePosixPath(*pathsegments)

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

    @property
    def str(self):
        return str(self.path)

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

    def touch(self):
        self.write(b'', ext='.bin')

    def not_empty(self, filter_pattern=None):

        if self.is_dir():
            for p in self.iterdir():
                if p.not_empty():
                    return True
                if p.is_file():
                    if filter_pattern is not None:
                        if not re.match(filter_pattern, p.name):
                            return True
                    else:
                        return True
        return False

    def copy(self, dst, ignore=None, include=None):

        if type(dst) is str:
            dst = self.gen(dst)

        if self.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            for p in self.iterdir():
                p.copy(dst.joinpath(p.name), ignore=ignore, include=include)
        else:
            dst.parent.mkdir()
            ext = self.suffix
            if ignore is not None:
                if type(ignore) is str:
                    ignore = [ignore]
                if ext in ignore:
                    return
            if include is not None:
                if type(include) is str:
                    include = [include]
                if ext not in include:
                    return
            with self.open("rb") as f:
                with dst.open("wb") as g:
                    g.write(f.read())

    def rmtree(self, ignore=None, include=None):
        if self.is_file():
            self.unlink()
        elif self.is_dir():
            delete_dir = True
            for item in self.iterdir():
                if item.is_dir():
                    item.rmtree(ignore=ignore, include=include)
                else:
                    ext = item.suffix
                    if ignore is not None:
                        if type(ignore) is str:
                            ignore = [ignore]
                        if ext in ignore:
                            delete_dir = False
                            continue
                    if include is not None:
                        if type(include) is str:
                            include = [include]
                        if ext not in include:
                            delete_dir = False
                            continue
                    item.unlink()
            if delete_dir:
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

    def clean(self, ignore=None, include=None):

        if self.exists():
            self.rmtree(ignore=ignore, include=include)
        else:
            if self.parent.exists():
                for p in self.parent.iterdir():
                    if p.stem == self.name:
                        p.rmtree(ignore=ignore, include=include)

        self.mkdir(parents=True)
        self.rmdir()

    @property
    def is_local(self):
        return self.url.scheme == 'file'

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
        return str(self)
        # raise TypeError("For BeamPath (named bp), use bp.open(mode) instead of open(bp, mode)")

    def __call__(self, mode="rb"):
        self.mode = mode
        return self

    def open(self, mode="rb", buffering=- 1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
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

    def __eq__(self, other):

        if type(self) != type(other):
            return False
        p = self.resolve()
        o = other.resolve()

        return p.as_uri() == o.as_uri()

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

    def glob(self, *args, **kwargs):
        for path in self.path.glob(*args, **kwargs):
            yield self.gen(path)

    def rglob(self, *args, **kwargs):
        for path in self.path.rglob(*args, **kwargs):
            yield self.gen(path)

    def absolute(self):
        path = self.path.absolute()
        return self.gen(path)

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

    def rename(self, target):
        return NotImplementedError

    def replace(self, target):
        return NotImplementedError

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
                import fastavro
                for record in fastavro.reader(fo):
                    x.append(record)
            elif ext in ['.adjlist', '.gexf', '.gml', '.pajek', '.graphml']:
                import networkx as nx
                read = getattr(nx, f'read_{ext[1:]}')
                x = read(fo, **kwargs)

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

            # HDF5 (.h5, .hdf5)
            elif ext in ['.h5', '.hdf5']:
                import h5py
                with h5py.File(fo, 'r') as f:
                    x = {key: f[key][...] for key in f.keys()}

            # YAML (.yaml, .yml)
            elif ext in ['.yaml', '.yml']:
                import yaml
                x = yaml.safe_load(fo)

            # XML (.xml)
            elif ext == '.xml':
                import xml.etree.ElementTree as ET
                x = ET.parse(fo).getroot()

            # MAT (.mat)
            elif ext == '.mat':
                from scipy.io import loadmat
                x = loadmat(fo)

            # ZIP (.zip)
            elif ext == '.zip':
                import zipfile
                with zipfile.ZipFile(fo, 'r') as zip_ref:
                    x = {name: zip_ref.read(name) for name in zip_ref.namelist()}

            # MessagePack (.msgpack)
            elif ext == '.msgpack':
                import msgpack
                x = msgpack.unpackb(fo.read(), raw=False)

            # GeoJSON (.geojson)
            elif ext == '.geojson':
                import geopandas as gpd
                x = gpd.read_file(fo)

            # WAV (.wav)
            elif ext == '.wav':
                from scipy.io.wavfile import read as wav_read
                x = wav_read(fo)

            elif ext in ['.joblib', '.z', '.gz', '.bz2', '.xz', '.lzma']:
                import joblib
                x = joblib.load(fo.read(), **kwargs)

            elif ext == '.safetensors':
                from safetensors.torch import load
                x = load(fo.read())

            else:
                x = fo.read()

        return x

    def read_text(self):
        return self.read(ext='.txt')

    def read_bytes(self):
        return self.read(ext='.bin')

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
            elif ext == '.avro':
                import fastavro
                fastavro.writer(fo, x, **kwargs)
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
            elif ext in ['.adjlist', '.gexf', '.gml', '.pajek', '.graphml']:
                import networkx as nx
                write = getattr(nx, f'write_{ext[1:]}')
                write(x, fo, **kwargs)
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

            # HDF5 (.h5, .hdf5)
            elif ext in ['.h5', '.hdf5']:
                import h5py
                with h5py.File(fo, 'w') as f:
                    for key, value in x.items():
                        f.create_dataset(key, data=value)

            # YAML (.yaml, .yml)
            elif ext in ['.yaml', '.yml']:
                import yaml
                yaml.safe_dump(x, fo)

            # XML (.xml)
            elif ext == '.xml':
                import xml.etree.ElementTree as ET

                tree = ET.ElementTree(x)
                tree.write(fo)

            # MAT (.mat)
            elif ext == '.mat':
                from scipy.io import savemat
                savemat(fo, x)

            # ZIP (.zip)
            elif ext == '.zip':
                import zipfile

                with zipfile.ZipFile(fo, 'w') as zip_ref:
                    for name, content in x.items():
                        zip_ref.writestr(name, content)

            # MessagePack (.msgpack)
            elif ext == '.msgpack':
                import msgpack
                fo.write(msgpack.packb(x, use_bin_type=True))

            # GeoJSON (.geojson)
            elif ext == '.geojson':
                import geopandas as gpd
                gpd.GeoDataFrame(x).to_file(fo, driver='GeoJSON')

            # WAVt (.wav)
            elif ext == '.wav':
                from scipy.io.wavfile import write as wav_write
                wav_write(fo, *x)

            elif ext in ['.joblib', '.z', '.gz', '.bz2', '.xz', '.lzma']:
                import joblib
                if ext != '.joblib':
                    compress_methods = {'.z': 'zlib', '.gz': 'gzip', '.bz2': 'bz2', '.xz': 'xz', '.lzma': 'lzma'}
                    compress_method = compress_methods[ext]
                    if 'compress' not in kwargs:
                        kwargs['compress'] = compress_method
                    else:
                        kwargs['compress'] = (compress_method, kwargs['compress'])
                joblib.dump(x, fo, **kwargs)

            elif ext == '.safetensors':
                from safetensors.torch import save
                raw_data = save(x, **kwargs)
                fo.write(raw_data)

            else:
                raise ValueError(f"Unsupported extension type: {ext} for file {x}.")

        return self

    def resolve(self, strict=False):
        return self.gen(self.path.resolve(strict=strict))


class BeamKey:

    def __init__(self, config_path=None, **kwargs):
        self.keys = {}

        self._config_path = config_path
        if self._config_path is None:
            self._config_path = Path.home().joinpath('conf.pkl')

        self._config_file = None
        self.hparams = kwargs

    def set_hparams(self, hparams):

        for k, v in hparams.items():
            self.hparams[k] = v
        # clear config file
        self._config_path = None

    @property
    def config_path(self):
        if self._config_path is None:
            if 'config_file' in self.hparams:
                self._config_path = Path(self.hparams['config_file'])
            else:
                self._config_path = Path.home().joinpath('conf.pkl')
        return self._config_path

    @property
    def config_file(self):
        if self._config_file is None:
            if self.config_path is not None and self.config_path.is_file():
                self._config_file = pd.read_pickle(self.config_path)
        return self._config_file

    def store(self, name=None, value=None, store_to_env=True):
        if name is not None:
            self.keys[name] = value

        # store to environment
        if store_to_env:
            os.environ[name] = str(value)

        config_file = self.config_file
        if config_file is None:
            config_file = {}

        for k, v in self.keys.items():
            config_file[k] = v

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(config_file, self.config_path)
        self._config_file = config_file

    def __setitem__(self, key, value):
        self(key, value)

    def __getitem__(self, item):
        return self(item)

    def __call__(self, name, value=None, store=True):

        if value is not None:
            self.keys[name] = value
            if store:
                self.store(name, value)
            return value
        elif name in self.keys:
            value = self.keys[name]
        elif name in self.hparams and self.hparams[name] is not None:
            value = self.hparams[name]
            self.keys[name] = value
        elif name in os.environ:
            value = os.environ[name]
            self.keys[name] = value
        elif self.config_file is not None and name in self.config_file:
            value = self.config_file[name]
            self.keys[name] = value
        else:
            ValueError(f"Cannot find key: {name} in BeamKey")

        return value


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
        return self.parsed_url.username

    @property
    def hostname(self):
        return self.parsed_url.hostname

    @property
    def password(self):
        return self.parsed_url.password

    @property
    def port(self):
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
        return PurePosixPath(path).as_posix()

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


def normalize_host(hostname, port=None, default='localhost'):
    if hostname is None:
        hostname = default
    if port is None:
        host = f"{hostname}"
    else:
        host = f"{hostname}:{port}"

    return host