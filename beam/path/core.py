import fnmatch
import io
import json
import os
from collections import namedtuple
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import PurePosixPath, PureWindowsPath, Path
import pandas as pd
import numpy as np
import re

from ..type import check_type, Types
from ..type.utils import is_beam_data, is_beam_processor, is_pil
from ..base import BeamResource, BeamURL, base_paths

BeamFile = namedtuple('BeamFile', ['data', 'timestamp'])
targets = {'pl': 'polars', 'pd': 'pandas', 'cf': 'cudf', 'pa': 'pyarrow',
           'polars': 'polars', 'pandas': 'pandas', 'cudf': 'cudf', 'pyarrow': 'pyarrow',
           'native': 'native'}

# list all extensions by theirs a priori probability
prioritized_extensions = ['.pkl', '.parquet', '.csv', '.fea', '.json', '.ndjson', '.pt', '.orc', '.txt', '.bin', '.npz',
                          '.pickle', '.dill', '.npy', 'yaml',
                          '.scipy_npz', '.flac', '.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt',
                          '.avro', '.adjlist', '.gexf', '.gml', '.pajek', '.graphml', '.ini', '.h5', '.hdf5', '.yaml',
                          '.yml', '.xml', '.mat', '.zip', '.msgpack', '.cloudpickle', '.geojson', '.wav', '.joblib',
                          '.z', '.gz', '.bz2', '.xz', '.lzma', '.safetensors', '.png', '.jpg', '.jpeg', '.gif',
                          '.bmp', '.tiff', '.tif', '.webp']


def normalize_host(hostname, port=None, default='localhost'):
    if hostname is None:
        hostname = default
    if port is None:
        host = f"{hostname}"
    else:
        host = f"{hostname}:{port}"

    return host


def strip_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def get_target(path, deep=1, target=None):
    if target is not None:
        return targets[target]
    ext = path.suffix
    if ext is None:
        return None
    if ext in targets:
        return targets[ext]
    if deep == 0:
        return None
    return get_target(path.parent.joinpath(path.stem), deep-1)


class PureBeamPath(BeamResource):
    feather_index_mark = "feather_index:"

    # all the extensions that are considered textual (should be read as .txt)
    textual_extensions = ['.txt', '.text', '.py', '.sh', '.c', '.cpp', '.h', '.hpp', '.java', '.js', '.css',
                          '.html', '.md', '.log']
    text_based_extensions = textual_extensions + ['.json', '.orc', '.yaml', '.yml', '.ndjson', '.csv', '.ini', '.jsonl']

    def __init__(self, *pathsegments, scheme=None, client=None, **kwargs):
        if len(pathsegments) == 1 and isinstance(pathsegments[0], PureBeamPath):
            pathsegments = pathsegments[0].parts

        if scheme == 'nt':
            self.path = PureWindowsPath(*pathsegments)
        else:
            self.path = PurePosixPath(*pathsegments)

        super().__init__(resource_type='storage', scheme=scheme, path=str(self.path), **kwargs)


        self.mode = "rb"
        self.file_object = None
        self.close_fo_after_read = None
        self._client = client
        self.open_kwargs = dict(mode="rb", buffering=- 1, encoding=None, errors=None,
                                newline=None, closefd=True, opener=None)

    @property
    def client(self):
        return self._client

    @client.setter
    def client(self, value):
        self._client = value

    @property
    def str(self):
        return str(self.path)

    @property
    def list(self):
        return list(self)

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
        if type(name) is int:
            return list(self)[name]
        return self.joinpath(name)

    def __setitem__(self, key, value):
        p = self.joinpath(key)
        p.write(value)

    def touch(self, mode=0o666, exist_ok=True):
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
            if '://' in dst:
                from .resource import beam_path
                dst = beam_path(dst)
            else:
                dst = self.gen(dst)

        if self.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            for p in self.iterdir():
                p.copy(dst.joinpath(p.name), ignore=ignore, include=include)

        elif self.is_file() and dst.is_dir():
            dst = dst.joinpath(self.name)
            self.copy(dst, ignore=ignore, include=include)

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

    def getctime(self):
        return None

    def getatime(self):
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

    def __call__(self, *args, **kwargs):
        return self.open(*args, **kwargs)

    def open(self, mode="rb", buffering=- 1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        self.mode = mode
        self.open_kwargs = dict(mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline,
                                closefd=closefd, opener=opener)
        return self

    def close_at_exit(self):
        if self.close_fo_after_read or self.mode in ['wb', 'w']:
            self.close()

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

    def glob(self, pattern, case_sensitive=None):
        case_sensitive = case_sensitive or True
        return self._glob_recursive(self, pattern, case_sensitive=case_sensitive)

    def _glob_recursive(self, current_path, pattern, case_sensitive=True):

        if not case_sensitive:
            pattern = pattern.lower()

        for p in current_path.iterdir():

            # If item is a directory and pattern requires recursion
            if '**' in pattern and p.is_dir():
                # Correctly pass `item` as `current_path` for recursive exploration
                yield from self._glob_recursive(p, pattern, case_sensitive=case_sensitive)
            # Match the item's name against the pattern

            name = str(p.relative_to(self))
            if not case_sensitive:
                name = name.lower()

            if fnmatch.fnmatch(name, pattern):
                yield p

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

    def iter_content(self, ext=None, **kwargs):
        for p in self.iterdir():
            yield p.read(ext=ext, **kwargs)

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

    def __contains__(self, item):
        return self.joinpath(item).exists()

    def read(self, ext=None, target=None, **kwargs):

        """

        @param ext:
        @param kwargs:
        @return:

        Supports the following formats:
        - .fea: Feather
        - .csv: CSV
        - .pkl, .pickle: Pickle
        - .dill: Dill
        - .npy: Numpy
        - .json: JSON
        - .ndjson: Newline delimited JSON
        - .orc: ORC
        - .txt, '.text', '.py', '.sh', '.c', '.cpp', '.h', '.hpp', '.java', '.js', '.css',
                          '.html', '.md': Text
        - .npz: Numpy zip
        - .scipy_npz: Scipy sparse matrix
        - .flac: Soundfile
        - .parquet: Parquet
        - .pt: PyTorch
        - .xls, .xlsx, .xlsm, .xlsb, .odf, .ods, .odt: Excel
        - .avro: Avro
        - .adjlist, .gexf, .gml, .pajek, .graphml: NetworkX
        - .ini: ConfigParser
        - .h5, .hdf5: HDF5
        - .yaml, .yml: YAML
        - .xml: XML
        - .mat: MAT
        - .zip: ZIP
        - .msgpack: MessagePack
        - .cloudpickle: Cloudpickle
        - .geojson: GeoJSON
        - .wav: Soundfile
        - .joblib, .z, .gz, .bz2, .xz, .lzma: Joblib
        - .safetensors: SafeTensors
        - .bin: Bytes

        """

        self.close_fo_after_read = True
        if ext is None:
            ext = self.suffix
        ext = ext.lower()

        target = get_target(self, target=target)
        if target == 'pyarrow':
            if ext in ['.fea', '.feather']:
                import pyarrow.feather as pdu
            elif ext == '.orc':
                import pyarrow.orc as pdu
            elif ext == '.csv':
                import pyarrow.csv as pdu
            else:
                pdu = pd

        elif target == 'polars':
            import polars as pdu
        elif target == 'cudf':
            import cudf as pdu
        else:
            pdu = pd

        # read .bmd (beam-data) and .bmpr (beam-processor) files

        if ext == '.bmd':
            from ..data import BeamData
            lazy = kwargs.pop('lazy', False)
            return BeamData.from_path(self, lazy=lazy, **kwargs)

        if ext == '.bmpr':
            from ..processor import Processor
            return Processor.from_path(self, **kwargs)

        if ext == '.abm':
            from ..auto import AutoBeam
            return AutoBeam.from_bundle(self, **kwargs)

        with self(mode=PureBeamPath.mode('read', ext)) as fo:

            if ext in ['.fea', '.feather']:

                import pyarrow as pa
                # x = feather.read_feather(pa.BufferReader(fo.read()), **kwargs)
                x = pd.read_feather(fo, **kwargs)

                c = x.columns
                for ci in c:
                    if PureBeamPath.feather_index_mark in ci:
                        index_name = strip_prefix(ci, PureBeamPath.feather_index_mark)
                        x = x.rename(columns={ci: index_name})
                        x = x.set_index(index_name)
                        break

            elif ext == '.csv':
                x = pdu.read_csv(fo, **kwargs)
            elif ext in ['.pkl', '.pickle']:
                x = pd.read_pickle(fo, **kwargs)
            elif ext == '.dill':
                import dill
                x = dill.load(fo, **kwargs)
            elif ext in ['.npy', '.npz', '.npzc']:
                if ext in ['.npz', '.npzc']:
                    self.close_fo_after_read = False

                x = np.load(fo, allow_pickle=True, **kwargs)
            elif ext in PureBeamPath.textual_extensions:
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
                target = get_target(self, deep=1, target=target)
                if target == 'pyarrow':
                    import pyarrow.parquet as pq
                    x = pq.read_table(fo, **kwargs)
                else:
                    x = pdu.read_parquet(fo, **kwargs)
            elif ext == '.pt':
                import torch
                x = torch.load(fo, **kwargs)
            elif ext in ['.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt']:
                x = pdu.read_excel(fo, **kwargs)
            elif ext == '.avro':
                x = []
                import fastavro
                for record in fastavro.reader(fo):
                    x.append(record)
            elif ext in ['.adjlist', '.gexf', '.gml', '.pajek', '.graphml']:
                import networkx as nx
                read = getattr(nx, f'read_{ext[1:]}')
                x = read(fo, **kwargs)

            elif ext == '.ini':
                import configparser
                x = configparser.ConfigParser()
                x.read_file(fo)
                x = {section: dict(x.items(section)) for section in x.sections()}

            elif ext in ['.json', '.ndjson', '.jsonl']:

                nd = ext in ['.ndjson', '.jsonl']
                try:
                    if 'schema' in kwargs:
                        x = []
                        from fastavro import json_reader, parse_schema
                        schema = parse_schema(kwargs['schema'])
                        for record in json_reader(fo, schema=schema):
                            x.append(record)
                    if target == 'native' or nd:
                        x = json.load(fo, **kwargs)
                    elif target == 'pandas':
                        x = pd.read_json(fo, lines=nd, **kwargs)
                    elif target == 'polars':
                        import polars as pl
                        x = pl.read_json(fo, **kwargs)
                    elif target == 'cudf':
                        import cudf
                        x = cudf.read_json(fo, lines=nd, **kwargs)
                    elif target == 'pyarrow':
                        from pyarrow import json as pa
                        x = pa.read_json(fo, **kwargs)

                    else:
                        x = json.load(fo, **kwargs)
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

            elif ext == '.cloudpickle':
                import cloudpickle
                x = cloudpickle.load(fo)

            # GeoJSON (.geojson)
            elif ext == '.geojson':
                import geopandas as gpd
                x = gpd.read_file(fo)

            # WAV (.wav)
            elif ext == '.wav':
                from scipy.io.wavfile import read as wav_read
                x = wav_read(fo)

            elif ext in ['.joblib']:
                import joblib
                x = joblib.load(fo, **kwargs)

            elif ext in ['.z', '.gz']:
                import gzip
                with gzip.open(fo, 'rb') as file:
                    x = file.read()
                    x = self.read_inner_content(x, **kwargs)

            elif ext in ['.bz2']:
                import bz2
                with bz2.open(fo, 'rb') as file:
                    x = file.read()
                    x = self.read_inner_content(x, **kwargs)

            elif ext in ['.xz', '.lzma']:
                import lzma
                with lzma.open(fo, 'rb') as file:
                    x = file.read()
                    x = self.read_inner_content(x, **kwargs)

            elif ext == '.safetensors':
                from safetensors.torch import load
                x = load(fo.read())

            elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp']:
                if target == 'cv2':
                    import cv2
                    x = cv2.imread(fo, **kwargs)
                elif target in ['PIL', 'pillow'] or target is None:

                    x = fo.read()
                    x = io.BytesIO(x)
                    from PIL import Image
                    x = Image.open(x, **kwargs)
                else:
                    raise ValueError(f"Unknown target: {target}")
            else:
                x = fo.read()

        return x

    def read_inner_content(self, content, inner_ext=None, **kwargs):

        p = self.stem
        if '.' in p:
            inner_ext = inner_ext or f".{p.split('.')[-1]}"

        if inner_ext is not None:
            content = IOPath('/', data=content).read(ext=inner_ext, **kwargs)

        return content

    def readlines(self):
        return self.read(readlines=True, ext='.txt')

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

        if ext not in PureBeamPath.text_based_extensions:
            m = f"{m}b"

        return m

    def write_text(self, x, encoding=None, errors=None, newline=None):
        return self.write(x, ext='.txt')

    def write_bytes(self, x):
        return self.write(x, ext='.bin')

    def write(self, *args, ext=None, **kwargs):

        x = None
        if len(args) >= 1:
            x = args[0]

        if ext is None:
            ext = self.suffix
        ext = ext.lower()

        x_type = check_type(x)

        # write .bmd (beam-data) and .bmpr (beam-processor) files
        if ext == '.bmd':
            if is_beam_data(x):
                x.to_path(self)
            else:
                from ..data import BeamData
                BeamData(x).to_path(self)
            return self

        if ext == '.bmpr':
            assert is_beam_processor(x), f"Expected Processor, got {type(x)}"
            x.to_path(self, **kwargs)
            return self

        if ext == '.abm':
            from ..auto import AutoBeam
            AutoBeam.to_bundle(x, self)
            return self

        with self(mode=PureBeamPath.mode('write', ext)) as fo:

            if ext in ['.fea', '.feather']:

                if x_type.minor == Types.polars:
                    import polars as pl
                    x.to_feather(fo, **kwargs)
                elif x_type.minor == Types.cudf:
                    import cudf
                    x.to_feather(fo, **kwargs)
                else:
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

                if x_type.minor == Types.polars:
                    x.write_csv(fo, **kwargs)
                elif x_type.minor == Types.cudf:
                    x.to_csv(fo, **kwargs)
                else:
                    x = pd.DataFrame(x)
                    x.to_csv(fo, **kwargs)

            elif ext == '.avro':
                import fastavro
                fastavro.writer(fo, x, **kwargs)
            elif ext in ['.pkl', '.pickle']:
                pd.to_pickle(x, fo, **kwargs)
            elif ext == '.dill':
                import dill
                dill.dump(x, fo, **kwargs)
            elif ext == '.npy':
                np.save(fo, x, **kwargs)
            elif ext == '.json':
                if 'schema' in kwargs:
                    from fastavro import json_writer, parse_schema
                    schema = parse_schema(kwargs['schema'])
                    json_writer(fo, schema, x)
                else:
                    json.dump(x, fo, **kwargs)
            elif ext in ['.ndjson', '.jsonl']:
                if 'schema' in kwargs:
                    from fastavro import json_writer, parse_schema
                    schema = parse_schema(kwargs['schema'])
                    json_writer(fo, schema, x)
                else:
                    for xi in x:
                        json.dump(xi, fo, **kwargs)
                        fo.write("\n")
            elif ext == '.txt':
                fo.write(str(x))
            elif ext == '.npz':
                np.savez(fo, *args, **kwargs)
            elif ext == '.npzc':
                np.savez_compressed(fo, *args, **kwargs)
            elif ext in ['.adjlist', '.gexf', '.gml', '.pajek', '.graphml']:
                import networkx as nx
                write = getattr(nx, f'write_{ext[1:]}')
                write(x, fo, **kwargs)
            elif ext == '.scipy_npz':
                import scipy
                scipy.sparse.save_npz(fo, x, **kwargs)
                # self.rename(f'{path}.npz', path)
            elif ext == '.parquet':
                if x_type.minor == Types.polars:
                    x.write_parquet(fo, **kwargs)
                elif x_type.minor == Types.cudf:
                    x.to_parquet(fo, **kwargs)
                else:
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

            elif ext == '.cloudpickle':
                import cloudpickle
                cloudpickle.dump(x, fo)

            # GeoJSON (.geojson)
            elif ext == '.geojson':
                import geopandas as gpd
                gpd.GeoDataFrame(x).to_file(fo, driver='GeoJSON')

            # WAVt (.wav)
            elif ext == '.wav':
                from scipy.io.wavfile import write as wav_write
                wav_write(fo, *x)

            elif '.joblib' in ext:
                import joblib

                compress_methods = ext.split('_')[1:]
                if len(compress_methods) > 0:
                    cm = compress_methods[0]
                    compress_methods = {'z': 'zlib', 'gz': 'gzip', 'bz2': 'bz2', 'xz': 'xz', 'lzma': 'lzma'}
                    cm = compress_methods[cm]

                    if 'compress' not in kwargs:
                        kwargs['compress'] = cm
                    else:
                        kwargs['compress'] = (cm, kwargs['compress'])

                joblib.dump(x, fo, **kwargs)

            elif ext in ['.z', '.gz']:
                import gzip
                x = self.serialize_inner_content(x, **kwargs)
                with gzip.open(fo, 'wb') as file:
                    file.write(x)

            elif ext in ['.bz2']:
                import bz2
                x = self.serialize_inner_content(x, **kwargs)
                with bz2.open(fo, 'wb') as file:
                    file.write(x)

            elif ext in ['.xz', '.lzma']:
                import lzma
                x = self.serialize_inner_content(x, **kwargs)
                with lzma.open(fo, 'wb') as file:
                    file.write(x)

            elif ext == '.safetensors':
                from safetensors.torch import save
                raw_data = save(x, **kwargs)
                fo.write(raw_data)

            elif ext == '.ini':
                import configparser
                x = configparser.ConfigParser()
                x.read_dict(x)
                x.write(fo)

            elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp']:
                if is_pil(x):
                    from PIL import Image
                    x.save(fo, **kwargs)
                else:
                    import cv2
                    cv2.imwrite(fo, x, **kwargs)
            elif ext in PureBeamPath.textual_extensions:
                assert isinstance(x, str), f"Expected str, got {type(x)}"
                fo.write(x)

            elif ext == '.xlsx':
                if x_type.minor == Types.polars:
                    x.write_excel(fo, **kwargs)
                elif x_type.minor == Types.cudf:
                    x.to_excel(fo, **kwargs)
                else:
                    x = pd.DataFrame(x)
                    x.to_excel(fo, **kwargs)

            elif ext == '.bin':
                assert isinstance(x, bytes), f"Expected bytes, got {type(x)}"
                fo.write(x)

            else:
                raise ValueError(f"Unsupported extension type: {ext} for file {x}.\n"
                                 f"Use write_bytes or write_text instead,  \n"
                                 f"or use one of the existing extensions by explicitly setting the ext argument,\n"
                                 f"e.g. path.write(content, ext=.pkl).")

        return self

    def serialize_inner_content(self, content, inner_ext=None, **kwargs):

        p = self.stem
        if '.' in p:
            inner_ext = inner_ext or f".{p.split('.')[-1]}"
        inner_ext = inner_ext or '.pkl'
        io_path = IOPath('/').write(content, ext=inner_ext, **kwargs)
        return io_path.data['/']

    def resolve(self, strict=False):
        return self


class DictBasedPath(PureBeamPath):
    def __init__(self, *pathsegments, scheme=None, client=None, data=None, **kwargs):
        super().__init__(*pathsegments, scheme=scheme, **kwargs)
        if client is None:
            client = {}
        if data is not None:
            client = data
        self.client = client

    @property
    def data(self):
        return self.get_data()

    def get_data(self):
        return self.client

    def set_data(self, data):
        self.client = data

    def is_file(self):
        raise NotImplementedError

    def is_dir(self):
        client = self.client
        if len(self.parts) == 1:
            return True
        for p in self.parts[1:]:
            if not isinstance(client, dict):
                return False
            if p not in client:
                return False
            client = client[p]
        return isinstance(client, dict)

    def exists(self):
        client = self.client
        if len(self.parts) == 1:
            return True
        for p in self.parts[1:]:
            if p not in client:
                return False
        return True

    @property
    def _obj(self):
        client = self.client
        for p in self.parts[1:]:
            client = client[p]
        return client

    @property
    def _parent(self):
        client = self.client
        for p in self.parts[1:-1]:
            client = client[p]
        return client

    def iterdir(self):
        client = self._obj
        for p in client:
            yield self.gen(self.path.joinpath(p))

    def mkdir(self, *args, parents=True, exist_ok=True, **kwargs):

        if self.is_root():
            return

        if not exist_ok:
            if self.exists():
                raise FileExistsError

        if not parents:
            if self._parent is None:
                raise FileNotFoundError

        client = self.client
        for p in self.parts[1:]:
            if p not in client:
                client[p] = {}
            elif p in client and not isinstance(client[p], dict):
                raise NotADirectoryError

    def rmdir(self):
        client = self._parent
        del client[self.parts[-1]]

    def unlink(self, missing_ok=False):

        if self.is_root():
            raise ValueError("Cannot delete root")

        client = self._parent
        try:
            del client[self.parts[-1]]
        except KeyError as e:
            if not missing_ok:
                raise e

    def rename(self, target):

        if target.is_root():
            raise ValueError("Cannot rename to root")

        if type(target) is str:
            target = self.gen(target)
        target_parent = target._parent
        target_parent[target.parts[-1]] = self._obj
        self.unlink()

    def replace(self, target):
        self.rename(target)

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError


class IOPath(DictBasedPath):

    # this class represents a in memory file system in the form of dictionaries of dictionaries of bytes/strings objects
    # like other PureBeamPath implementations, it is a pathlib/beam_path api

    def __init__(self, *pathsegments, client=None, data=None, **kwargs):
        super().__init__(*pathsegments, scheme='io', client=client, data=data, **kwargs)

    def is_file(self):
        client = self.client
        if len(self.parts) == 1:
            return False
        for p in self.parts[1:]:
            if not isinstance(client, dict):
                return False
            if p not in client:
                return False
            client = client[p]
        return isinstance(client, (bytes, str))

    def __enter__(self):
        if self.mode in ["rb", "r"]:
            self.file_object = BytesIO(self._obj)
        elif self.mode in ['wb', 'w']:
            self.file_object = BytesIO() if 'b' in self.mode else StringIO(newline=self.open_kwargs['newline'])
        else:
            raise ValueError

        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.mode in ["wb", "w"]:
            parent = self._parent
            name = self.parts[-1]
            self.file_object.seek(0)
            content = self.file_object.getvalue()
            parent[name] = content

        self.close_at_exit()


class DictPath(DictBasedPath):

    def __init__(self, *pathsegments, client=None, data=None, **kwargs):
        super().__init__(*pathsegments, scheme='dict', client=client, data=data, **kwargs)

    def is_file(self):
        client = self.client
        if len(self.parts) == 1:
            return False
        for p in self.parts[1:]:
            if not isinstance(client, dict):
                return False
            if p not in client:
                return False
            client = client[p]
        return isinstance(client, BeamFile)

    def write(self, *args, ext=None, **kwargs):
        assert len(args) == 1, "DictPath.write takes exactly one argument"
        x = args[0]
        self._parent[self.parts[-1]] = BeamFile(x, timestamp=datetime.now())

    def read(self, ext=None, **kwargs):
        return self._obj.data


class BeamKey:

    def __init__(self, config_path=None, **kwargs):
        self.keys = {}

        self._config_path = config_path
        if self._config_path is None:
            self._config_path = Path(base_paths.global_config)

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
            if 'config_file' in self.hparams and self.hparams['config_file'] is not None:
                self._config_path = Path(self.hparams['config_file'])
            else:
                self._config_path = Path(base_paths.global_config)
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
