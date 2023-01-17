import itertools
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from .utils import check_type, slice_to_index, as_tensor, to_device, recursive_batch, as_numpy, beam_device, \
    recursive_device, recursive_len, recursive
import pandas as pd
import math
import hashlib
import sys
import warnings
import argparse
from collections import namedtuple
from .utils import divide_chunks, collate_chunks, recursive_chunks, iter_container, logger, \
    recursive_size_summary, recursive_len, is_arange, listdir_fullpath, is_chunk, rmtree, \
    recursive_size, recursive_flatten, recursive_collate_chunks, recursive_keys
from .parallel import parallelize
from collections import OrderedDict
import os
import fastavro
import pyarrow as pa
import pathlib
from argparse import Namespace
import scipy
import sys
from pathlib import Path
from functools import partial


DataBatch = namedtuple("DataBatch", "index label data")


class BeamData(object):

    feather_index_mark = "index:"
    configuration_file_name = '.conf'
    info_file_name = '.info'
    default_data_file_name = 'data'
    chunk_file_extension = '_chunk'
    partition_directory_name = '_part'

    def __init__(self, *args, data=None, path=None, name=None,
                 index=None, label=None, columns=None, lazy=True, device=None, target_device=None,
                 override=True, compress=None, chunk_strategy='files', chunksize=int(1e9), chunklen=None, n_chunks=None,
                 partition=None, archive_len=int(1e6), orient='columns', read_kwargs=None, write_kwargs=None,
                 **kwargs):

        '''

        @param args:
        @param data:
        @param path: if not str, requires to support the pathlib Path attributes and operations, can be container of paths
        @param lazy:
        @param kwargs:

        Possible orientations are: row/column/other

        There are 4 possible ways of data orientation:

        1. simple: simple representation of tabular data where there is only a single data array in self.data.
        This orientation should support the fastest getitem operations

        2. columns: in this orientation, each data element represents different set of columns or information about the
        same data so each data element has the same length and each row in each data element corresponds to the same object.

        3. index:  in this orientation the rows are spread over different data elements so the data elements may have
         different length but their shape[1:] is identical so we can collect batch of elements and concat them together.

         4. none: each data element represents different set of data points but each data point may have different nature.
         this could model for example node properties in Knowledge graph where there are many types of nodes with different
         features.

         If data is both cached in self.data and stored in self.all_paths, the cached version is always preferred.

        '''

        assert len(list(filter(lambda x: x is not None, [data, path]))), \
            "Requires ether data, root_path or all_paths to be not None"

        self.lazy = lazy
        self.override = override
        self.compress = compress
        self.chunksize = chunksize
        self.chunklen = chunklen
        self.n_chunks = n_chunks
        self.partition = partition
        self.archive_len = archive_len
        self.target_device = target_device
        self.index = index
        self.label = label
        self.columns = columns
        self.orient = orient
        self.chunk_strategy = chunk_strategy

        self.stored = False
        self.cached = True
        self.indices = None

        self._columns_map = None
        self._device = None
        self._len = None
        self._orientation = None
        self._data_types = None
        self._data_type = None
        self._objects_type = None
        self._flatten_data = None
        self._info = None
        self._conf = None

        self.read_kwargs = {} if read_kwargs is None else read_kwargs
        self.write_kwargs = {} if write_kwargs is None else write_kwargs

        # first we check if the BeamData object is cached (i.e. the data is passed as an argument)
        if len(args) == 1:
            self.data = args[0]
        elif len(args):
            self.data = list(args)
        elif len(kwargs):
            self.data = kwargs
        elif data is not None:
            self.data = data
        else:
            self.data = None
            self.cached = False
            # in this case the data is not cached, so it should be stored ether in root_path or in all_paths

        if device is not None:
            self.as_tensor(device=device)

        if type(path) is str:
            path = Path(path)

        path_type = check_type(path)
        if path_type.major != 'container' and name is not None:
            path = path.joinpath(name)

        if path_type.major == 'container':
            self.root_path = BeamData.recursive_root_finder(path)
            self.all_paths = path
        elif path is not None:
            self.root_path = path
            self.all_paths = BeamData.recursive_map_path(path)
        else:
            self.root_path = None
            self.all_paths = None

        if self.all_paths is not None and not self.cached:
            self.stored = True
            if not lazy:
                self.cache()

    @property
    def objects_type(self):

        if self._objects_type is not None:
            return self._objects_type

        objects_types = recursive_flatten(self.data_types)
        objects_types = [v.minor for v in objects_types if v.minor != 'none']
        self._objects_type = pd.Series(objects_types).value_counts().values[0]
        return self._objects_type

    @property
    def conf(self):

        if self._conf is not None:
            return self._conf

        if self.stored:
            conf_path = self.root_path.joinpath(BeamData.configuration_file_name)
            if conf_path.is_file():
                self._conf = BeamData.read_file(conf_path)
                return self._conf

        if self.cached:
            self._conf = {'orientation': self.orientation,
                          'objects_type': self.objects_type,
                          'len': len(self), 'columns_map': self.columns_map}
            return self._conf

        self._conf = None
        return self._conf

    @property
    def info(self):

        if self._info is not None:
            return self._info

        if self.stored:
            info_path = self.root_path.joinpath(BeamData.info_file_name)
            if info_path.is_file():
                self._info = BeamData.read_file(info_path)
                return self._info

        if self.cached:

            if self.orientation in ['index', 'other']:
                fold_index = np.concatenate([np.arange(len(d)) for d in self.flatten_data])
                fold = np.concatenate([i * np.ones(len(d), dtype=np.int64) for i, d in enumerate(self.flatten_data)])
                lengths = np.array([len(d) for d in self.flatten_data])
                offset = np.cumsum(lengths, dim=0) - lengths
                offset = offset[fold] + fold_index

            else:
                fold_index = None
                fold = None
                offset = None

            index = np.concatenate(recursive_flatten([self.index]))
            info = {'fold': fold, 'fold_index': fold_index, 'offset': offset, 'map': np.arange(len(index))}
            self._info = pd.DataFrame(info, index=index)
            return self._info

        self._info = None
        return self._info

    @staticmethod
    def recursive_filter(x, info):

        def _recursive_filter(x, info, key=0):
            x_type = check_type(x)
            if x_type.major == 'container':

                keys = []
                values = []
                index = []

                for k, v in iter_container(x):
                    keys.append(k)
                    i, v, key = _recursive_filter(v, info, key=key)
                    values.append(v)
                    index.append(i)

                if not is_arange(keys):
                    values = dict(zip(keys, values))
                    index = dict(zip(keys, index))

                return index, values, key

            else:

                in_fold = info['fold_index'][info['fold'] == key]
                x_type = check_type(x)

                if x is None:
                    return None, None, key + 1
                elif x_type.minor == 'pandas':
                    return in_fold.index, x.iloc[in_fold], key + 1
                else:
                    return in_fold.index, x[in_fold], key + 1

        index, data, _ = _recursive_filter(x, info)
        return DataBatch(index=index,data=data, label=None)

    @property
    def index_mapper(self):

        info = self.info
        if 'map' in info.columns:
            return info['map']

        return None

    @staticmethod
    def clean_path(path):

        if path.exists():
            rmtree(path)
        else:
            if path.parent.exists():
                for p in path.parent.iterdir():
                    if p.stem == path.name:
                        rmtree(p)

        path.mkdir(parents=True)
        path.rmdir()

    @property
    def flatten_data(self):
        if self._flatten_data is not None:
            return self._flatten_data
        self._flatten_data = recursive_flatten(self.data)
        return self._flatten_data

    @property
    def device(self):
        if self._device is not None:
            return self._device

        self._device = recursive_device(self.data)
        return self._device

    def to(self, device):
        self.data = recursive(lambda x: x.to(device))(self.data)
        self._device = device
        return self

    def __len__(self):

        if self._len is not None:
            return self._len

        if self.stored and self.conf is not None:
            self._len = self.conf['len']
            return self._len

        if self.cached:
            if self.orientation == 'columns':
                self._len = recursive_len(self.data)
            else:
                self._len = sum(recursive_flatten(recursive(lambda x: len(x) if hasattr(x, '__len__') else 0)(self.data)))
            return self._len

        self._len = None
        return self._len

    @property
    def orientation(self):
        if self._orientation is not None:
            return self._orientation
        if self.cached:

            data_type = check_type(self.data)

            if data_type.major != 'container':
                self._orientation = 'simple'
                if hasattr(self.data, 'columns') and self.columns is None:
                    self.columns = self.data.columns
                if hasattr(self.data, 'index') and self.index is None:
                    self.index = self.data.index

            else:

                if self.orient == 'columns':
                    lens = recursive_flatten(recursive(lambda x: len(x) if hasattr(x, '__len__') else None)([self.data]))
                    lens = list(filter(lambda x: x is not None, lens))

                    lens_index = recursive_flatten(
                        recursive(lambda x: len(x) if hasattr(x, '__len__') else None)([self.index]))
                    lens_index = list(filter(lambda x: x is not None, lens_index))

                    if len(np.unique(lens) == 1) and sum(lens) > sum(lens_index):
                        self._orientation = 'columns'
                        return self._orientation

                lens = recursive_flatten(
                    recursive(lambda x: x.shape[1] if hasattr(x, 'shape') and len(x.shape) > 1 else None)([self.data]))

                lens = list(filter(lambda x: x is not None, lens))
                if len(np.unique(lens) == 1):
                    self._orientation = 'index'
                else:
                    self._orientation = 'other'

        elif self.stored:
            self._orientation = self.conf['orientation']

        else:
            self._orientation = 'other'

        return self._orientation

    def set_property(self, p):
        setattr(self, f"_{p}", None)
        return getattr(self, p)

    @property
    def dim(self):
        if self.orientation == 'columns':
            return 0
        if self.orientation == 'index':
            return 1
        return None

    @property
    def data_types(self):
        if self._data_types is not None:
            return self._data_types
        self._data_types = recursive(check_type)(self.data)
        return self._data_types

    @property
    def data_type(self):
        if self._data_type is not None:
            return self._data_type
        self._data_type = check_type(self.data)
        return self._data_type

    @staticmethod
    def read_file(path, **kwargs):

        if type(path) is str:
            path = Path(path)

        path_type = check_type(path)

        if path_type.minor == 'path':
            return BeamData.read_file_from_path(path, **kwargs)

        return NotImplementedError

    @staticmethod
    def write_file(data, path, overwrite=True, **kwargs):

        if type(path) is str:
            path = Path(path)

        path_type = check_type(path)

        if (not overwrite) and path.exists():
            raise NameError(f"File {path} exists. Please specify write_file(...,overwrite=True) to write on existing file")

        BeamData.clean_path(path)

        if path_type.minor == 'path':
            path = BeamData._write_file_to_path(data, path, **kwargs)
        else:
            raise NotImplementedError

        return path


    @staticmethod
    def read_tree(paths, **kwargs):

        if type(paths) is str:
            paths = Path(paths)

        paths_type = check_type(paths)
        if paths_type.major == 'container':
            keys = []
            values = []
            paths = []
            for k, next_path in iter_container(paths):
                values.append(BeamData.read_tree(next_path, **kwargs))
                keys.append(k)
                paths.append(next_path)

            if not is_arange(keys):
                values = dict(zip(keys, values))

            return values

        return BeamData.read_object(paths, **kwargs)

    @staticmethod
    def recursive_root_finder(all_paths, head=None):
        if head is None:
            head = []

        all_paths_type = check_type(all_paths)
        if all_paths_type.major == 'container':

            k, v = next(iter_container(all_paths))
            head.append(k)
            return BeamData.recursive_root_finder(v, head=head)

        if all_paths.is_file():
            return all_paths.parent.joinpath(all_paths.stem)

        for _ in head:
            all_paths = all_paths.parent

        return all_paths

    @staticmethod
    def recursive_map_path(path):

        if path.is_dir():

            keys = []
            values = []

            for next_path in path.iterdir():
                keys.append(next_path)
                values.append(BeamData.recursive_map_path(next_path))

            # if the directory contains chunks it is considered as a single path
            if all([is_chunk(p, chunk_pattern=BeamData.chunk_file_extension) for p in keys]):
                return path

            if not is_arange(keys):
                values = dict(zip(keys, values))

            return values

        # we store the files without their extension
        if path.is_file():
            return path.parent.joinpath(path.stem)

        return None

    def as_tensor(self, device=None, dtype=None, return_vector=False):

        func = partial(as_tensor, device=device, dtype=dtype, return_vector=return_vector)
        self.data = recursive(func)(self.data)
        self._objects_type = 'tensor'

    @property
    def values(self):

        if not self.cached:
            self.cache()

        return self.data

    @staticmethod
    def read_object(path, **kwargs):

        if type(path) is str:
            path = Path(path)

        if path.is_file():
            return BeamData.read_file(path, **kwargs)

        elif path.is_dir():

            keys = []
            values = []
            orientation = 0

            for next_path in path.iterdir():

                if next_path.stem == BeamData.configuration_file_name:
                    conf = pd.read_pickle(next_path)
                    orientation = conf['orientation']

                elif not next_path.name.startswith('.'):
                    keys.append(next_path.split(BeamData.chunk_file_extension)[0])
                    values.append(BeamData.read_file(next_path, **kwargs))

            if all([is_chunk(p, chunk_pattern=BeamData.chunk_file_extension) for p in keys]):
                return collate_chunks(*values, keys=keys, dim=orientation)

            if all([is_chunk(p, chunk_pattern=BeamData.partition_directory_name) for p in keys]):
                return recursive_collate_chunks(*values, dim=orientation)

        else:

            for p in path.partent.iterdir():
                if p.stem == path.stem:
                    return BeamData.read_file(p, **kwargs)

            logger.warning(f"No object found in path: {path}")
            return None

    @staticmethod
    def _read_file_from_path(path, **kwargs):

        ext = path.suffix

        if ext == '.fea':
            x = pd.read_feather(path, **kwargs)

            c = x.columns
            for ci in c:
                if BeamData.feather_index_mark in ci:
                    index_name = ci.lstrip(BeamData.feather_index_mark)
                    x = x.rename(columns={ci: index_name})
                    x = x.set_index(index_name)
                    break

        elif ext == '.csv':
            x = pd.read_csv(path, **kwargs)
        elif ext in ['.pkl', '.pickle']:
            x = pd.read_pickle(path, **kwargs)
        elif ext in ['.npy', '.npz']:
            x = np.load(path, allow_pickle=True, **kwargs)
        elif ext == '.scipy_npz':
            x = scipy.sparse.load_npz(path)
        elif ext == '.parquet':
            x = pd.read_parquet(path, **kwargs)
        elif ext == '.pt':
            x = torch.load(path, **kwargs)
        elif ext in ['.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt']:
            x = pd.read_excel(path, **kwargs)
        elif ext == '.avro':
            x = []
            with open(path, 'rb') as fo:
                for record in fastavro.reader(fo):
                    x.append(record)
        elif ext == '.json':
            x = []
            with open(path, 'r') as fo:
                for record in fastavro.json_reader(fo):
                    x.append(record)
        elif ext == '.orc':
            x = pa.orc.read(path, **kwargs)

        else:
            raise ValueError("Unknown extension type.")

        return x

    @staticmethod
    def _write_file_to_path(x, path, **kwargs):

        if type(path) is str:
            path = Path(path)

        ext = path.suffix

        if ext == '.fea':
            x = pd.DataFrame(x)
            index_name = x.index.name if x.index.name is not None else 'index'
            df = x.reset_index()
            new_name = BeamData.feather_index_mark + index_name
            x = df.rename(columns={index_name: new_name})
            x.to_feather(path, **kwargs)
        elif ext == '.csv':
            x = pd.DataFrame(x)
            x.to_csv(path, **kwargs)
        elif ext in ['.pkl', '.pickle']:
            pd.to_pickle(x, path, **kwargs)
        elif ext == '.npy':
            np.save(path, x, **kwargs)
        elif ext == '.npz':
            np.savez(path, x, **kwargs)
        elif ext == '.scipy_npz':
            scipy.sparse.save_npz(path, x, **kwargs)
            os.rename(f'{path}.npz', path)
        elif ext == '.parquet':
            x = pd.DataFrame(x)
            x.to_parquet(path, **kwargs)
        elif ext == '.pt':
            torch.save(x, path, **kwargs)
        else:
            raise ValueError("Unsupported extension type.")

        return path

    @staticmethod
    def write_data(data, path, sizes=None, chunk_strategy='files', archive_size=int(1e6), chunksize=int(1e9),
              chunklen=None, n_chunks=None, partition=None, file_type=None, **kwargs):

        if type(path) is str:
            path = Path(path)

        if sizes is None:
            sizes = recursive_size(data)

        data_type = check_type(data)

        if data_type.major == 'container':

            size_summary = sum(recursive_flatten(sizes))

            if size_summary < archive_size:
                BeamData.write_object(data, path, size=size_summary, archive=True, **kwargs)

            elif chunk_strategy == 'data':

                if (n_chunks is None) and (chunklen is None):
                    n_chunks = max(int(np.round(size_summary / chunksize)), 1)
                elif (n_chunks is not None) and (chunklen is not None):
                    logger.warning("processor.write requires only one of chunklen|n_chunks. Defaults to using n_chunks")
                elif n_chunks is None:
                    n_chunks = max(int(np.round(recursive_len(data) / chunklen)), 1)

                if n_chunks > 1:

                    for i, part in enumerate(divide_chunks(data, n_chunks=n_chunks)):
                        name = f'{i:06}{BeamData.partition_directory_name}'
                        BeamData.write_data(part, path.joinpath(name), sizes=size_summary, archive_size=archive_size,
                                            n_chunks=1, partition=partition, file_type=file_type, **kwargs)

                else:

                    for k, v in iter_container(data):
                        if type(k) is not str:
                            k = f'{k:06}'
                        BeamData.write_data(v, path.joinpath(k), sizes=sizes[k], archive_size=archive_size,
                                            n_chunks=1, partition=partition,
                                            file_type=file_type, **kwargs)

            else:
                for k, v in iter_container(data):
                    if type(k) is not str:
                        k = f'{k:06}'
                    BeamData.write_data(v, path.joinpath(k), sizes=sizes[k], archive_size=archive_size,
                                        chunksize=chunksize, chunklen=chunklen,
                                        n_chunks=n_chunks, partition=partition,
                                        file_type=file_type, **kwargs)

        else:
            BeamData.write_object(data, path, size=sizes, archive_size=archive_size,
                                        chunksize=chunksize, chunklen=chunklen,
                                        n_chunks=n_chunks, partition=partition,
                                        file_type=file_type, **kwargs)

        return paths

    @staticmethod
    def write_object(data, path, override=True, size=None, archive=False, compress=None, chunksize=int(1e9),
              chunklen=None, n_chunks=None, partition=None, file_type=None, **kwargs):

        if type(path) is str:
            path = Path(path)

        if not override:
            if path.exists() or (path.parent.is_dir() and any(p.stem == path.stem for p in path.parent.iterdir())):
                logger.warning(f"path {path} exists. To override, specify override=True")
                return

        if archive:
            object_path = BeamData.write_file(data, path.with_suffix('.pkl'), override=override, **kwargs)

        else:

            if (n_chunks is None) and (chunklen is None):
                if size is None:
                    size = sum(recursive_flatten(recursive_size(data)))
                n_chunks = max(int(np.round(size / chunksize)), 1)
            elif (n_chunks is not None) and (chunklen is not None):
                logger.warning("processor.write requires only one of chunklen|n_chunks. Defaults to using n_chunks")
            elif n_chunks is None:
                n_chunks = max(int(np.round(recursive_len(data) / chunklen)), 1)

            data_type = check_type(data)
            if partition is not None and data_type.minor == 'pandas':
                priority = ['.parquet', '.fea', '.pkl']
            elif data_type.minor in ['pandas', 'numpy']:
                priority = ['.fea', '.parquet', '.pkl']
            elif data_type.minor == 'scipy_sparse':
                priority = ['scipy_npz', 'npy', '.pkl']
            elif data_type.minor == 'tensor':
                priority = ['.pt']
            else:
                priority = ['.pkl']

            if file_type is not None:
                priority.insert(file_type, 0)

            if n_chunks > 1:
                data = list(divide_chunks(data, n_chunks=n_chunks))
            else:
                data = [data]

            if len(data) > 1:
                path.mkdir()
                BeamData.write_file({'orientation': 0}, path.joinpath(BeamData.configuration_file_name))
                object_path = path

            for i, di in data:

                if len(data) > 1:
                    path_i = path.joinpath(f"{i:06}{BeamData.chunk_file_extension}")
                else:
                    path_i = path

                for ext in priority:
                    file_path = path_i.with_suffix(ext)

                    if len(data) == 1:
                        object_path = file_path
                    try:
                        kwargs = {}
                        if ext == '.parquet':
                            if compress is False:
                                kwargs['compression'] = None
                            BeamData.write_file(di, file_path, partition_cols=partition, coerce_timestamps='us',
                                            allow_truncated_timestamps=True, **kwargs)
                        elif ext == '.fea':
                            if compress is False:
                                kwargs['compression'] = 'uncompressed'
                            BeamData.write_file(di, file_path, **kwargs)

                        elif ext == '.pkl':
                            if compress is False:
                                kwargs['compression'] = 'none'
                            BeamData.write_file(di, file_path, **kwargs)

                        elif ext == '.scipy_npz':
                            if compress is False:
                                kwargs['compressed'] = True
                            BeamData.write_file(di, file_path, **kwargs)

                        else:
                            BeamData.write_file(di, file_path, **kwargs)

                        error = False
                        priority = [ext]
                        break

                    except Exception as e:
                        logger.warning(f"Failed to write file: {file_path.name}. Trying with the next file extension")
                        logger.debug(e)
                        error = True
                        if os.path.exists(file_path):
                            os.remove(file_path)

                if error:
                    logger.error(f"Could not write file: {path_i.name}.")

        return object_path
    @property
    def columns_map(self):

        if self._columns_map is not None:
            return self._columns_map

        if self.columns is not None:
            self._columns_map = {str(k): i for i, k in enumerate(self.columns)}

        self._columns_map = None
        return self._columns_map

    def keys(self):
        if self.orientation == 'simple':
            return self.columns
        return recursive_keys(self.data)

    def concatenate_data(self, data):

        if self.objects_type == 'tensor':
            return torch.cat(data)
        if self.objects_type == 'pandas':
            return pd.concat(data, axis=0)
        if self.objects_type == 'numpy':
            return np.concatenate(data)

    def store(self, data=None, path=None, compress=None, chunksize=int(1e9),
              chunklen=None, n_chunks=None, partition=None, chunk_strategy='files',
              archive_len=int(1e6), override=True, **kwargs):

        if path is None:
            path = self.root_path

        if data is None:
            data = self.data

        compress = self.compress if compress is None else compress
        chunksize = self.chunksize if chunksize is None else chunksize
        chunklen = self.chunklen if chunklen is None else chunklen
        n_chunks = self.n_chunks if n_chunks is None else n_chunks

        partition = self.partition if partition is None else partition
        override = self.override if override is None else override
        archive_len = self.archive_len if archive_len is None else archive_len
        chunk_strategy = self.chunk_strategy if chunk_strategy is None else chunk_strategy

        data_type = check_type(data)
        if data_type.major != 'container':
            data = {BeamData.default_data_file_name: data}

        BeamData.write_data(data, path, chunk_strategy=chunk_strategy, archive_size=archive_len, chunksize=chunksize,
              chunklen=chunklen, n_chunks=n_chunks, partition=partition, compress=compress, override=override, **kwargs)

        # store info and conf files
        info_path = path.joinpath(BeamData.info_file_name)
        BeamData.write_object(self.info, info_path, archive=True)
        conf_path = path.joinpath(BeamData.configuration_file_name)
        BeamData.write_object(self.conf, conf_path, archive=True)

        self.stored = True
        self.root_path = path
        self.data = data
        self.all_paths = BeamData.recursive_map_path(path)

    def cache(self, path=None, **kwargs):

        if path is None:

            if self.all_paths is not None:
                path = self.all_paths
            else:
                path = self.root_path

            root_path = self.root_path

        else:

            path_type = check_type(path)
            if path_type.major == 'container':
                root_path = BeamData.recursive_root_finder(path)
            else:
                root_path = path

        # read the conf and info files

        if not self.stored:
            logger.warning("stored=False, data is seems to be un-synchronized")

        data = self.read_tree(path=path, **kwargs)

        if type(data) is dict and 'data' in data and len(data) == 1:
            data = data['data']

        self.root_path = root_path
        self.all_paths = BeamData.recursive_map_path(root_path)
        self.data = data
        self.stored = True
        self.cached = True

    def inverse_map(self, ind):

        ind = slice_to_index(ind, l=len(self), sliced=self.index)

        index_type = check_type(ind)
        if index_type.major == 'scalar':
            ind = [ind]

        if self.index_mapper is not None:
            ind = self.index_mapper.loc[ind].values

        return ind

    def _loc(self, ind):
        ind = self.inverse_map(ind)
        return self.slice_index(ind)

    def _iloc(self, ind):

        ind = slice_to_index(ind, l=len(self), sliced=self.index)
        index_type = check_type(ind)
        if index_type.major == 'scalar':
            ind = [ind]

        return self.slice_index(ind)

    def slice_columns(self, columns):

        if not self.cached:
            raise LookupError(f"Cannot slice by index as data is not cached")

        if self.orientation == 'simple':
            pass
            data = self.data[columns]
        elif self.orientation == 'columns':
            pass
        elif self.orientation == 'index':
            pass
        elif self.orientation == 'other':
            pass
        else:
            raise ValueError(f"Cannot fetch batch for BeamData with orientation={self.orientation}")

        return BeamData(data=data, path=all_paths, lazy=self.lazy, **kwargs)

    def slice_index(self, index):

        if not self.cached:
            raise LookupError(f"Cannot slice by index as data is not cached")

        if self.orientation == 'simple':
            data = self.data[index]

        elif self.orientation == 'columns':

            data = recursive_batch(self.data, index)
            return BeamData(data, index=index)

        elif self.orientation == 'index':

            info =  self.info.iloc[index]
            info['reverse_index'] = np.arange(len(info))
            batch = []
            batch_index = []
            for i, d in enumerate(self.flatten_data):
                fold_info = info[info['fold'] == i].values
                batch.append(self.flatten_data[fold_info['fold_index']])
                batch_index.append(fold_info['reverse_index'].values)

            batch = self.concatenate_data(batch)
            batch_index = pd.Series(np.arange(len(info)), index=np.concatenate([batch_index]))
            batch_index = batch_index.sort_index().values
            batch = batch[batch_index]
            return batch

        elif self.orientation == 'other':

            info = self.info.iloc[index]
            db = BeamData.recursive_filter(self.data, info)
            return BeamData(db.data, index=db.index)

        else:
            raise ValueError(f"Cannot fetch batch for BeamData with orientation={self.orientation}")

        label = self.label.loc[index]
        return DataBatch(data=data, index=index, label=label)

    def slice_keys(self, keys):

        keys_type = check_type(keys)
        if keys_type.major == 'scalar':
            if self.cached:
                data = self.data[keys]
            else:
                data = None

            if self.stored:
                all_paths = self.all_paths[keys]
            else:
                all_paths = None

        else:

            data_type = check_type(self.data)

            if self.cached:

                data = [] if data_type.minor == 'list' else {}
                for k in keys:
                    data[k] = self.data[k]

            else:

                data = None

            if self.stored:

                all_paths = [] if data_type.minor == 'list' else {}
                for k in keys:
                    all_paths[k] = self.all_paths[k]

                if self.lazy:
                    data = None
                else:
                    data = BeamData.read_tree(all_paths)

            else:
                all_paths = None

        if self.orientation == 'columns':
            kwargs = dict(index=self.index, label=self.label)
        elif self.orientation == 'simple':
            kwargs = dict(columns=self.columns, index=self.index, label=self.label)
        elif self.orientation == 'index':
            kwargs = dict(columns=self.columns, label=self.index)
        else:
            kwargs = dict()

        return BeamData(data=data, path=all_paths, lazy=self.lazy, **kwargs)

    def inverse_columns_map(self, columns):

        columns_map = self.columns_map
        if check_type(columns).major == 'scalar':
            columns = columns_map[columns]
        else:
            columns = [columns_map[i] for i in columns]

        return columns

    def __setitem__(self, key, value):

        if self.orientation == 'simple':
            self.data.__setitem__(key, value)

        else:
            self.data[key] = value

        self.stored = False

    def __getitem__(self, item):

        '''

        @param item:
        @return:

        The axes of BeamData objects are considered to be in order: [keys, index, columns, <rest of shape>]
        if BeamData is orient==simple (meaning there are no keys), the first axis disappear.

        Optional item configuration:


        [keys] - keys is ether a slice, list or scalar.
        [index] - index is pandas/numpy/tensor array
        [keys, index] - keys is ether a slice, list or scalar and index is an array

        '''


        axes = ['keys', 'index', 'columns', 'other']
        obj = self
        item_type = check_type(item)
        if item_type.minor != 'tuple':
            item = (item, )
        for j, i in enumerate(item):
            if i == slice(None):
                axes.pop(0)
                continue

            i_type = check_type(i)

            # skip the first axis in this case
            if axes[0] == 'keys' and (i_type.minor in ['pandas', 'numpy', 'tensor'] or self.orientation == 'simple'):
                axes.pop(0)
            if self.orientation == 'simple' and axes[0] == 'index' and i_type.element == 'str':
                axes.pop(0)

            a = axes[0]
            if a == 'keys':
                obj = obj.slice_keys(i)
                axes.pop(0)

            elif a == 'index':

                if i_type.major == 'slice':
                    i = slice_to_index(i, l=len(obj))

                if not obj.cached:
                    obj.cache()

                obj = obj.slice_index(i)
                axes.pop(0)

            else:

                #todo: work on the case of simple orientation with columns

                if a == 'columns' and hasattr(self.data, 'columns'):
                    ind = item[j:]
                elif a == 'columns' and self.columns is not None:
                    ind = (self.inverse_map(i), *item[j+1:])
                else:
                    ind = item[j:]



                if type(obj) is BeamData:
                    obj = obj.slice_data(ind)

                elif type(obj) is DataBatch:
                    data = recursive_batch(obj.data, ind)
                    obj = DataBatch(data=data, index=obj.index, label=obj.label)

                else:
                    raise ValueError(f"Object type is {type(obj)}")

        return obj
