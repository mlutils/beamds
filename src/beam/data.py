import itertools
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from .utils import check_type, slice_to_index, as_tensor, to_device, recursive_batch, as_numpy, beam_device, \
    recursive_device, container_len, recursive, recursive_len, recursive_shape, recursive_types
import pandas as pd
import math
import hashlib
import sys
import warnings
import argparse
from collections import namedtuple
from .utils import divide_chunks, collate_chunks, recursive_chunks, iter_container, logger, \
    recursive_size_summary, container_len, is_arange, listdir_fullpath, is_chunk, rmtree, \
    recursive_size, recursive_flatten, recursive_collate_chunks, recursive_keys, recursive_slice_columns, \
    recursive_slice, recursive_flatten_with_keys, get_item_with_tuple_key
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
                 partition=None, archive_len=int(1e6), preferred_orientation='columns', read_kwargs=None, write_kwargs=None,
                 quick_getitem=False, orientation=None, info=None, **kwargs):

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

         4. packed: each data element represents different set of data points but each data point may have different nature.
         this could model for example node properties in Knowledge graph where there are many types of nodes with different
         features. In case there is a common index, it can be used to slice and collect the data like the original
         PackedFold object.

         If data is both cached in self.data and stored in self.all_paths, the cached version is always preferred.

        The orientation is inferred from the data. If all objects have same length they are assumed to represent columns
        orientation. If all objects have same shape[1:] they are assumed to represent index orientation. If one wish to pass
        an index orientation data where all objects have same length, one can pass the preferred_orientation='index' argument.

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
        self.preferred_orientation = preferred_orientation
        self.chunk_strategy = chunk_strategy
        self.quick_getitem = quick_getitem

        self.stored = False
        self.cached = True
        self.indices = None

        self._columns_map = None
        self._device = None
        self._len = None
        self._data_types = None
        self._data_type = None
        self._objects_type = None
        self._flatten_data = None
        self._flatten_data_with_keys = None
        self._conf = None
        self._info = info
        self._orientation = orientation

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

    @classmethod
    def from_indexed_pandas(cls, data, *args, **kwargs):

        @recursive
        def get_index(x):
            return x.index

        index = get_index(data)
        kwargs['index'] = index

        return cls(data, *args, **kwargs)

    @property
    def objects_type(self):

        if self._objects_type is not None:
            return self._objects_type

        objects_types = recursive_flatten(self.data_types)
        objects_types = [v.minor for v in objects_types if v.minor != 'none']

        u = np.unique(objects_types)

        if len(u) == 1:
            self._objects_type = u[0]
        else:
            self._objects_type = 'mixed'

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
                          'len': len(self),
                          'columns_map': self.columns_map}
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

            if self.orientation in ['index', 'packed']:
                fold_index = np.concatenate([np.arange(len(d)) for d in self.flatten_data])
                fold = np.concatenate([np.full(len(d), k) for k, d in enumerate(self.flatten_data)])

                # still not sure if i really need this column. if so, it should be fixed
                # fold_key = np.concatenate([np.full(len(d), k) for k, d in self.flatten_data_with_keys.items()])
                lengths = np.array([len(d) for d in self.flatten_data])
                offset = np.cumsum(lengths, axis=0) - lengths
                offset = offset[fold] + fold_index

            else:
                fold_index = None
                fold = None
                offset = None
                # fold_key = None

            if self.index is not None:
                index = np.concatenate(recursive_flatten([self.index]))
            else:
                index = np.arange(len(self))

            if self.label is not None:
                label = np.concatenate(recursive_flatten([self.label]))
            else:
                label = None

            info = {'fold': fold, 'fold_index': fold_index,
                    # 'fold_key': fold_key,
                    'offset': offset,
                    'map': np.arange(len(index)), 'label': label}

            self._info = pd.DataFrame(info, index=index)
            return self._info

        self._info = None
        return self._info

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
    def flatten_data_with_keys(self):
        if self._flatten_data_with_keys is not None:
            return self._flatten_data_with_keys
        self._flatten_data_with_keys = recursive_flatten_with_keys(self.data)
        return self._flatten_data_with_keys

    @property
    def device(self):
        if self._device is not None:
            return self._device

        if self.objects_type == 'tensor':
            self._device = recursive_device(self.data)
        else:
            self._device = None

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
                self._len = container_len(self.data)
            else:
                self._len = sum(recursive_flatten(recursive(lambda x: len(x) if hasattr(x, '__len__') else 0)(self.data)))
            return self._len

        self._len = None
        return self._len

    @property
    def orientation(self):

        if self._orientation is not None:
            return self._orientation

        if self._conf is not None:
            self._orientation = self._conf['orientation']
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

                if self.preferred_orientation == 'columns':
                    lens = recursive_flatten(recursive(lambda x: len(x) if hasattr(x, '__len__') else None)([self.data]),
                                             flat_array=True)
                    lens = list(filter(lambda x: x is not None, lens))

                    lens_index = recursive_flatten(
                        recursive(lambda x: len(x) if hasattr(x, '__len__') else None)([self.index]))
                    lens_index = list(filter(lambda x: x is not None, lens_index))

                    if len(np.unique(lens)) == 1 and sum(lens) > sum(lens_index):
                        self._orientation = 'columns'
                        return self._orientation

                shapes = recursive_flatten(
                    recursive(lambda x: tuple(x.shape[1:]) if hasattr(x, 'shape') and len(x.shape) > 1 else None)([self.data]))

                shapes = list(filter(lambda x: x is not None, shapes))
                if len(np.unique(shapes) == 1):
                    self._orientation = 'index'
                else:
                    self._orientation = 'packed'

        elif self.stored:
            self._orientation = self.conf['orientation']

        else:
            self._orientation = 'packed'

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
                    n_chunks = max(int(np.round(container_len(data) / chunklen)), 1)

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
                n_chunks = max(int(np.round(container_len(data) / chunklen)), 1)

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

    def _concatenate(self, data):

        if self.orientation == 'simple':
            dim = None
        elif self.orientation == 'columns':
            dim = 1
        elif self.orientation == 'index':
            dim = 0
        else:
            return data

        if self.objects_type == 'tensor':
            func = torch.cat = data
            kwargs = {'dim': dim}
        elif self.objects_type == 'pandas':
            func = pd.concat
            kwargs = {'axis': dim}
        elif self.objects_type == 'numpy':
            func = np.concatenate
            kwargs = {'axis': dim}
        else:
            logger.warning(f"Concatenation not implemented for {self.objects_type}, returning the original data")
            return data

        return func(data, **kwargs)

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

    def slice_data(self, index):

        if type(index) is not tuple:
            index = (index,)
        index = tuple([slice(None), slice(None), *index])

        if not self.cached:
            raise LookupError(f"Cannot slice as data is not cached")

        if self.orientation == 'simple':
            data = self.data.__getitem(index)

        elif self.orientation == 'index':
            data = recursive_slice(self.data, index)

        else:
            raise LookupError(f"Cannot slice by columns as data is not in simple or index orientation")

        if self.quick_getitem:
            return DataBatch(data=data, index=self.index, label=self.label)

        return BeamData(data=data, path=self.all_paths, lazy=self.lazy, index=self.index, label=self.label,
                        orientation=self.orientation)

    def slice_columns(self, columns):

        if not self.cached:
            raise LookupError(f"Cannot slice by columns as data is not cached")

        if self.orientation == 'simple':

            if hasattr(self.data, 'loc'):
                data = self.data[columns]
            else:
                data = self.data[:, columns]

        elif self.orientation == 'index':
            data = recursive_slice_columns(self.data, columns)

        else:
            raise LookupError(f"Cannot slice by columns as data is not in simple or index orientation")

        if self.quick_getitem:
            return DataBatch(data=data, index=self.index, label=self.label)

        return BeamData(data=data, path=self.all_paths, lazy=self.lazy, columns=columns,
                        index=self.index, label=self.label, orientation=self.orientation)

    @property
    def stack_values(self):
        data = self._concatenate(self.flatten_data)
        return data

    @staticmethod
    def recursive_filter(x, info):

        def _recursive_filter(x, info, flat_key=0):
            x_type = check_type(x)
            if x_type.major == 'container':

                keys = []
                values = []
                index = []
                label = []

                for k, v in iter_container(x):
                    i, v, l, flat_key = _recursive_filter(v, info, flat_key=flat_key)
                    if len(v):
                        values.append(v)
                        index.append(i)
                        keys.append(k)
                        label.append(l)

                if not is_arange(keys):
                    values = dict(zip(keys, values))
                    index = dict(zip(keys, index))
                    label = dict(zip(keys, label))

                return index, values, label, flat_key

            else:

                info_in_fold = info[info['fold'] == flat_key]
                in_fold_index = info_in_fold['fold_index']
                x_type = check_type(x)

                if x is None:
                    return None, None, None, flat_key + 1
                elif x_type.minor == 'pandas':
                    return in_fold_index.index, x.iloc[in_fold_index.values], info_in_fold['label'], flat_key + 1
                else:
                    return in_fold_index.index, x[in_fold_index.values], info_in_fold['label'], flat_key + 1

        index, data, label, _ = _recursive_filter(x, info)
        return DataBatch(index=index, data=data, label=label)

    def slice_index(self, index):

        if not self.cached:
            raise LookupError(f"Cannot slice by index as data is not cached")

        if self.orientation in ['simple', 'columns']:

            info = None
            if self.label is not None:
                label = self.label.loc[index]
            else:
                label = None

            if self.orientation == 'simple':
                if hasattr(self.data, 'loc'):
                    data = self.data.loc[index]
                else:
                    data = self.data[index]

            else:

                if self.index is not None:
                    iloc = self.index[index]
                else:
                    iloc = index

                data = recursive_batch(self.data, iloc)

        elif self.orientation in ['index', 'packed']:

            info = self.info.loc[index]
            db = BeamData.recursive_filter(self.data, info)
            data = db.data
            index = db.index
            label = db.label

        else:
            raise ValueError(f"Cannot fetch batch for BeamData with orientation={self.orientation}")

        if self.quick_getitem:
            return DataBatch(data=data, index=index, label=self.label)

        return BeamData(data=data, path=None, lazy=self.lazy, columns=self.columns,
                        index=index, label=label, orientation=self.orientation, info=info)

    @staticmethod
    def slice_scalar_or_list(data, keys, data_type=None, keys_type=None):

        if data is None:
            return None

        if data_type is None:
            data_type = check_type(data)

        if keys_type is None:
            keys_type = check_type(keys)

        if keys_type.major == 'scalar':
            return data[keys]
        else:
            sliced = [] if data_type.minor == 'list' else {}
            for k in keys:
                sliced[k] = data[k]
            return sliced

    def slice_keys(self, keys):

        data = None
        all_paths = None
        keys_type = check_type(keys)

        if self.cached:
            data = BeamData.slice_scalar_or_list(self.data, keys, keys_type=keys_type, data_type=self.data_type)

        if self.stored:
            all_paths = BeamData.slice_scalar_or_list(self.all_paths, keys, keys_type=keys_type,
                                                      data_type=self.data_type)

        if not self.lazy and self.stored and data is None:
            data = BeamData.read_tree(all_paths)

        index = self.index
        label = self.label
        # info = self._info

        if self.orientation != 'columns':
            if index is not None:
                index = BeamData.slice_scalar_or_list(index, keys, keys_type=keys_type, data_type=self.data_type)
            if label is not None:
                label = BeamData.slice_scalar_or_list(label, keys, keys_type=keys_type, data_type=self.data_type)

        if self.quick_getitem and data is not None:
            return DataBatch(data=data, index=index, label=label)

        # determining orientation and info can be ambiguous so we let BeamData to calculate it
        # from the index and label arguments

        # if self.orientation != 'columns' and info is not None:
        #         info = info.loc[index]
        # return BeamData(data=data, path=all_paths, lazy=self.lazy, columns=self.columns,
        #                 index=index, label=label, orientation=self.orientation, info=info)

        return BeamData(data=data, path=all_paths, lazy=self.lazy, columns=self.columns, index=index, label=label)

    def inverse_columns_map(self, columns):

        columns_map = self.columns_map
        if check_type(columns).major == 'scalar':
            columns = columns_map[columns]
        else:
            columns = [columns_map[i] for i in columns]

        return columns

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = f"BeamData: \n"
        s += f"  path: {self.root_path} \n"
        s += f"  orientation: {self.orientation} \t| columns: {self.columns} \n"
        s += f"  has_index: {self.index is not None} \t| has_label: {self.label is not None} \n"
        s += f"  lazy: {self.lazy}\t| stored: {self.stored}\t| cached: {self.cached} \n"
        s += f"  device: {self.device}\t| objects_type: {self.objects_type}\t| quick_getitem: {self.quick_getitem} \n"
        s += f"  keys: {self.keys()} \n"
        s += f"  sizes:\n"
        s += f"  {recursive_size(self.data)} \n"
        s += f"  shapes:\n"
        s += f"  {recursive_shape(self.data)} \n"
        s += f"  types:\n"
        s += f"  {recursive_types(self.data)} \n"
        return s

    def __setitem__(self, key, value):

        if self.orientation == 'simple':
            self.data.__setitem__(key, value)

        else:
            self.data[key] = value

        self.stored = False

    def apply(self, func, *args, **kwargs):
        data = recursive(func, *args, **kwargs)(self.data)

        return BeamData(data, index=self.index, label=self.label)

    def reset_index(self):
        return BeamData(self.data, index=None, label=self.label)

    def __iter__(self):
        if self.cached:
            for k, v in self.flatten_data_with_keys:

                index = None
                if self.index is not None:
                    index = get_item_with_tuple_key(self.index, k)

                label = None
                if self.label is not None:
                    label = get_item_with_tuple_key(self.label, k)

                yield k, BeamData(v, lazy=self.lazy, columns=self.columns, index=index, label=label)
        else:
            for k, p in recursive_flatten_with_keys(self.all_paths):

                index = None
                if self.index is not None:
                    index = get_item_with_tuple_key(self.index, k)

                label = None
                if self.label is not None:
                    label = get_item_with_tuple_key(self.label, k)

                yield k, BeamData(path=p, lazy=self.lazy, columns=self.columns, index=index, label=label)

    def sample(self, n, replace=True):

        if replace:
            ind = np.random.choice(len(self), size=n, replace=True)
        else:
            ind = np.random.randint(len(self), size=n)

        ind = self.info.loc[ind].index
        return self[ind]

    def __getitem__(self, item):

        '''

        @param item:
        @return:

        The axes of BeamData objects are considered to be in order: [keys, index, columns, <rest of shape>]
        if BeamData is orient==simple (meaning there are no keys), the first axis disappears.

        Optional item configuration:


        [keys] - keys is ether a slice, list or scalar.
        [index] - index is pandas/numpy/tensor array
        [keys, index] - keys is ether a slice, list or scalar and index is an array

        '''

        if self.orientation == 'simple':
            axes = ['index', 'columns', 'else']
        else:
            axes = ['keys', 'index', 'columns', 'else']

        obj = self
        item_type = check_type(item)
        if item_type.minor != 'tuple':
            item = (item, )
        for i, ind_i in enumerate(item):

            # skip if this is a full slice
            if ind_i == slice(None):
                axes.pop(0)
                continue

            i_type = check_type(ind_i)

            # skip the first axis in these case
            if axes[0] == 'keys' and (i_type.minor in ['pandas', 'numpy', 'slice', 'tensor']):
                axes.pop(0)
            if axes[0] == 'keys' and (i_type.minor == 'list' and i_type.element == 'int'):
                axes.pop(0)
            # for orientation == 'simple' we skip the first axis if we slice over columns
            if self.orientation == 'simple' and axes[0] == 'index' and i_type.element == 'str':
                axes.pop(0)

            a = axes.pop(0)
            if a == 'keys':
                obj = obj.slice_keys(ind_i)

            elif a == 'index':

                if i_type.major == 'slice':
                    ind_i = slice_to_index(ind_i, l=len(obj))

                if i_type.element == 'bool':
                    ind_i = self.info.iloc[ind_i].index

                if not isinstance(obj, BeamData):
                    ValueError(f"quick_getitem supports only a single index slice")

                obj = obj.slice_index(ind_i)

            elif a == 'columns':

                if not isinstance(obj, BeamData):
                    ValueError(f"quick_getitem supports only a single index slice")

                obj = obj.slice_columns(ind_i)

            else:

                ind_i = item[i:]
                obj = obj.slice_data(ind_i)
                break

        return obj
