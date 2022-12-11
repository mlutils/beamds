import itertools
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from .utils import check_type, slice_to_index, as_tensor, to_device, recursive_batch, as_numpy, beam_device, \
    recursive_device, recursive_len, recursive_func
import pandas as pd
import math
import hashlib
import sys
import warnings
import argparse
from collections import namedtuple
from .utils import divide_chunks, collate_chunks, recursive_chunks, iter_container, logger, \
    recursive_size_summary, recursive_len, is_arange, listdir_fullpath, is_chunk, rmtree, \
    recursive_size,recursive_flatten
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


DataBatch = namedtuple("DataBatch", "index label data")


class BeamData(object):

    feather_index_mark = "index:"
    configuration_file_name = '.conf'
    chunk_file_extension = '_chunk'

    def __init__(self, *args, data=None, root_path=None, all_paths=None, label=None,
                 lazy=True, device=None, columns=None, index=None, sort_index=False, part=None,
                 quick_getitem=True, override=True, compress=None, chunksize=int(1e9),
                 chunklen=None, n_chunks=None, partition=None, archive_len=int(1e6),
                 read_kwargs=None, write_kwargs=None, target_device=None,
                 **kwargs):

        '''

        @param args:
        @param data:
        @param root_path: if not str, requires to support the pathlib Path attributes and operations
        @param all_paths:
        @param lazy:
        @param kwargs:

        Possible orientations are: row/column/other

        '''

        self.lazy = lazy

        self.stored = False
        self.cached = False
        self.synced = False

        self.recursive_stored = None
        self.recursive_cached = None
        self.recursive_synced = None
        self.orientation = None

        self.read_kwargs = {} if read_kwargs is None else read_kwargs
        self.write_kwargs = {} if write_kwargs is None else write_kwargs

        root_path_type = check_type(root_path)
        if root_path_type.minor == 'str':
            root_path = Path(root_path)

        if len(args) == 1:
            arg_type = check_type(args[0])
            if arg_type.major == 'container':
                self.data = args[0]
        elif len(args):
            self.data = list(args)
        elif len(kwargs):
            self.data = kwargs
        elif data is not None:
            self.data = data
        else:
            self.data = None

        if all_paths is None:
            self.root_path = root_path
            self.all_paths = BeamData.recursive_map_path(root_path)
        else:
            self.all_paths = all_paths
            self.root_path = BeamData.recursive_root_finder(all_paths)

        if self.all_paths:
            self.stored = True
            if not lazy:
                self.data = BeamData.read_tree(all_paths, **self.read_kwargs)

            self.synced = True

        if self.data is not None:
            self.cached = True
            self.synced = (not self.synced)

        self.target_device = target_device
        if not lazy:
            self.cache()

        if index is not None:
            index = pd.Series(index)

        self.index = index
        self.label = label
        self.part = part
        self.len = None

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

    def to(self, device):
        self.data = recursive_func(self.data, lambda x: x.to(device))
        return self

    def __len__(self):
        if self.len is None:
            if self.cached:
                if self.dim == 0:
                    len = recursive_len(self.data)
                if self.dim == 1:
                    len = sum(recursive_flatten(recursive_func(self.data,
                                                               lambda x: len(x) if hasattr(x, '__len__') else 0)))

                len = None
            self.len = len
        return self.len

    def set_orientation(self):
        if self.cached:
            lens = recursive_flatten(recursive_func([self.data], lambda x: len(x) if hasattr(x, '__len__') else None))
            if len(np.unique(lens) == 1) and lens[0] is not None:
                self.orientation = 'index'
            else:
                lens = recursive_flatten(
                    recursive_func([self.data], lambda x: x.shape[1] if hasattr(x, 'shape') and len(x.shape) > 1 else None))
                if len(np.unique(lens) == 1) and lens[0] is not None:
                    self.orientation = 'columns'
                else:
                    self.orientation = 'none'

        elif self.root_path.is_dir():
            conf = BeamData.read_file_from_path(self.root_path.joinpath(BeamData.configuration_file_name))
            self.orientation = conf['orientation']

        else:
            self.orientation = 'index'

    @property
    def dim(self):
        if self.orientation == 'index':
            return 0
        if self.orientation == 'columns':
            return 1
        return None

    @staticmethod
    def read_file(path, **kwargs):

        if type(path) is str:
            path = Path(path)

        path_type = check_type(path)

        if path_type.minor == 'path':
            return BeamData.read_file_from_path(path, **kwargs)

        return NotImplementedError

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
            if all([is_chunk(p, chunk_patter=BeamData.chunk_file_extension) for p in keys]):
                return path

            if not is_arange(keys):
                values = dict(zip(keys, values))

            return values

        # we store the files without their extension
        if path.is_file():
            return path.parent.joinpath(path.stem)

        return None

    def as_tensor(self):
        return NotImplementedError

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

                keys.append(next_path.split(BeamData.chunk_file_extension)[0])
                values.append(BeamData.read_file(next_path, **kwargs))
            return collate_chunks(*values, keys=keys, dim=orientation)

        else:

            for p in path.partent.iterdir():
                if p.stem == path.stem:
                    return BeamData.read_file(p, **kwargs)

            logger.warning(f"No object found in path: {path}")
            return None

    @staticmethod
    def read_file_from_path(path, **kwargs):

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
    def write_file_to_path(x, path, overwrite=True, **kwargs):

        if type(path) is str:
            path = Path(path)

        if (not overwrite) and path.exists():
            logger.error(f"File {path} exists. Please specify write_file(...,overwrite=True) to write on existing file")
            return

        BeamData.clean_path(path)

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

    @staticmethod
    def write_data(data, path, sizes=None, archive_size=int(1e6), **kwargs):

        if type(path) is str:
            path = Path(path)

        if sizes is None:
            sizes = recursive_size(data)

        data_type = check_type(data)

        if data_type.major == 'container':

            size_summary = sum(recursive_flatten(sizes))
            if size_summary < archive_size:
                BeamData.write_object(data, path, size=size_summary, archive=True, **kwargs)

            else:
                for k, v in iter_container(data):
                    if type(k) is not str:
                        k = f'{k:06}'
                    BeamData.write_data(v, path.joinpath(k), sizes=sizes[k] **kwargs)

        else:
            BeamData.write_object(data, path, size=sizes, **kwargs)


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
            BeamData.write_file_to_path(data, path.with_suffix('.pkl'), override=override, **kwargs)

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
                BeamData.write_file_to_path({'orientation': 0}, path.joinpath(BeamData.configuration_file_name))

            for i, di in data:

                if len(data) > 1:
                    path_i = path.joinpath(f"{i:06}{BeamData.chunk_file_extension}")
                else:
                    path_i = pathlib.Path(path)

                for ext in priority:
                    file_path = path_i.with_suffix(ext)
                    try:
                        kwargs = {}
                        if ext == '.parquet':
                            if compress is False:
                                kwargs['compression'] = None
                            BeamData.write_file_to_path(di, file_path, partition_cols=partition, coerce_timestamps='us',
                                            allow_truncated_timestamps=True, **kwargs)
                        elif ext == '.fea':
                            if compress is False:
                                kwargs['compression'] = 'uncompressed'
                            BeamData.write_file_to_path(di, file_path, **kwargs)

                        elif ext == '.pkl':
                            if compress is False:
                                kwargs['compression'] = 'none'
                            BeamData.write_file_to_path(di, file_path, **kwargs)

                        elif ext == '.scipy_npz':
                            if compress is False:
                                kwargs['compressed'] = True
                            BeamData.write_file_to_path(di, file_path, **kwargs)

                        else:
                            BeamData.write_file_to_path(di, file_path, **kwargs)

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


    def store(self, compress=None, chunksize=int(1e9),
              chunklen=None, n_chunks=None, partition=None, file_type=None, override=True, **kwargs):
        assert self.root_path is not None, "path is unknown, Please define BeamData with path"
        assert self.cached, "data is unavailable, Please define BeamData with valid data"

        self.write(compress=compress, chunksize=chunksize,
                   chunklen=chunklen, n_chunks=n_chunks, partition=partition,
                   file_type=file_type, override=override, **kwargs)

        self.all_paths = self.read(lazy=True)
        self.stored = True
        self.synced = True

    def cache(self, **kwargs):

        if not self.stored:
            logger.warning("data is unavailable in dick, returning None object")

        path = None
        if self.all_paths is not None:
            path = self.all_paths

        self.data = self.read(path=path, **kwargs)

        self.cached = True
        self.synced = True

    def get_batch(self, index):
        return recursive_batch(self.data, index)

    def __getitem__(self, item):

        item_type = check_type(item)
        if item_type.major == 'slice':
            item = slice_to_index(item, l=len(self))

        if item_type.element in ['int', 'slice']:

            if not self.cached:
                self.cache()

            sample = self.get_batch(item)

            if self.target_device is not None:
                sample = to_device(sample, device=self.target_device)

            if self.index is not None:
                ind = self.index.iloc[item].values
            else:
                ind = item

            if self.label is not None:
                label = self.label[item]
            else:
                label = None
                
            return DataBatch(index=ind, label=label, data=sample)


        if self.cached:
            return self.data[item]

        item_paths = self.all_paths[item]
        item_type = check_type(item_paths)

        if item_type.major == 'scalar':
            return BeamData.read_file(item_paths)

        return BeamData(all_paths=item_paths)
