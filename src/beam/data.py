import itertools
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from .utils import check_type, slice_to_index, as_tensor, to_device, recursive_batch, as_numpy, beam_device, \
    recursive_device, recursive_len
import pandas as pd
import math
import hashlib
import sys
import warnings
import argparse
from collections import namedtuple
from .utils import divide_chunks, collate_chunks, recursive_chunks, iter_container, logger, \
    recursive_size, recursive_len, is_arange, listdir_fullpath, is_chunk
from .parallel import parallelize
from collections import OrderedDict
import os
import fastavro
import pyarrow as pa
import shutil
import pathlib
from argparse import Namespace
import scipy
import sys
from pathlib import Path


class BeamData(object):

    feather_index_mark = "index:"

    def __init__(self, *args, data=None, root_path=None, all_paths=None,
                 lazy=True, device=None, columns=None, index=None, sort_index=False,
                 quick_getitem=True, **kwargs):

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
                self.data = BeamData.read_all_paths(all_paths)

            self.synced = True


        if self.data is not None:
            self.cached = True
            self.synced = (not self.synced)

        if not lazy:
            self.cache()

    def read_all_paths(self, all_paths, **kwargs):

        all_paths_type = check_type(all_paths)
        if all_paths_type.major == 'container':
            keys = []
            values = []
            paths = []
            for k, next_path in iter_container(all_paths):
                values.append(self.read_all_paths(next_path, **kwargs))
                keys.append(k)
                paths.append(next_path)

            if not is_arange(keys):
                values = dict(zip(keys, values))

            return values

        if all_paths.is_dir():
            values = []
            for k, next_path in all_paths.iterdir():
                values.append(BeamData.read_file(path=next_path, **kwargs))

            values = collate_chunks(*values, dim=self.orientation_dimension)




        return None

    @staticmethod
    def recursive_root_finder(all_paths, head=None):
        if head is None:
            head = []

        all_paths_type = check_type(all_paths)
        if all_paths_type.major == 'container':

            k, v =  next(iter_container(all_paths))
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
            if all([is_chunk(p) for p in keys]):
                return path

            if not is_arange(keys):
                values = dict(zip(keys, values))

            return values

        # we store the files without their extension
        if path.is_file():
            return path.parent.joinpath(path.stem)

        return None


    def get_orientation(self):
        pass

    def as_tensor(self):
        return NotImplementedError

    @property
    def values(self):
        return self.data

    @staticmethod
    def read_file(path, **kwargs):

        _, ext = os.path.splitext(path)

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
    def write_file(x, path, overwrite=True, **kwargs):

        if (not overwrite) and (os.path.isdir(path) or os.path.isfile(path)):
            logger.error(f"File {path} exists. Please specify write_file(...,overwrite=True) to write on existing file")
            return

        BeamData.clean_path(path)

        _, ext = os.path.splitext(path)

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
    def clean_path(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

        os.makedirs(path, exist_ok=True)
        os.rmdir(path)

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


    def __getitem__(self, item):

        if self.cached:
            return self.data[item]

        item_paths = self.all_paths[item]
        item_type = check_type(item_paths)

        if item_type.major == 'scalar':
            return BeamData.read_file(item_paths)

        return BeamData(all_paths=item_paths)


    def read(self, path=None, relative=True, lazy=False, collate=True, **kwargs):

        path_type = check_type(path)

        if path is None:
            path = self.root_path
        elif relative and path_type.major == 'scalar':
            path = os.path.join(self.root_path, path)

        if path_type.major in ['container', 'array']:

            values = []
            keys = []
            paths = []
            for p, next_path in iter_container(path):

                values.append(self.read(path=next_path, relative=False, lazy=lazy, **kwargs))
                keys.append(p)
                paths.append(next_path)

            if all(['_chunk' in p for p in paths]) and collate:
                values = collate_chunks(*values, dim=0)
            elif not is_arange(keys):
                values = dict(zip(keys, values))
            return values

        elif os.path.isfile(path):
            if lazy:
                return path
            return BeamData.read_file(path, **kwargs)

        elif os.path.isdir(path):

            values = []
            keys = []
            for p in sorted(os.listdir(path)):

                next_path = os.path.join(path, p)
                values.append(self.read(path=next_path, relative=False, lazy=lazy, **kwargs))

                if os.path.isfile(next_path):
                    p, _ = os.path.splitext(p)

                keys.append(p)

            if all(['_chunk' in p for p in keys]) and collate:
                values = collate_chunks(*values, dim=0)
            elif not is_arange(keys):
                values = dict(zip(keys, values))
            return values

        elif any(str(path) in str(p) for p in listdir_fullpath(os.path.dirname(path))):

            list_dir = listdir_fullpath(os.path.dirname(path))
            i = [os.path.splitext(p)[0] for p in list_dir].index(path)

            path = list_dir[i]

            if lazy:
                return path
            return BeamData.read_file(path, **kwargs)

        else:
            return None


    def write(self, x=None, path=None, root=True, relative=True, compress=None, chunksize=int(1e9),
              chunklen=None, n_chunks=None, partition=None, file_type=None, override=True, archive=None,
              **kwargs):

        if x is None:
            x = self.data

        if path is None:
            path = self.root_path
        elif relative:
            path = os.path.join(self.root_path, path)

        if root:
            if (n_chunks is None) and (chunklen is None):
                max_size = recursive_size(x, mode='max')
                n_chunks = max(int(np.round(max_size / chunksize)), 1)
            elif (n_chunks is not None) and (chunklen is not None):
                logger.warning("processor.write requires only one of chunklen|n_chunks. Defaults to using n_chunks")
            elif n_chunks is None:
                n_chunks = max(int(np.round(recursive_len(x) / chunklen)), 1)

            if override:
                BeamData.clean_path(path)

        x_type = check_type(x)

        if x_type.major == 'container' and (not (archive == True)):
            os.mkdir(path)

            file_type_type = check_type(file_type)
            archive_type = check_type(archive)

            for k, v in iter_container(x):

                if file_type_type.major == 'container':
                    ft = file_type[k]
                else:
                    ft = file_type

                if archive_type.major == 'container':
                    ar = archive[k]
                else:
                    ar = archive

                self.write(v, os.path.join(path, str(k)), relative=relative, n_chunks=n_chunks,
                           root=False, compress=compress, file_type=ft, **kwargs)
        else:

            if partition is not None and x_type.minor == 'pandas':
                priority = ['.parquet', '.fea', '.pkl']
            elif x_type.minor in ['pandas', 'numpy']:
                priority = ['.fea', '.parquet', '.pkl']
            elif x_type.minor == 'scipy_sparse':
                priority = ['scipy_npz', 'npy', '.pkl']
            elif x_type.minor == 'tensor':
                priority = ['.pt']
            else:
                priority = ['.pkl']

            if file_type is not None:
                priority.insert(file_type, 0)

            x = list(divide_chunks(x, n_chunks=n_chunks))

            if len(x) > 1:
                os.mkdir(path)

            for i, xi in x:

                if len(x) > 1:
                    path_i = pathlib.Path(os.path.join(path, f"{i:06}_chunk"))
                else:
                    path_i = pathlib.Path(path)

                for ext in priority:
                    file_path = path_i.with_suffix(ext)
                    try:
                        kwargs = {}
                        if ext == '.parquet':
                            if compress is False:
                                kwargs['compression'] = None
                            self.write_file(xi, file_path, partition_cols=partition, coerce_timestamps='us',
                                            allow_truncated_timestamps=True, **kwargs)
                        elif ext == '.fea':
                            if compress is False:
                                kwargs['compression'] = 'uncompressed'
                            self.write_file(xi, file_path, **kwargs)

                        elif ext == '.pkl':
                            if compress is False:
                                kwargs['compression'] = 'none'
                            self.write_file(xi, file_path, **kwargs)

                        elif ext == '.scipy_npz':
                            if compress is False:
                                kwargs['compressed'] = True
                            self.write_file(xi, file_path, **kwargs)

                        else:
                            self.write_file(xi, file_path, **kwargs)

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
