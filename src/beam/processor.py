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
    recursive_size, recursive_len, is_arange, listdir_fullpath
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

class Processor(object):

    def __init__(self, *args, **kwargs):
        pass


class BeamData(object):

    feather_index_mark = "index:"

    def __init__(self, *args, path=None, data_paths=None, to_memory=False, **kwargs):

        self.in_disk = False
        self.in_memory = False
        self.synchronized = False

        if data_paths is None:
            self.root_path = path
            self.data_paths = self.read(lazy=True)
        else:
            self.data_paths = data_paths
            self.root_path = ''

        if self.data_paths:
            self.in_disk = True
            self.synchronized = True

        # if data_paths_types
        self.data = None
        if len(args) == 1:
            arg_type = check_type(args[0])
            if arg_type.major == 'container':
                self.data = args[0]
        else:
            assert len(args) * len(kwargs) == 0, "Please use either args or kwargs"

        if len(args):
            self.data = list(args)
        elif len(kwargs):
            self.data = kwargs

        if self.data is not None:
            self.in_memory = True
            self.synchronized = (not self.synchronized)

        if to_memory:
            self.to_memory()

    @staticmethod
    def read_file(path, **kwargs):

        _, ext = os.path.splitext(path)

        if ext == '.fea':
            x = pd.read_feather(path, **kwargs)

            c = x.columns
            for ci in c:
                if BeamData.feather_index_mark in c:
                    index_name = ci.lstrip(BeamData.feather_index_mark)
                    x = x.rename(columns={ci: index_name})
                    x = x.set_index(index_name)

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

    def to_disk(self, compress=None, chunksize=int(1e9),
              chunklen=None, n_chunks=None, partition=None, file_type=None, override=True, **kwargs):
        assert self.root_path is not None, "path is unknown, Please define BeamData with path"
        assert self.in_memory, "data is unavailable, Please define BeamData with valid data"

        self.write(compress=compress, chunksize=chunksize,
                   chunklen=chunklen, n_chunks=n_chunks, partition=partition,
                   file_type=file_type, override=override, **kwargs)

        self.data_paths = self.read(lazy=True)
        self.in_disk = True
        self.synchronized = True

    def to_memory(self, **kwargs):

        if not self.in_disk:
            logger.warning("data is unavailable in dick, returning None object")

        path = None
        if self.data_paths is not None:
            path = self.data_paths

        self.data = self.read(path=path, **kwargs)

        self.in_memory = True
        self.synchronized = True


    def __getitem__(self, item):

        if self.in_memory:
            return self.data[item]

        item_paths = self.data_paths[item]
        item_type = check_type(item_paths)

        if item_type.major == 'scalar':
            return BeamData.read_file(item_paths)

        return BeamData(data_paths=item_paths)


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
                order = ['.parquet', '.fea', '.pkl']
            elif x_type.minor in ['pandas', 'numpy']:
                order = ['.fea', '.parquet', '.pkl']
            elif x_type.minor == 'scipy_sparse':
                order = ['scipy_npz', 'npy', '.pkl']
            elif x_type.minor == 'tensor':
                order = ['.pt']
            else:
                order = ['.pkl']

            if file_type is not None:
                order.insert(file_type, 0)

            x = list(divide_chunks(x, n_chunks=n_chunks))

            if len(x) > 1:
                os.mkdir(path)

            for i, xi in x:

                if len(x) > 1:
                    path_i = pathlib.Path(os.path.join(path, f"{i:06}_chunk"))
                else:
                    path_i = pathlib.Path(path)

                for ext in order:
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
                        order = [ext]
                        break

                    except Exception as e:
                        logger.warning(f"Failed to write file: {file_path.name}. Trying with the next file extension")
                        logger.debug(e)
                        error = True
                        if os.path.exists(file_path):
                            os.remove(file_path)

                if error:
                    logger.error(f"Could not write file: {path_i.name}.")


class Pipeline(Processor):

    def __init__(self, *ts, track_steps=False, **kwts):

        super().__init__()
        self.track_steps = track_steps
        self.steps = {}

        self.transformers = OrderedDict()
        for i, t in enumerate(ts):
            self.transformers[i] = t

        for k, t in kwts.items():
            self.transformers[k] = t

    def transform(self, x, **kwargs):

        self.steps = []

        for i, t in self.transformers.items():

            kwargs_i = kwargs[i] if i in kwargs.keys() else {}
            x = t.transform(x, **kwargs_i)

            if self.track_steps:
                self.steps[i] = x

        return x


class Reducer(Processor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reduce(self, *xs, **kwargs):
        return collate_chunks(*xs, dim=1, **kwargs)


class Transformer(Processor):

    def __init__(self, *args, state=None, n_jobs=0, n_chunks=None,
                 chunksize=None, squeeze=True, path=None, **kwargs):

        super(Transformer, self).__init__(*args, **kwargs)

        if (n_chunks is None) and (chunksize is None):
            n_chunks = 1

        self.transformers = None
        self.chunksize = chunksize
        self.n_chunks = n_chunks
        self.n_jobs = n_jobs
        self.state = state
        self.squeeze = squeeze
        self.kwargs = kwargs
        self.path = path

    def chunks(self, x, chunksize=None, n_chunks=None, squeeze=None):

        if (chunksize is None) and (n_chunks is None):
            chunksize = self.chunksize
            n_chunks = self.n_chunks
        if squeeze is None:
            squeeze = self.squeeze

        for k, c in recursive_chunks(x, chunksize=chunksize, n_chunks=n_chunks, squeeze=squeeze):
            yield k, c

    def _transform(self, x, key=None, is_chunk=False, fit=False, **kwargs):
        raise NotImplementedError

    def worker(self, x, key=None, is_chunk=False, data_in='memory', strategy='memory', fit=False, **kwargs):

        if data_in == 'disk':
            bd = BeamData(data_paths=x)
            bd.to_memory()
            x = bd.data

        x = self._transform(x, key=key, is_chunk=is_chunk, **kwargs)

        if strategy == 'disk':
            bd = BeamData(x, path=self.path)
            bd.to_disk()
            x = bd.data_paths

        return x

    def fit(self, x, **kwargs):
        return NotImplementedError

    def fit_transform(self, x, **kwargs):
        self.fit(x, **kwargs)
        return self.transform(x, **kwargs)

    def collate(self, x, **kwargs):
        return collate_chunks(*x, dim=0, **kwargs)

    def transform(self, x, chunksize=None, n_chunks=None, n_jobs=None, squeeze=None, parent_strategy='disk',
                  worker_strategy='memory', **kwargs):

        if (chunksize is None) and (n_chunks is None):
            chunksize = self.chunksize
            n_chunks = self.n_chunks
        if squeeze is None:
            squeeze = self.squeeze
        if n_jobs is None:
            n_jobs = self.n_jobs

        data_in = 'memory'
        if isinstance(x, BeamData):
            if parent_strategy == 'disk':
                if x.in_disk:
                    x = x.data_paths
                else:
                    x.to_disk()
                    x = x.data_paths
                data_in = 'disk'

            elif parent_strategy == 'memory':
                if x.in_memory:
                    x = x.data
                else:
                    x.to_memory()
                    x = x.data

        chunks = []
        kwargs_list = []

        is_chunk = (chunksize != 1) or (not squeeze)

        for k, c in self.chunks(x, chunksize=chunksize, n_chunks=n_chunks, squeeze=squeeze):
            chunks.append((c, ))
            kwargs_list.append({'key': k, 'is_chunk': is_chunk,
                                'strategy': worker_strategy, 'data_in': data_in})

        x = parallelize(self.worker, chunks, constant_kwargs=kwargs, kwargs_list=kwargs_list,
                           workers=n_jobs, method='apply_async', collate=False)

        return self.collate(x, **kwargs)





