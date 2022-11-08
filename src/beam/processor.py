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
    recursive_size, recursive_len
from .parallel import parallelize
from collections import OrderedDict
import os
import fastavro
import pyarrow as pa
import shutil
import pathlib
from argparse import Namespace


class Processor(object):

    def __init__(self, *args, **kwargs):
        pass


class BeamData(object):

    feather_index_mark = "index:"

    def __init__(self, *args, path=None, **kwargs):
        self.root_path = path
        if len(args) == 0:
            arg_type = check_type(args[0])
            if arg_type.major == 'container':
                self.data = args[0]
        else:
            assert len(args) * len(kwargs) == 0, "Please use either args or kwargs"

        if len(args):
            self.data = list(args)
        else:
            self.data = kwargs

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
            x = np.load(path, **kwargs)
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
    def write_file(x, path, **kwargs):

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
            np.save(x, path, **kwargs)
        elif ext == '.npz':
            np.savez(x, path, **kwargs)
        elif ext == '.parquet':
            x = pd.DataFrame(x)
            x.to_parquet(path, **kwargs)
        elif ext == '.pt':
            torch.save(x, path, **kwargs)
        else:
            raise ValueError("Unsupported extension type.")

    def write(self, x=None, path=None, root=True, relative=True, compress=None, chunksize=int(1e9),
              chunklen=None, n_chunks=None, partition=None, file_type=None, **kwargs):

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

            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)

            os.makedirs(path, exist_ok=True)
            os.rmdir(path)

        x_type = check_type(x)

        if x_type.major == 'container':
            os.mkdir(path)

            file_type_type = check_type(file_type)

            for k, v in iter_container(x):

                if file_type_type.major == 'container':
                    ft = file_type[k]
                else:
                    ft = file_type

                self.write(v, os.path.join(path, str(k)), relative=relative, n_chunks=n_chunks,
                           root=False, compress=compress, file_type=ft, **kwargs)
        else:

            if partition is not None and x_type.minor == 'pandas':
                order = ['.parquet', '.fea', '.pkl']
            elif x_type.minor in ['pandas', 'numpy']:
                order = ['.fea', '.parquet', '.pkl']
            elif x_type.minor == 'tensor':
                order = ['.pt']
            else:
                order = ['.pkl']

            if file_type is not None:
                order.insert(file_type, 0)

            x = list(divide_chunks(x, n_chunks=n_chunks))

            if len(x) > 1:
                os.mkdir(path)

            for i, xi in enumerate(x):

                if len(x) > 1:
                    path_i = pathlib.Path(os.path.join(path, f"{i:06}"))
                else:
                    path_i = pathlib.Path(path)

                for ext in order:
                    file_path = path_i.with_suffix(ext)
                    try:
                        kwargs = {}
                        if ext == '.parquet':
                            if compress is False:
                                kwargs['compression'] = None
                            self.write_file(xi, file_path, partition=partition, coerce_timestamps='us',
                                            allow_truncated_timestamps=True, **kwargs)
                        elif ext == '.fea':
                            if compress is False:
                                kwargs['compression'] = 'uncompressed'
                            self.write_file(xi, file_path, **kwargs)

                        elif ext == '.pkl':
                            if compress is False:
                                kwargs['compression'] = 'none'
                            self.write_file(xi, file_path, **kwargs)

                        else:
                            self.write_file(xi, file_path, **kwargs)

                        error = False
                        order = [ext]
                        break

                    except:
                        logger.warning(f"Failed to write file: {file_path.name}. Trying with the next file extension")
                        error = True

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

    def __init__(self, *args, state=None, n_jobs=0, n_chunks=None, chunksize=None, **kwargs):

        super(Transformer, self).__init__(*args, **kwargs)

        if (n_chunks is None) and (chunksize is None):
            n_chunks = 1

        self.transformers = None
        self.chunksize = chunksize
        self.n_chunks = n_chunks
        self.n_jobs = n_jobs
        self.state = state
        self.kwargs = kwargs

    def chunks(self, x):
        for c in recursive_chunks(x, chunksize=self.chunksize, n_chunks=self.n_chunks):
            yield c

    def _transform(self, x, index=None, **kwargs):
        raise NotImplementedError

    def fit(self, x, **kwargs):
        return NotImplementedError

    def fit_transform(self, x, **kwargs):
        self.fit(x, **kwargs)
        return self.transform(x, **kwargs)

    def collate(self, x, **kwargs):
        return collate_chunks(*x, dim=0, **kwargs)

    def transform(self, x, **kwargs):

        chunks = list(self.chunks(x))
        chunks = [(c,) for c in chunks]
        kwargs_list = [{'index': i} for i in range(len(chunks))]

        x = parallelize(self._transform, chunks, constant_kwargs=kwargs, kwargs_list=kwargs_list,
                           workers=self.n_jobs, method='apply_async')

        return self.collate(x, **kwargs)





