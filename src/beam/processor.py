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


class Processor(object):

    def __init__(self, *args, root_dir=None, path_to_data=None, **kwargs):
        self.root_dir = root_dir
        self.path_to_data = path_to_data

    def read_file(self, path=None, relative=True, **kwargs):

        if path is None:
            path = self.root_dir
        elif relative:
            path = os.path.join(self.root_dir, path)

        _, ext = os.path.splitext(path)

        if ext == '.fea':
            x = pd.read_feather(path, **kwargs)

            c = x.columns
            for ci in c:
                if '/feather_index' in c:
                    index_name = ci.split('/feather_index')[0]
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

    def write_file(self, x, path, relative=True, **kwargs):

        if relative:
            path = os.path.join(self.root_dir, path)

        _, ext = os.path.splitext(path)

        if ext == '.fea':
            x = pd.DataFrame(x)

            index_name = x.index.name if x.index.name is not None else 'index'
            df = x.reset_index()
            new_name = index_name + '/feather_index'
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




    def write(self, x, path, root=True, relative=True, compress=None, chunksize=int(1e9),
              chunklen=None, n_chunks=None, partition=None,  **kwargs):

        if root:
            if (n_chunks is None) and (chunklen is None):
                max_size = recursive_size(x, mode='max')
                n_chunks = int(np.round(max_size / chunksize))
            elif (n_chunks is not None) and (chunklen is not None):
                logger.warning("processor.write requires only one of chunklen|n_chunks. Defaults to using n_chunks")
            elif n_chunks is None:
                n_chunks = int(np.round(recursive_len(x) / chunklen))

        if relative:
            path = os.path.join(self.root_dir, path)

        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
            os.rmdir(path)

        x_type = check_type(x)

        if x_type.major == 'container':
            os.mkdir(path)
            for k, v in iter_container(x):
                self.write(k, os.path.join(path, str(k)), relative=relative, root=False, compress=compress,  **kwargs)
        else:

            #TODO: divide into chunks before saving
            x = divide_chunks(x, c)

            if partition is not None and x_type.minor == 'pandas':
                order = ['.parquet', '.fea', '.pkl']
            elif x_type.minor in ['pandas', 'numpy']:
                order = ['.fea', '.parquet', '.pkl']
            elif x_type.minor == 'tensor':
                order = ['.pt']
            else:
                order = ['.pkl']

            path = pathlib.Path(path)
            for ext in order:
                file_path = path.with_suffix(ext)
                try:
                    kwargs = {}
                    if ext == '.parquet':
                        if compress is False:
                            kwargs['compression'] = None

                        self.write_file(x, file_path, partition=partition, coerce_timestamps='us',
                                        allow_truncated_timestamps=True, **kwargs)
                    elif ext == '.fea':
                        if compress is False:
                            kwargs['compression'] = 'uncompressed'
                        self.write_file(x, file_path, **kwargs)

                    elif ext == '.pkl':
                        if compress is False:
                            kwargs['compression'] = 'none'
                        self.write_file(x, file_path, **kwargs)

                    elif ext == '.pt':
                        self.write_file(x, file_path, **kwargs)
                    else:
                        raise NotImplementedError

                    error = False

                except:
                    logger.warning(f"Failed to write file: {file_path.name}. Trying with the next file extension")
                    error = True

            if error:
                logger.error(f"Could not write file: {path.name}.")


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

    def __init__(self, *args, n_jobs=0, n_chunks=1, chunksize=None, **kwargs):

        super(Transformer, self).__init__(*args, **kwargs)
        self.transformers = None
        self.chunksize = chunksize
        self.n_chunks = n_chunks
        self.n_jobs = n_jobs
        self.args = args
        self.kwargs = kwargs

    def chunks(self, x):
        for c in recursive_chunks(x, chunksize=self.chunksize, n_chunks=self.n_chunks):
            yield c

    def _transform(self, x, **kwargs):
        raise NotImplementedError

    def fit(self, x, **kwargs):
        return NotImplementedError

    def fit_transform(self, x, **kwargs):
        self.fit(x, **kwargs)
        return self.transform(x, **kwargs)

    def collate(self, x, **kwargs):
        return NotImplementedError

    def transform(self, x, **kwargs):

        x = parallelize(self._transform, list(self.chunks(x)), constant_kwargs=kwargs,
                           workers=self.n_jobs, method='apply_async')

        return self.collate(x, **kwargs)





