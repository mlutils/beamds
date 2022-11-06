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
from .utils import divide_chunks, collate_chunks, recursive_chunks
from .multiprocessing import parallelize
from collections import OrderedDict
import os
import fastavro
import pyarrow as pa


class Processor(object):

    def __init__(self, *args, root_path=None, **kwargs):
        self.root_path = root_path

    @staticmethod
    def read(path, relative=True, **kwargs):

        _, ext = os.path.splitext(path)

        if ext == 'fea':
            x = pd.read_feather(path, **kwargs)
        elif ext == 'csv':
            x = pd.read_csv(path, **kwargs)
        elif ext in ['pkl', 'pickle']:
            x = pd.read_pickle(path, **kwargs)
        elif ext in ['npy', 'npz']:
            x = np.load(path, **kwargs)
        elif ext == 'parquet':
            x = pd.read_parquet(path, **kwargs)
        elif ext == 'pt':
            x = torch.load(path, **kwargs)
        elif ext in ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt']:
            x = pd.read_excel(path, **kwargs)
        elif ext == 'avro':
            x = []
            with open(path, 'rb') as fo:
                for record in fastavro.reader(fo):
                    x.append(record)
        elif ext == 'json':
            x = []
            with open(path, 'r') as fo:
                for record in fastavro.json_reader(fo):
                    x.append(record)
        elif ext == 'orc':
            x = pa.orc.read(path, **kwargs)

        else:
            raise ValueError("Unknown extension type.")

        return x

    def write(self, x, path, **kwargs):

        _, ext = os.path.splitext(path)

        if ext == 'fea':
            x = pd.DataFrame(x)
            x.to_feather(path, **kwargs)
        elif ext == 'csv':
            x = pd.DataFrame(x)
            x.to_csv(path, **kwargs)
        elif ext in ['pkl', 'pickle']:
            pd.to_pickle(x, path, **kwargs)
        elif ext == 'npy':
            np.save(x, path, **kwargs)
        elif ext == 'npz':
            np.savez(x, path, **kwargs)
        elif ext == 'parquet':
            x = pd.DataFrame(x)
            x.to_parquet(path, **kwargs)
        elif ext == 'pt':
            torch.save(x, path, **kwargs)
        else:
            raise ValueError("Unsupported extension type.")

    # parquet: allow_truncated_timestamps=True, coerce_timestemps='us'


class Pipeline(Processor):

    def __init__(self, *ts, track_steps=False, **kwts):

        super(Pipeline, self).__init__()
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
        super(Reducer, self).__init__(*args, **kwargs)

    def reduce(self, *xs, **kwargs):
        return collate_chunks(list(xs), dim=1)


class Transformer(Processor):

    def __init__(self, *args, n_jobs=0, chunksize=1, **kwargs):

        super(Transformer, self).__init__(*args, **kwargs)
        self.transformers = None
        self.chunksize = chunksize
        self.n_jobs = n_jobs
        self.args = args
        self.kwargs = kwargs

    def chunks(self, x):
        for c in recursive_chunks(x, chunksize=self.chunksize):
            yield c

    def _transform(self, x, **kwargs):
        raise NotImplementedError

    def fit(self, x, **kwargs):
        return NotImplementedError

    def collate(self, x, **kwargs):
        return NotImplementedError

    def transform(self, x, **kwargs):

        x = parallelize(self._transform, list(self.chunks(x)), constant_kwargs=kwargs,
                           workers=self.n_jobs, method='apply_async')

        return self.collate(x, **kwargs)





