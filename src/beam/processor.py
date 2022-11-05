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
from .utils import divide_chunks, collate_chunks
from .multiprocessing import parallelize
from collections import OrderedDict
import os
import fastavro
import pyarrow as pa


class Processor(object):

    def __init__(self, *args, aggregator=False, track_steps=False, pipeline=False, **kwargs):
        self.pipeline = pipeline
        self.aggregator = aggregator
        self.track_statistics = track_steps

        if pipeline or aggregator:
            self.transformers = OrderedDict()
            for i, t in enumerate(args):
                self.transformers[i] = t

            for k, t in kwargs.items():
                self.transformers[k] = t

    @staticmethod
    def pipeline(*ps):
        return Processor(*ps, pipeline=True)

    @staticmethod
    def read(path, **kwargs):

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

    def write(self, x, path):
        raise NotImplementedError


    # def transform(self):
    #
    #     if self.pipeline:
    #
    #         for i, t in self.transformers.items():
    #             x = t.transform(x)
    #             if self.track_statistics:
    #                 self.steps[i] = x
    #         return x

class Reducer(Processor):

    def __init__(self, *args, **kwargs):
        super(Reducer, self).__init__(*args, **kwargs)

    def reduce(self, *args, **kwargs):
        return collate_chunks(list(args), dim=1)


class Transformer(object):

    def __init__(self, *args, n_jobs=0, chunksize=1, **kwargs):

        self.transformers = None
        self.chunksize = chunksize
        self.steps = {}

        self.n_jobs = n_jobs
        self.args = args
        self.kwargs = kwargs

    def _transform(self, x=None, **argv):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        return NotImplementedError

    def transform(self, x, **kwargs):
        return parallelize(self._transform, x, constant_kwargs=kwargs)





