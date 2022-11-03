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
from collections import OrderedDict


class Processor(object):

    def __init__(self, *args, aggregator=False, pipeline=False, **kwargs):
        self.pipeline = pipeline
        self.aggregator = aggregator

        if pipeline or aggregator:
            self.transformers = OrderedDict()
            for i, t in enumerate(args):
                self.transformers[i] = t

            for k, t in kwargs.items():
                self.transformers[k] = t

    def read(self, path):
        raise NotImplementedError

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

    @staticmethod
    def pipeline(*ts):
        return Processor(*ts, pipeline=True)


class Reducer(Processor):

    def __init__(self, *args, **kwargs):
        super(Reducer, self).__init__(*args, **kwargs)

    def reduce(self, *args, **kwargs):
        return collate_chunks(list(args))


class Transformer(object):

    def __init__(self, *args, n_jobs=0, chunksize=1, track_steps=False,
                 **kwargs):

        self.transformers = None
        self.track_statistics = track_steps
        self.chunksize = chunksize
        self.steps = {}

        self.n_jobs = n_jobs
        self.args = args
        self.kwargs = kwargs

    def transform_chunk(self, x=None, **argv):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        return NotImplementedError

    def transform(self, x, **argv):

        if not self.n_jobs:

            chunks = []
            for xi in divide_chunks(x, chunksize=self.chunksize):
                chunks.append(self.transform_chunk(xi, **argv))

            return collate_chunks(chunks)

        return self.__transform__(x)

    @staticmethod
    def pipeline(*ts):
        return Transformer(*ts, pipeline=True)





