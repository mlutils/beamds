from typing import Any

import numpy as np
from dataclasses import dataclass

from ..data import BeamData
from ..processor import Processor
from ..type.utils import is_beam_data
from ..utils import as_scipy_csr, as_scipy_coo, as_numpy, as_tensor


@dataclass
class Similarities:
    index: Any
    distance: Any
    sparse_scores: Any = None
    metric: str = None
    model: str = None


class BeamSimilarity(Processor):

    def __init__(self, *args, metric=None, **kwargs):
        super().__init__(*args, metric=metric, **kwargs)
        self.metric = self.hparams.metric
        self.index = None
        self._is_trained = None
        self._is_range_index = None
        self._is_numeric_index = None
        self.reset()

    @property
    def is_trained(self):
        return self._is_trained

    def reset(self):
        self.index = np.array([])
        self._is_trained = False

    @staticmethod
    def extract_data_and_index(x, index=None, convert_to='numpy'):
        if is_beam_data(x):
            index = x.index
            x = x.values

        if convert_to is None:
            pass
        elif convert_to == 'numpy':
            x = as_numpy(x)
        elif convert_to == 'tensor':
            x = as_tensor(x)
        elif convert_to == 'scipy_csr':
            x = as_scipy_csr(x)
        elif convert_to == 'scipy_coo':
            x = as_scipy_coo(x)
        else:
            raise ValueError(f"Unknown conversion: {convert_to}")

        return x, as_numpy(index)

    @property
    def metric_type(self):
        return self.metric

    def add(self, x, index=None, **kwargs):
        raise NotImplementedError

    def search(self, x, k=1) -> Similarities:
        raise NotImplementedError

    def train(self, x):
        raise NotImplementedError

    def remove_ids(self, ids):
        raise NotImplementedError

    def reconstruct(self, id0):
        raise NotImplementedError

    def reconstruct_n(self, id0, id1):
        raise NotImplementedError

    @property
    def ntotal(self):
        if self.index is not None:
            return len(self.index)
        return 0

    def __len__(self):
        return self.ntotal

    def get_index(self, index):
        return self.index[as_numpy(index)]

    def add_index(self, x, index=None):

        if self.index is None or not len(self.index):
            if index is None:
                try:
                    l = len(x)
                except TypeError:
                    l = x.shape[0]
                index = np.arange(l)
                self._is_range_index = True
                self._is_numeric_index = True
            else:
                index = as_numpy(index)
                self._is_range_index = False
                if index.dtype.kind in 'iuf':
                    self._is_numeric_index = True
            self.index = index
        else:
            if index is None:
                index = np.arange(len(x)) + self.index.max() + 1
            else:
                index = as_numpy(index)
            self.index = np.concatenate([self.index, index])

        return index

