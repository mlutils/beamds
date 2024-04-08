from typing import Any

import numpy as np
from dataclasses import dataclass

from ..data import BeamData
from ..core import Processor
from ..utils import as_scipy_csr, as_scipy_coo, as_numpy, as_tensor


@dataclass
class Similarities:
    index: Any
    distance: Any
    values: Any = None
    sparse_scores: Any = None
    metric: str = None
    model: str = None


class BeamSimilarity(Processor):

    def __init__(self, *args, metric=None, **kwargs):
        super().__init__(*args, metric=metric, **kwargs)
        self.metric = self.get_hparam('metric', metric)
        self.index = None
        self._is_trained = None
        self.reset()

    @property
    def is_trained(self):
        return self._is_trained

    def reset(self):
        self.index = None
        self._is_trained = False

    @staticmethod
    def extract_data_and_index(x, index=None, convert_to='numpy'):
        if isinstance(x, BeamData) or hasattr(x, 'beam_class') and x.beam_class == 'BeamData':
            index = x.index
            x = x.values

        if convert_to == 'numpy':
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

    def search(self, x, k=1):
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

    def save_state(self, path, ext=None, **kwargs):
        state = {attr: getattr(self, attr) for attr in self.exclude_pickle_attributes}
        state['hparams'] = self.hparams
        bd = BeamData(state, path=path)
        bd.store(**kwargs)

    def load_state(self, path, ext=None, **kwargs):
        bd = BeamData(path=path)
        state = bd.cache(**kwargs).values
        for attr in self.exclude_pickle_attributes:
            setattr(self, attr, state[attr])

    def get_index(self, index):
        return self.index[as_numpy(index)]

    def add_index(self, x, index=None):

        if self.index is None:
            if index is None:
                index = np.arange(len(x))
            else:
                index = as_numpy(index)
            self.index = index
        else:
            if index is None:
                index = np.arange(len(x)) + self.index.max() + 1
            else:
                index = as_numpy(index)
            self.index = np.concatenate([self.index, index])

