from typing import Any
from dataclasses import dataclass

from .. import BeamData
from ..core import Processor


@dataclass
class Similarities:
    index: Any
    distance: Any
    values: Any = None
    sparse_scores: Any = None


class BeamSimilarity(Processor):

    def __init__(self, *args, metric=None, **kwargs):
        super().__init__(*args, metric=metric, **kwargs)
        self.metric = self.get_hparam('metric', metric)

    @property
    def is_trained(self):
        return False

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
        raise NotImplementedError

    def __len__(self):
        return self.ntotal

    def save_state(self, path, ext=None, **kwargs):
        state = {attr: getattr(self, attr) for attr in self.state_attributes}
        state['hparams'] = self.hparams
        bd = BeamData(state, path=path)
        bd.store(**kwargs)

    def load_state(self, path, ext=None, **kwargs):
        bd = BeamData(path=path)
        state = bd.cache(**kwargs).values
        for attr in self.state_attributes:
            setattr(self, attr, state[attr])

