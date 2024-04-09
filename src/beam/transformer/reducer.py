from ..processor.core import Processor
from ..utils import collate_chunks


class Reducer(Processor):

    def __init__(self, hparams, *args, dim=1, **kwargs):
        super().__init__(hparams, *args, **kwargs)
        self.dim = dim

    def reduce(self, *xs, **kwargs):
        return collate_chunks(*xs, dim=self.dim, **kwargs)
