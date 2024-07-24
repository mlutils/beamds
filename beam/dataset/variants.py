import warnings
import torch

from ..utils import check_type, as_tensor, as_numpy
from ..utils import check_type
from .universal_dataset import UniversalDataset
from ..type import Types


class LazyReplayBuffer(UniversalDataset):

    def __init__(self, size, device='cpu'):
        self.max_size = size
        self.size = 0
        self.ptr = 0
        self._target_device = device
        super().__init__(device=device)

    def build_buffer(self, x):
        return torch.zeros(self.size, *x.shape, device=self.target_device, dtype=x.dtype)

    def store(self, *args, **kwargs):

        if len(args) == 1:
            d = args[0]
        elif len(args):
            d = args
        else:
            d = kwargs

        if self.data is None:
            if isinstance(d, dict):
                self.data = {k: self.build_buffer(v) for k, v in d.items()}
                self._data_type = 'dict'
            elif isinstance(d, list) or isinstance(d, tuple):
                self.data = [self.build_buffer(v) for v in d]
                self._data_type = 'list'
            else:
                self.data = self.build_buffer(d)
                self._data_type = 'simple'

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.data_type == 'dict':
                for k, v in d.items():
                    self.data[k][self.ptr] = as_tensor(v, device=self.target_device)
            elif self.data_type == 'list':
                for i, v in enumerate(self.data):
                    self.data[i][self.ptr] = as_tensor(v, device=self.target_device)
            else:
                self.data[self.ptr] = as_tensor(d, device=self.target_device)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def reset(self):
        self.ptr = 0
        self.data = None
        self.size = 0

    def __len__(self):
        return self.size


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, alg, *args, **kwargs):
        super().__init__()

        if type(dataset) != UniversalDataset:
            dataset = UniversalDataset(dataset)

        self.dataset = dataset
        self.alg = alg
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, ind):

        ind_type = check_type(ind, element=False)
        if ind_type.major == Types.scalar:
            ind = [ind]

        ind, data = self.dataset[ind]
        dataset = UniversalDataset(data)
        res = self.alg.predict(dataset, *self.args, **self.kwargs)

        return ind, res.values
