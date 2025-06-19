import warnings
import torch

from ..utils import check_type, as_tensor, as_numpy
from ..utils import check_type
from .universal_dataset import UniversalDataset
from ..type import Types


class LazyReplayBuffer(UniversalDataset):

    def __init__(self, size, *args, device='cpu', **kwargs):
        super().__init__(*args, device=device, **kwargs)
        self.max_size = size
        self.size = 0
        self.ptr = 0
        self._target_device = device

    def build_buffer(self, x):
        if x is None:
            return None
        return torch.zeros(self.max_size, *x.shape, device=self.target_device, dtype=x.dtype)

    def build_buffer_from_batch(self, x):
        if x is None:
            return None
        return torch.zeros(self.max_size, *x.shape[1:], device=self.target_device, dtype=x.dtype)

    def store(self, *args, **kwargs):

        if len(args) == 1:
            d = args[0]
        elif len(args):
            d = args
        else:
            d = kwargs

        if self.data is None:
            self._data_type = check_type(d)
            if self.data_type.minor == Types.dict:
                self.data = {k: self.build_buffer(v) for k, v in d.items()}
            elif self.data_type.minor in [Types.list, Types.tuple]:
                self.data = [self.build_buffer(v) for v in d]
            else:
                self.data = self.build_buffer(d)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.data_type.minor == Types.dict:
                for k, v in d.items():
                    if v is not None:
                        self.data[k][self.ptr] = as_tensor(v, device=self.target_device)
            elif self.data_type.minor in [Types.list, Types.tuple]:
                for i, v in enumerate(self.data):
                    if v is not None:
                        self.data[i][self.ptr] = as_tensor(v, device=self.target_device)
            else:
                self.data[self.ptr] = as_tensor(d, device=self.target_device)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def store_batch(self, *args, **kwargs):
        if len(args) == 1:
            d = args[0]
        elif len(args):
            d = args
        else:
            d = kwargs

        if self.data is None:
            self._data_type = check_type(d)
            if self.data_type.minor == Types.dict:
                self.data = {k: self.build_buffer_from_batch(v) for k, v in d.items()}
            elif self.data_type.minor in [Types.list, Types.tuple]:
                self.data = [self.build_buffer_from_batch(v) for v in d]
            else:
                self.data = self.build_buffer_from_batch(d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.data_type.minor == Types.dict:
                for k, v in d.items():
                    if v is not None:
                        v = as_tensor(v, device=self.target_device)
                        n = len(v)
                        # handle wrap around
                        self.data[k][self.ptr:min(self.ptr + n, self.max_size)] = v[:self.max_size - self.ptr]
                        if self.ptr + n > self.max_size:
                            self.data[k][:n - (self.max_size - self.ptr)] = v[self.max_size - self.ptr:]

            elif self.data_type.minor in [Types.list, Types.tuple]:
                for i, v in enumerate(self.data):
                    if v is not None:
                        v = as_tensor(v, device=self.target_device)
                        n = len(v)
                        # handle wrap around
                        self.data[i][self.ptr:min(self.ptr + n, self.max_size)] = v[:self.max_size - self.ptr]
                        if self.ptr + n > self.max_size:
                            self.data[i][:n - (self.max_size - self.ptr)] = v[self.max_size - self.ptr:]
            else:
                d = as_tensor(d, device=self.target_device)

                # handle wrap around
                n = len(d)
                self.data[self.ptr:min(self.ptr + n, self.max_size)] = d[:self.max_size - self.ptr]
                if self.ptr + n > self.max_size:
                    self.data[:n - (self.max_size - self.ptr)] = d[self.max_size - self.ptr:]

        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)


    def reset(self):
        self.ptr = 0
        self.data = None
        self.size = 0
        self._data_type = None

    def __len__(self):
        return self.size

    def __getitem__(self, ind):
        if self.size == 0:
            raise IndexError("Cannot index empty buffer")

        # Handle single integer index
        if isinstance(ind, int):
            # Convert negative indices to positive
            if ind < 0:
                ind = ind % self.size

            # Check bounds
            if ind >= self.size:
                raise IndexError(f"Index {ind} out of range for buffer of size {self.size}")

            # Calculate actual index in circular buffer relative to ptr
            actual_idx = (self.ptr + ind) % self.max_size

            # Return data at this index
            if self.data_type.minor == Types.dict:
                return {k: v[actual_idx] if v is not None else None for k, v in self.data.items()}
            elif self.data_type.minor in [Types.list, Types.tuple]:
                result = [v[actual_idx] if v is not None else None for v in self.data]
                return result if self.data_type.minor == Types.list else tuple(result)
            else:
                return self.data[actual_idx]

        # Handle slicing
        elif isinstance(ind, slice):
            # Convert slice to indices
            start, stop, step = ind.indices(self.size)

            # Generate list of indices
            ind = as_tensor(list(range(start, stop, step)))

            if len(ind) == 0:
                # Return empty structure matching data type
                if self.data_type.minor == Types.dict:
                    return {k: v[0:0] if v is not None else None for k, v in self.data.items()}
                elif self.data_type.minor in [Types.list, Types.tuple]:
                    result = [v[0:0] if v is not None else None for v in self.data]
                    return result if self.data_type.minor == Types.list else tuple(result)
                else:
                    return self.data[0:0]

        elif isinstance(ind, torch.Tensor) and ind.dtype in [torch.int32, torch.int64]:
            pass

        else:
            raise TypeError(f"Unsupported index type: {type(ind)}. Expected int, slice, or tensor of indices.")

        if self.size == self.max_size:
            actual_indices = (ind + self.ptr) % self.max_size
        else:
            actual_indices = ind

        # Gather data at these indices
        if self.data_type.minor == Types.dict:
            return {k: v[actual_indices] if v is not None else None for k, v in self.data.items()}
        elif self.data_type.minor in [Types.list, Types.tuple]:
            result = [v[actual_indices] if v is not None else None for v in self.data]
            return result if self.data_type.minor == Types.list else tuple(result)
        else:
            return self.data[actual_indices]


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
