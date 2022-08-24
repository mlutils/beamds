
import torch
from .utils import check_type, slice_to_index, as_tensor
import pandas as pd
from .data_tensor import DataTensor


class PackedFolds(object):

    def __init__(self, data, index=None, names=None, fold=None, fold_index=None, device=None,
                 sort_index=False, quick_getitem=True):

        self.quick_getitem = quick_getitem
        self.names_dict = None
        self.names = None
        self.sampling_method = 'folds'

        if names is not None:
            self.names = names
            self.names_dict = {n: i for i, n in enumerate(self.names)}

        data_type = check_type(data)

        if data_type.minor == 'list':
            self.data = [as_tensor(v, device=device) for v in data]

        elif data_type.minor == 'dict':
            if names is None:
                self.names = list(data.keys())
                self.names_dict = {n: i for i, n in enumerate(self.names)}

            self.data = [as_tensor(data[k], device=device) for k in self.names]

        elif data_type.major == 'array':
            self.data = as_tensor(data, device=device)
            self.sampling_method = 'no_folds'
        else:
            raise ValueError("data should be either dict/list/array")

        fold_type = check_type(fold)

        if fold_type.element == 'str':
            fold, names_map = pd.factorize(fold)

            if self.names_dict is not None:
                assert all([i == self.names_dict[n] for i, n in enumerate(names_map)]), "fold and data maps must match"

            else:
                self.names = list(names_map)
                self.names_dict = {n: i for i, n in enumerate(self.names)}

            fold = as_tensor(fold)

            if self.sampling_method == 'no_folds':
                self.sampling_method = 'foldable'

        elif fold_type.element == 'int':
            fold = as_tensor(fold)

            if self.sampling_method == 'no_folds':
                self.sampling_method = 'foldable'

        elif fold is None:
            if self.sampling_method == 'no_folds':
                assert len(names) == 1, "this is the single fold case"
                fold = torch.zeros(len(self.data), dtype=torch.int64)
                self.sampling_method = 'offset'
            else:
                fold = torch.cat([i * torch.ones(len(d), dtype=torch.int64) for i, d in enumerate(self.data)])

        else:
            raise ValueError

        if data_type.minor in ['dict', 'list']:
            lengths = torch.LongTensor([len(di) for di in self.data])
        else:
            lengths = torch.bincount(fold)

        if fold_index is not None:
            fold_index = as_tensor(fold_index)
        else:
            fold_index = torch.cat([torch.arange(l) for l in lengths])

        # update names
        if self.names is None:
            if data_type.minor in ['dict', 'list']:
                self.names = list(range(len(self.data)))
            else:
                self.names = torch.sort(torch.unique(fold)).values

            self.names_dict = {n: i for i, n in enumerate(self.names)}

        # merge data if possible for faster slicing
        if data_type.minor in ['dict', 'list']:
            dtype = [di.dtype for di in self.data]
            shape = [di.shape[1:] for di in self.data]
            if all([d == dtype[0] for d in dtype]) and all([s == shape[0] for s in shape]):
                self.data = torch.cat(self.data)
                self.sampling_method = 'offset'

        index_type = check_type(index)

        if index_type.minor == 'list':
            index = torch.concat([as_tensor(v, return_vector=True) for v in index])

        elif index_type.minor == 'dict':
            index = torch.concat([as_tensor(index[k], return_vector=True) for k in self.names])

        elif index_type.major == 'array':
            index = as_tensor(index)

        elif index is None:

            if self.sampling_method in ['offset', 'foldable']:
                index = torch.arange(len(self.data))
                self.sampling_method = 'index'
            else:
                index = torch.arange(sum([len(d) for d in self.data]))

        else:
            raise ValueError

        cumsum = torch.cumsum(lengths, dim=0)
        offset = cumsum - lengths
        offset = offset[fold] + fold_index

        info = {'fold': fold, 'fold_index': fold_index, 'offset': offset}

        if self.sampling_method == 'index':
            index = None

        self.info = DataTensor(info, index=index, device=device)

        if sort_index:
            self.sort_index()

    def sort_index(self, ascending=True):

        self.info = self.info.sort_index(ascending=ascending)

        return self

    def __len__(self):
        return len(self.info)

    def get_fold(self, name):

        fold = self.names_dict[name]
        info = self.info[self.info['fold'] == fold]
        index = info.index

        if self.sampling_method == 'folds':
            data = self.data[fold]
        elif self.sampling_method == 'index':
            data = self.data[index]
        elif self.sampling_method == 'offset':
            data = self.data[info['offset'].values]
        else:
            raise Exception(f"Sampling method unsupported: {self.sampling_method}")

        return PackedFolds(data=data, index=index, names=[name], device=self.device)

    def apply(self, functions):

        functions_type = check_type(functions)

        if functions_type.minor == 'list':
            data = [f(d) for d, f in zip(self.data, functions)]

        elif functions_type.minor == 'dict':
            data = [f(self.get_fold(k)) for k, f in functions.items()]
        else:
            raise ValueError

        return PackedFolds(data=data, index=self.index, names=self.names, device=self.device)

    @property
    def index(self):
        return self.info.index

    @property
    def fold(self):
        return self.info['fold'].values

    @property
    def fold_index(self):
        return self.info['fold_index'].values

    @property
    def offset(self):
        return self.info['offset'].values

    @property
    def shape(self):

        if self.sampling_method == 'fold':
            shape = {k: d.shape for k, d in zip(self.names, self.data)}
        else:
            shape = self.data.shape
        return shape

    @property
    def values(self):

        if self.sampling_method == 'fold':
            data = torch.cat(self.data)
        else:
            data = self.data

        return data[self.info['offset'].values]

    @property
    def tag(self):
        if len(self.names) == 1:
            return self.names[0]
        return 'hetrogenous'

    @property
    def device(self):
        if self.sampling_method == 'folds':
            device = self.data[0].device
        else:
            device = self.data.device

        return device

    def to(self, device):

        if self.sampling_method == 'folds':
            self.data = [di.to(device) for di in self.data]
        else:
            self.data = self.data.to(device)

        self.info = self.info.to(device)

        return self

    def __repr__(self):
        if self.sampling_method == 'folds':
            data = {k: self.data[self.names_dict[k]] for k in self.names}
        else:
            data = {k: self.get_fold(k).data for k in self.names}

        if len(data) == 1:
            data = next(iter(data.values()))

        return repr(data)

    def __getitem__(self, ind):

        ind_type = check_type(ind, check_minor=False)
        if ind_type.major == 'scalar' and ind_type.element == 'str':
            return self.get_fold(ind)
        if self.sampling_method == 'index' and self.quick_getitem:
            return self.data[ind]

        if type(ind) is tuple:
            ind_rest = ind[1:]
            ind = ind[0]
        else:
            ind_rest = tuple()

        ind = slice_to_index(ind, l=len(self), sliced=self.index)
        if ind_type.major == 'scalar':
            ind = [ind]

        info = self.info.loc[ind]

        if self.sampling_method == 'folds':
            fold, fold_index = info[['fold', 'fold_index']].values.T

            uq = torch.sort(torch.unique(fold)).values
            names = [self.names[i] for i in uq]

            if len(uq) == 1:

                data = self.data[uq[0]].__getitem__((fold_index, *ind_rest))
                if len(ind) == 1:
                    return data[0]

                return PackedFolds(data=[data], names=names, index=ind, device=self.device)

            fold_index = [fold_index[fold == i] for i in uq]
            data = [self.data[i].__getitem__((j, *ind_rest)) for i, j in zip(uq, fold_index)]

            fold = None

        else:

            fold, offset = info[['fold', 'offset']].values.T
            index = info.index
            ind = index if self.sampling_method == 'index' else offset

            data = self.data.__getitem__((ind, *ind_rest))

            uq = torch.sort(torch.unique(fold)).values
            names = [self.names[i] for i in uq]

            if self.sampling_method == 'index':
                ind = None
            else:
                ind = index

        return PackedFolds(data=data, names=names, index=ind, fold=fold, device=self.device)