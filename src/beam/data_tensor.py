import torch
import pandas as pd
from torch import nn
import warnings
from collections import namedtuple
import numpy as np


class Iloc(object):

    def __init__(self, pointer):
        self.pointer = pointer

    def __getitem__(self, ind):
        return self.pointer._iloc(ind)


class Loc(object):

    def __init__(self, pointer):
        self.pointer = pointer

    def __getitem__(self, ind):
        return self.pointer._loc(ind)


class DataTensor(object):

    def __init__(self, data, columns=None, index=None, requires_grad=False, device=None, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data, **kwargs)

        if device is not None:
            data = data.to(device)

        if requires_grad:
            data = nn.Parameter(data)

        assert len(data.shape) == 2, "DataTensor must be two-dimensional"
        n_rows, n_columns = data.shape

        if columns is None:
            if data.shape[1] == 1:
                columns = ['']
            else:
                columns = [i for i in range(data.shape[1])]

        assert len(columns) == n_columns, "Number of keys must be equal to the tensor 2nd dim"

        self.casting_type = int
        if index is None or type(index) is slice:
            index = torch.arange(n_rows, device=data.device)
        else:
            assert hasattr(index, '__len__') and type(index) is not str, 'index must be a sequnce of strings or numbers'
            if type(index[0]) is str:
                self.casting_type = str

        self.index_map = {self.casting_type(k): i for i, k in enumerate(index)}

        self.index = index
        self.data = data

        self.columns = columns
        self.columns_map = {k: i for i, k in enumerate(columns)}

        self.iloc = Iloc(self)
        self.loc = Loc(self)

    def __len__(self):
        return len(self.index)

    def _loc(self, index):

        if type(index) is slice:
            index = torch.arange(len(self))[index]

        if not (hasattr(index, '__len__') and type(index) is not str):
            index = [index]

        ind = [self.index_map[self.casting_type(i)] for i in index]

        data = self.data[ind]

        return DataTensor(data, columns=self.columns, index=index)

    def _iloc(self, ind):

        if type(ind) is slice:
            ind = torch.arange(len(self))[ind]

        if not (hasattr(ind, '__len__') and type(ind) is not str):
            ind = [ind]

        if issubclass(type(self.index), torch.Tensor):
            index = self.index[ind]
        else:
            index = [self.index[i] for i in ind]

        data = self.data[ind]

        return DataTensor(data, columns=self.columns, index=index)

    def to(self, device):
        self.data = self.data.to(device)

        return self

    def __repr__(self):

        if issubclass(type(self.index), torch.Tensor):
            index = self.index.data.cpu().numpy()
        else:
            index = self.index

        repr_data = repr(pd.DataFrame(data=self.data.detach().cpu().numpy(),
                                      columns=self.columns, index=index))

        inf = f'DataTensor:\ndevice:\t\t{self.data.device}\nrequires_grad:\t{self.data.requires_grad}'

        if self.data.requires_grad and self.data.grad_fn is not None:
            grad_info = f'\ngrad_fn:\t{self.data.grad_fn.name()}'
        else:
            grad_info = ''

        return f'{repr_data}\n\n{inf}{grad_info}'

    def __setitem__(self, ind, data):

        if type(data) is DataTensor:
            data = data.data

        if type(ind) is tuple:

            index = ind[0]
            columns = ind[1]

            if hasattr(index, '__len__') and type(index) is not str:
                ind_index = [self.index_map[self.casting_type(i)] for i in index]

            elif type(index) is not slice:
                ind_index = self.index_map[self.casting_type(index)]
                single_row = True
            else:
                ind_index = index

            if hasattr(columns, '__len__') and type(columns) is not str:
                ind_columns = [self.columns_map[i] for i in columns]
            else:
                ind_columns = self.columns_map[columns]
                single_column = True

            self.data[ind_index, ind_columns] = data

        else:

            columns = ind
            data = torch.cat([self.data, data], dim=1)

            if not (hasattr(columns, '__len__') and type(columns) is not str):
                columns = [columns]

            columns = sum([self.columns, columns], [])
            index = self.index

            self.__init__(data, columns=columns, index=index)

    def __getitem__(self, ind):

        single_row = False
        single_column = False

        if type(ind) is tuple:

            index = ind[0]
            columns = ind[1]

            if hasattr(index, '__len__') and type(index) is not str:
                ind_index = [self.index_map[self.casting_type(i)] for i in index]

            elif type(index) is not slice:
                ind_index = self.index_map[self.casting_type(index)]
                single_row = True
            else:
                ind_index = index

        else:
            columns = ind
            ind_index = slice(None)
            index = self.index

        if hasattr(columns, '__len__') and type(columns) is not str:
            ind_columns = [self.columns_map[i] for i in columns]
        else:
            ind_columns = self.columns_map[columns]
            single_column = True

        data = self.data[slice(None), ind_columns]
        data = data[ind_index]

        if single_column:
            data = data.unsqueeze(1)
            columns = ['']

        if single_row:
            data = data.unsqueeze(0)
            index = [index]

        x = DataTensor(data, columns=columns, index=index)

        return x


prototype = torch.Tensor([0])


def decorator(f_str):
    def apply(x, *args, **kargs):

        f = getattr(x.data, f_str)

        args = list(args)
        for i, a in enumerate(args):
            if type(a) is DataTensor:
                args[i] = a.data

        for k, v in kargs.items():
            if type(v) is DataTensor:
                kargs[k] = v.data

        r = f(*args, **kargs)
        if 'return_types' in str(type(r)):
            data = r.values
        else:
            data = r

        if isinstance(data, torch.Tensor):

            if len(data.shape) == 2:
                n_rows, n_columns = data.shape

            elif len(data.shape) == 1 and len(x.index) != len(x.columns):

                warnings.warn("Warning: Trying to infer columns or index dimensions from the function output")

                if len(x.columns) == len(data):
                    n_columns = len(x.columns)
                    data = data.unsqueeze(0)
                    n_rows = 1

                elif len(x.index) == len(data):
                    n_rows = len(x.index)
                    data = data.unsqueeze(1)
                    n_columns = 1

                else:
                    return r

            else:
                return r

            index = x.index if n_rows == len(x.index) else [f_str]
            columns = x.columns if n_columns == len(x.columns) else [f_str]

            if index is not None or columns is not None:
                data = DataTensor(data, columns=columns, index=index)
                if 'return_types' in str(type(r)):

                    ReturnType = namedtuple(f_str, ['values', 'indices'])
                    r = ReturnType(data, r.indices)

                else:
                    r = data
        return r

    return apply


for p in dir(prototype):
    try:
        f = getattr(prototype, p)
        if callable(f) and p not in dir(DataTensor):
            setattr(DataTensor, p, decorator(p))
    except RuntimeError:
        pass
    except TypeError:
        pass