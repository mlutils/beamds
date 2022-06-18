import torch
import pandas as pd
from torch import nn
import warnings
from collections import namedtuple
import numpy as np
from .utils import check_type, slice_to_array, as_tensor, is_boolean

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
    def __init__(self, data, columns=None, index=None, requires_grad=False, device=None, series=False, **kwargs):
        super().__init__(**kwargs)

        self.series = series

        if isinstance(data, pd.DataFrame):
            columns = data.columns
            index = data.index
            data = data.values
        elif issubclass(type(data), dict):
            columns = list(data.keys())
            data = torch.stack([data[c] for c in columns], dim=1)

        if not isinstance(data, torch.Tensor):
            data = as_tensor(data, **kwargs)

        self.device = data.device if device is None else device

        data = data.to(self.device)

        if requires_grad and not data.requires_grad:
            data.requires_grad_()

        assert len(data.shape) == 2, "DataTensor must be two-dimensional"
        n_rows, n_columns = data.shape

        index_type = check_type(index)
        if index is None:
            index = torch.arange(n_rows)
            self.index_map = None
            self.mapping_method = 'simple'
        elif index_type.major == 'array' and index_type.element == 'int':
            index = as_tensor(index)
            ind = torch.stack([index, torch.zeros_like(index)])

            self.index_map = torch.sparse_coo_tensor(indices=ind, values=torch.arange(n_rows))
            self.mapping_method = 'sparse'
        else:
            self.index_map = {str(k): i for i, k in enumerate(index)}
            self.mapping_method = 'dict'

        columns_type = check_type(columns)
        if columns is None:

            if data.shape[1] == 1:
                columns = ['']
                self.columns_format = 'str'

            else:
                columns = [int(i) for i in torch.arange(n_columns)]
                self.columns_format = 'int'

        elif columns_type.major == 'array' and columns_type.element == 'int':

            columns = [int(i) for i in columns]
            self.columns_format = 'int'

        elif columns_type.major == 'array':

            columns = [str(i) for i in columns]
            self.columns_format = 'str'

        else:
            raise ValueError

        self.columns_map = {str(k): i for i, k in enumerate(columns)}
        assert len(columns) == n_columns, "Number of keys must be equal to the tensor 2nd dim"

        self.index = index
        self.data = data
        self.columns = columns

        self.iloc = Iloc(self)
        self.loc = Loc(self)

    def __len__(self):
        return len(self.index)

    def inverse_columns_map(self, columns):

        cast = int if self.columns_format == 'int' else str

        if check_type(columns).major == 'scalar':
            columns = self.columns_map[cast(columns)]
        else:
            columns = [self.columns_map[cast(i)] for i in columns]

        return columns

    def inverse_map(self, ind):

        ind = slice_to_array(ind, l=self.data.shape[0])
        index_type = check_type(ind)

        if self.mapping_method == 'simple':
            pass
        elif self.mapping_method == 'sparse':

            if index_type.major == 'scalar':
                ind = self.index_map[ind, 0]
            else:
                ind = torch.index_select(self.index_map, 0, ind).coalesce().values()

        elif self.mapping_method == 'dict':
            if index_type.major == 'scalar':
                ind = self.index_map[str(ind)]
            else:
                ind = [self.index_map[str(i)] for i in ind]
        else:
            return NotImplementedError

        return ind

    def apply(self, func, dim=0):

        def remove_dt(d):
            if type(d) is DataTensor:
                return d.sort_values
            return d

        if dim == 1:
            data = torch.concat([remove_dt(func(DataTensor(di.unsqueeze(0), columns=self.columns)))
                                        for di in self.data], dim=0)
        elif dim == 0:
            data = torch.concat([remove_dt(func(DataTensor(di.unsqueeze(0), index=self.index)) )
                                           for di in self.data.T], dim=1)

        return DataTensor(data, columns=self.columns, index=self.index)

    @property
    def values(self):

        data = self.data
        if self.series:
            data = data.squeeze(1)

        return data

    def _iloc(self, ind):

        ind = slice_to_array(ind, l=self.data.shape[0])
        index_type = check_type(ind)

        if index_type.major == 'scalar':
            ind = [ind]

        if self.mapping_method == 'simple':
            index = ind
        elif self.mapping_method == 'sparse':
            index = self.index[ind]
        elif self.mapping_method == 'dict':
            index = [self.index[str(i)] for i in ind]
        else:
            raise NotImplementedError

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

            ind_index = self.inverse_map(index)
            ind_columns = self.inverse_columns_map(columns)

            self.data[ind_index, ind_columns] = data
            return

        else:

            columns = ind

            existing_columns = set(self.columns).difference(columns)
            new_columns = set(columns).difference(self.columns)

            assert not len(existing_columns) * len(new_columns), "Cannot assign new and existing columns in a single operations"

            if len(existing_columns):

                ind_columns = self.inverse_columns_map(columns)
                self.data[:, ind_columns] = data
                return

            if len(existing_columns):

                if check_type(columns).major == 'scalar':

                    data = data.unsqueeze(1)
                    columns = [columns]

                data = torch.cat([self.data, data], dim=1)
                columns = self.columns + columns

                self.__init__(data, columns=columns, index=self.index)

        raise ValueError

    def _loc(self, ind):

        series = False
        if type(ind) is tuple:

            index = ind[0]
            columns = ind[1]

            ind_columns = self.inverse_columns_map(columns)
            if check_type(ind_columns).major == 'scalar':
                ind_columns = [int(ind_columns)]
                columns = [columns]
                series = True

        else:

            index = ind
            columns = self.columns
            ind_columns = slice(None)

        index = slice_to_array(index, l=self.data.shape[0])
        ind_index = self.inverse_map(index)
        if check_type(ind_index).major == 'scalar':
            index = [index]
            ind_index = [int(ind_index)]

        data = self.data[ind_index][slice(None), ind_columns]
        return DataTensor(data, columns=columns, index=index, series=series)


    def __getitem__(self, ind):

        series = False
        ind_type = check_type(ind)

        if (len(ind) == self.data.shape[0]) and is_boolean(ind):

            data = self.data[ind]
            index = self.index[ind]

            return DataTensor(data, columns=self.columns, index=index)

        columns = ind
        ind_columns = self.inverse_columns_map(columns)
        if check_type(ind_columns).major == 'scalar':
            ind_columns = [int(ind_columns)]
            columns = [columns]
            series = True

        data = self.data[slice(None), ind_columns]
        return DataTensor(data, columns=columns, index=self.index, series=series)


prototype = torch.Tensor([0])


def decorator(f_str):
    def apply(x, *args, **kargs):

        f = getattr(x.data, f_str)

        args = list(args)
        for i, a in enumerate(args):
            if type(a) is DataTensor :
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

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for p in dir(prototype):
        try:
            f = getattr(prototype, p)
            if callable(f) and p not in dir(DataTensor):
                setattr(DataTensor, p, decorator(p))
        except RuntimeError:
            pass
        except TypeError:
            pass


