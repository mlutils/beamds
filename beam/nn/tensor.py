import pickletools
import warnings
from collections import namedtuple

import numpy as np
import pandas as pd
import torch

from ..path import beam_path
from ..utils import check_type, slice_to_index, as_tensor, is_boolean, as_numpy, beam_device
from ..type import Types
from ..base import Loc, Iloc


class LazyTensor:

    def __init__(self, path, device=None):
        self.path = beam_path(path)
        self._device = device
        self._dtype = None
        self._shape = None
        self._header = None
        self._element_size = None
        self.offset = 0x140

    @property
    def header(self):
        if self._header is None:
            with self.path.open('rb') as fo:

                fo.seek(0x40)
                frames = list(pickletools.genops(fo))
                self._header = pd.DataFrame([{'name': op.name, 'val': v, 'index': p, 'proto': op.proto,
                                              'arg_name': op.arg.name if op.arg is not None else None,
                                              'arg_n': op.arg.n if op.arg is not None else None} for op, v, p in frames])

        return self._header

    @property
    def dtype(self):
        if self._dtype is None:
            dtype = self.header[self.header.name == 'GLOBAL'].iloc[1].val
            types_dict = {'torch LongStorage': torch.int64,
                          'torch IntStorage': torch.int32,
                          'torch FloatStorage': torch.float32,
                          'torch DoubleStorage': torch.float64,
                          'torch ByteStorage': torch.uint8,
                          'torch CharStorage': torch.int8,
                          'torch ShortStorage': torch.int16,
                          'torch HalfStorage': torch.float16,
                          'torch BoolStorage': torch.bool,
                          'torch ComplexFloatStorage': torch.complex64,
                          'torch ComplexDoubleStorage': torch.complex128,
                          'torch BFloat16Storage': torch.bfloat16,
                          }
            self._dtype = types_dict[dtype]
        return self._dtype

    @property
    def shape(self):
        if self._shape is None:
            mark_index = self.header[self.header.name == 'MARK'].iloc[2].name
            df = self.header.iloc[mark_index + 1:]
            self._shape = as_tensor(df.loc[:(df.name == 'TUPLE').idxmax() - 1].val.values.astype(np.int))

        return self._shape

    def __len__(self):
        return self.shape[0]

    def read_buffer(self, fo, offset, shape):

        count = shape.prod() if len(shape) > 0 else 1
        fo.seek(offset)
        mv = memoryview(fo.read(count * self.element_size))
        data = torch.frombuffer(mv, dtype=self.dtype)
        if len(shape) == 0:
            assert len(data) == 1, "Data length should be 1 in this case"
            data = data[0]
        else:
            data = data.reshape(*shape)

        if self.device is not None:
            data = data.to(self.device)

        return data

    @property
    def device(self):
        if self._device is None:
            self._device = beam_device(self.header[self.header.name == 'BINUNICODE'].iloc[2].val)
        return self._device

    @property
    def element_size(self):
        if self._element_size is None:
            self._element_size = torch.tensor([], dtype=self.dtype).element_size()
        return self._element_size

    @staticmethod
    def is_contiguous(ind):

        ind_type = check_type(ind)
        if ind_type.major == Types.slice:
            return ind.step is None or ind.step == 1
        elif ind_type.major == Types.array:
            ind = as_tensor(ind)
            return bool(torch.all(torch.diff(ind) == 1))
        elif ind_type.major == Types.scalar and ind_type.minor == Types.int:
            return True

        return False

    def __getitem__(self, item):

        item_type = check_type(item)
        if item_type.minor != Types.tuple:
            item = (item,)

        offset = self.offset

        for i, ind in enumerate(item):
            ind_type = check_type(ind)
            if ind_type.major == Types.scalar:
                assert ind_type.element == Types.int, "Index must be integer"
                count = self.shape[i + 1:].prod() if i < len(self.shape) - 1 else 1
                offset += ind * self.element_size * count
            else:
                break

        ind = item[i]
        skip = 0

        if not LazyTensor.is_contiguous(ind):
            ind = as_tensor(slice_to_index(ind, l=len(self)))
            count = self.shape[i + 1:].prod() if i < len(self.shape) - 1 else 1
            # in case ind is scalar we expand it to 1D
            ind = ind.unsqueeze(0) if ind.dim() == 0 else ind
            offsets = offset + ind * self.element_size * count
            shape = self.shape[i + 1:]
            i += 1
            skip += 1
        else:
            offsets = torch.LongTensor([offset])
            shape = self.shape[i:]

        if i < len(item):
            ind = item[i]
            if LazyTensor.is_contiguous(ind):
                ind = slice_to_index(ind, l=len(self))
                count = shape[1:].prod() if len(shape[1:]) else 1
                offsets = offsets + ind[0] * self.element_size * count
                shape = torch.cat([torch.tensor([len(ind)]), shape[1:]])
                skip += 1

        with self.path.open('rb') as fo:
            data = [self.read_buffer(fo, offset, shape) for offset in offsets]

        if len(data) > 1:
            data = torch.stack(data)
        else:
            data = data[0]

        if i < len(item) - 1:
            index = (*([slice(None)] * skip), *item[i + 1:])
            data = data[index]

        return data


class DataTensor(object):
    def __init__(self, data, columns=None, index=None, requires_grad=False, device=None, series=False, **kwargs):
        super().__init__(**kwargs)

        self.series = series

        if isinstance(data, pd.DataFrame):
            columns = data.columns
            index = data.index
            data = data.values
        elif isinstance(data, dict):
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
        else:
            if index_type.minor == Types.tensor:
                index = as_numpy(index)
            elif index_type.major == Types.scalar:
                index = [index]

            self.index_map = pd.Series(index=index, data=np.arange(len(index)))
            self.mapping_method = 'series'

        columns_type = check_type(columns)
        if columns is None:

            if data.shape[1] == 1:
                columns = ['']
                self.columns_format = 'str'

            else:
                columns = [int(i) for i in torch.arange(n_columns)]
                self.columns_format = 'int'

        elif columns_type.major == Types.array and columns_type.element == Types.int:

            columns = [int(i) for i in columns]
            self.columns_format = 'int'

        elif columns_type.major == Types.array:

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

    def __eq__(self, other):
        return self.values.__eq__(other)

    def __ge__(self, other):
        return self.values.__ge__(other)

    def __ne__(self, other):
        return self.values.__ne__(other)

    def __lt__(self, other):
        return self.values.__lt__(other)

    def __gt__(self, other):
        return self.values.__gt__(other)

    def __le__(self, other):
        return self.values.__le__(other)

    def __len__(self):
        return len(self.index)

    def inverse_columns_map(self, columns):

        cast = int if self.columns_format == 'int' else str

        if check_type(columns).major == Types.scalar:
            columns = self.columns_map[cast(columns)]
        else:
            columns = [self.columns_map[cast(i)] for i in columns]

        return columns

    def inverse_map(self, ind):

        ind = slice_to_index(ind, l=len(self), sliced=self.index)

        index_type = check_type(ind)

        if self.mapping_method == 'simple':
            pass
        elif self.mapping_method == 'series':
            if index_type.minor == Types.tensor:
                ind = as_numpy(ind)
            ind = as_tensor(self.index_map[ind].values, return_vector=True)
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

    def save(self, path):

        index = None if self.mapping_method == 'simple' else self.index

        state = {'data': self.data, 'index': index, 'columns': self.columns,
                 'requires_grad': self.data.requires_grad, 'device': self.data.device, 'series': self.series}
        torch.save(state, path)

    @staticmethod
    def load(path, map_location=None):

        state = torch.load(path)
        device = map_location if map_location is not None else state['device']
        return DataTensor(state['data'], index=state['index'], device=device, columns=state['columns'],
                          series=state['series'], requires_grad=state['requires_grad'])

    @property
    def values(self):

        data = self.data
        if self.series:
            data = data.squeeze(1)

        return data

    def _iloc(self, ind):

        ind = slice_to_index(ind, l=self.data.shape[0])
        index_type = check_type(ind)

        if index_type.major == Types.scalar:
            ind = [ind]

        index = self.inverse_map(ind)
        data = self.data[ind]

        return DataTensor(data, columns=self.columns, index=index)

    def to(self, device):
        self.data = self.data.to(device)

        return self

    def __repr__(self):

        if isinstance(self.index, torch.Tensor):
            index = as_numpy(self.index.data)
        else:
            index = self.index

        repr_data = repr(pd.DataFrame(data=as_numpy(self.data.detach()),
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

                if check_type(columns).major == Types.scalar:

                    data = data.unsqueeze(1)
                    columns = [columns]

                data = torch.cat([self.data, data], dim=1)
                columns = self.columns + columns

                self.__init__(data, columns=columns, index=self.index)

        raise ValueError

    def sort_index(self, ascending=True):

        sorted_index = torch.sort(self.index, descending=not ascending).values
        return self.loc[sorted_index]

    def _loc(self, ind):

        series = False
        if type(ind) is tuple:

            index = ind[0]
            columns = ind[1]

            ind_columns = self.inverse_columns_map(columns)
            if check_type(ind_columns).major == Types.scalar:
                ind_columns = [int(ind_columns)]
                columns = [columns]
                series = True

        else:

            index = ind
            columns = self.columns
            ind_columns = slice(None)

        index = slice_to_index(index, l=len(self), sliced=self.index)
        if check_type(index).major == Types.scalar:
            index = [index]

        ind_index = self.inverse_map(index)
        data = self.data[ind_index][slice(None), ind_columns]

        return DataTensor(data, columns=columns, index=index, series=series)

    def __getitem__(self, ind):

        series = False

        if (len(ind) == self.data.shape[0]) and is_boolean(ind):

            if len(ind) == 1:
                ind = torch.where(ind)[0]

            data = self.data[ind]
            index = self.index[ind]

            return DataTensor(data, columns=self.columns, index=index)

        columns = ind
        ind_columns = self.inverse_columns_map(columns)
        if check_type(ind_columns).major == Types.scalar:
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


