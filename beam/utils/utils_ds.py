import copy
import os
import sys
import subprocess
import numpy as np

import random
import pandas as pd
import pickle
import hashlib
from functools import partial
import itertools

import re
from .utils_all import (check_type, check_minor_type, slice_array, is_arange, DataObject, is_container,
                        jupyter_like_traceback)
from ..type import BeamType, Types, is_scalar

from ..importer import torch
from ..importer import scipy


def slice_to_index(s, l=None, arr_type=Types.tensor, sliced=None):

    if arr_type == Types.tensor:
        f = torch.arange
    elif arr_type == Types.numpy:
        f = np.arange
    elif arr_type == Types.pandas:
        f = pd.RangeIndex
    elif arr_type == Types.list:
        f = lambda start, stop, step: list(range(start, stop, step))
    else:
        raise ValueError(f"Unsupported array type: {arr_type}")

    if isinstance(s, slice):

        if s == slice(None):
            if sliced is not None:
                return sliced
            elif l is not None:
                return f(l)
            else:
                return ValueError(f"Cannot slice: {s} without length info")

        if l is None:
            l = len(sliced) if sliced is not None else 0

        step = s.step
        if step is None:
            step = 1

        start = s.start
        if start is None:
            start = 0 if step > 0 else l - 1
        elif start < 0:
            start = l + start

        stop = s.stop
        if stop is None:
            stop = l if step > 0 else -1
        elif stop < 0:
            stop = l + stop

        if sliced is not None:
            return slice_array(sliced, slice(start, stop, step))

        return f(start, stop, step)

    if sliced is not None:
        return slice_array(sliced, s)
    return s


def beam_device(device):
    if isinstance(device, torch.device) or device is None:
        return device
    device = str(device)
    if device == 'cuda':
        device = '0'
    return torch.device(int(device) if device.isnumeric() else device)


def as_something_recursively(as_something_func):
    def as_func_recursively(x, **kwargs):
        x_type = check_type(x)
        if x_type.major == Types.container and x_type.minor == Types.dict:
            return {k: as_func_recursively(v, **kwargs) for k, v in x.items()}
        elif x_type.major == Types.other:
            for k, v in x.__dict__.items():
                setattr(x, k, as_func_recursively(v, **kwargs))
            return x
        elif x_type.major == Types.container and x_type.minor in [Types.list, Types.tuple]:
            if x_type.element not in [Types.object, Types.unknown]:
                try:
                    return as_something_func(x, x_type=x_type, **kwargs)
                except:
                    pass
            if x_type.minor == Types.tuple:
                return tuple(as_func_recursively(xi, **kwargs) for xi in x)
            return [as_func_recursively(xi, **kwargs) for xi in x]
        elif x is None:
            return None

        return as_something_func(x, x_type=x_type, **kwargs)

    return as_func_recursively


@as_something_recursively
def as_tensor(x, x_type=None, device=None, dtype=None, brain=False,
              half=False, return_vector=False, convert_to_tensor=True, copy=False,
              convert_scalar=False, **kwargs) -> torch.Tensor:

    if x_type is None:
        x_type = check_type(x, element=False)

    if not convert_to_tensor and not x_type.is_torch:
        return x
    if not convert_scalar and x_type.is_scalar:
        return x

    device = beam_device(device)

    if dtype is None and hasattr(x, 'dtype'):
        dtype = str(x.dtype)
        if 'int' in dtype:
            dtype = torch.int64
        elif 'float' in dtype or 'object' in dtype:
            dtype = (torch.bfloat16 if brain else torch.float16) if half else torch.float32
        elif 'complex' in dtype:
            dtype = torch.complex32 if half else torch.complex64
        elif 'double' in dtype:
            dtype = torch.float64
        else:
            dtype = torch.float32

    if x_type.minor in [Types.pandas, Types.cudf]:
        x = x.values

    elif x_type.minor == Types.polars:
        x = x.to_numpy()

    elif x_type.minor in [Types.PackedTensor, Types.PackedArray]:
        data = torch.as_tensor(x.data, dtype=dtype, device=device)
        length = torch.as_tensor(x.length, dtype=torch.int64, device=device)
        return PackedTensor(data, length)

    if copy:
        x = torch.tensor(x, device=device, dtype=dtype)
    else:
        x = torch.as_tensor(x, device=device, dtype=dtype)
    if return_vector:
        if not len(x.shape):
            x = x.unsqueeze(0)

    return x


@as_something_recursively
def as_numpy(x, dtype=None, squeeze=False, **kwargs):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.array(x, dtype=dtype)

    if squeeze and x.size == 1:
        str_type = str(x.dtype)
        if 'float' in str_type:
            x = float(x)
        elif 'int' in str_type:
            x = int(x)
        elif 'complex' in str_type:
            x = complex(x)

    return x


def as_dataframe(x, target='pandas', **kwargs):
    x_type = check_type(x)
    if not x_type.is_dataframe:
        if x_type.is_torch:
            x = as_numpy(x)

    if target == 'pandas':
        return pd.DataFrame(x, **kwargs)

    if target == 'polars':
        import polars as pl
        return pl.DataFrame(x, **kwargs)

    if target == 'cudf':
        import cudf
        return cudf.DataFrame(x, **kwargs)

    raise ValueError(f"Unsupported target type: {target}")


def as_list(x, length=None, **kwargs):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.cpu().tolist()
    if isinstance(x, pd.Series):
        return x.values.tolist()
    if isinstance(x, pd.DataFrame):
        return [x[col].values.tolist() for col in x.columns]
    if isinstance(x, pd.Index):
        return x.values.tolist()
    if isinstance(x, scipy.sparse.csr_matrix):
        return x.toarray().tolist()
    if isinstance(x, scipy.sparse.coo_matrix):
        return x.toarray().tolist()
    if isinstance(x, dict):
        return list(x.values())
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, slice):
        return slice_to_index(x, length, arr_type=Types.list)
    return list(x)


def as_scipy_csr(x):
    # Handle PyTorch Tensors
    if isinstance(x, torch.Tensor):
        x = x.cpu()  # Ensure the tensor is on CPU
        x = x.coalesce()
        if x.layout == torch.sparse_coo:
            # Convert sparse COO tensor to CSR
            ind = x.indices().numpy()
            val = x.values().numpy()
            coo = scipy.sparse.coo_matrix((val, ind), shape=x.shape)
            return coo.tocsr()
        elif x.layout == torch.sparse_csr:
            # Directly create sparse CSR matrix from CSR components
            crow_indices = x.crow_indices().numpy()
            col_indices = x.col_indices().numpy()
            values = x.values().numpy()
            return scipy.sparse.csr_matrix((values, col_indices, crow_indices), shape=x.shape)
        else:
            # Convert dense tensor to CSR matrix
            return scipy.sparse.csr_matrix(x.numpy())

    # Handle NumPy arrays directly
    elif isinstance(x, np.ndarray):
        return scipy.sparse.csr_matrix(x)

    elif isinstance(x, scipy.sparse.coo_matrix):
        return x.tocsr()

    elif isinstance(x, scipy.sparse.csr_matrix):
        return x

    # Handle tuple input as (rows, cols, data) assuming it's in COO format
    elif isinstance(x, tuple) and len(x) == 3:
        coo = scipy.sparse.coo_matrix((x[2], (x[0], x[1])))
        return coo.tocsr()

    # Handle dictionary input with keys 'row', 'col', and 'val' assuming it's in COO format
    elif isinstance(x, dict) and {'row', 'col', 'val'}.issubset(x.keys()):
        coo = scipy.sparse.coo_matrix((x['val'], (x['row'], x['col'])))
        return coo.tocsr()

    else:
        raise ValueError("Unsupported input type for conversion to scipy.sparse.csr_matrix")


def as_scipy_coo(x):
    # Handle PyTorch Tensors
    if isinstance(x, torch.Tensor):
        x = x.cpu()  # Ensure the tensor is on CPU
        x = x.coalesce()
        if x.layout == torch.sparse_coo:
            # Extract indices and values for sparse COO tensor
            ind = x.indices().numpy()
            val = x.values().numpy()
            return scipy.sparse.coo_matrix((val, ind), shape=x.shape)
        elif x.layout == torch.sparse_csr:
            # Convert sparse CSR tensor to COO
            crow_indices = x.crow_indices().numpy()
            col_indices = x.col_indices().numpy()
            values = x.values().numpy()
            # Convert CSR components to COO format
            row_indices = np.repeat(np.arange(len(crow_indices) - 1), np.diff(crow_indices))
            return scipy.sparse.coo_matrix((values, (row_indices, col_indices)), shape=x.shape)
        else:
            # Convert dense tensor to COO matrix
            return scipy.sparse.coo_matrix(x.numpy())

    # Handle NumPy arrays directly
    elif isinstance(x, np.ndarray):
        return scipy.sparse.coo_matrix(x)

    elif isinstance(x, scipy.sparse.coo_matrix):
        return x

    elif isinstance(x, scipy.sparse.csr_matrix):
        return x.tocoo()

    # Handle tuple input as (rows, cols, data)
    elif isinstance(x, tuple) and len(x) == 3:
        return scipy.sparse.coo_matrix((x[2], (x[0], x[1])))

    # Handle dictionary input with keys 'row', 'col', and 'val'
    elif isinstance(x, dict) and {'row', 'col', 'val'}.issubset(x.keys()):
        return scipy.sparse.coo_matrix((x['val'], (x['row'], x['col'])))

    else:
        raise ValueError("Unsupported input type for conversion to scipy.sparse.coo_matrix")


def to_device(data, device='cuda', half=False, dtype=None, brain=False):
    return as_tensor(data, device=device, half=half, convert_to_tensor=False, dtype=dtype, brain=brain)


def concat_polars_horizontally(data, **kwargs):
    import polars as pl
    data = [v.with_column(pl.arange(0, v.height).alias("key")) for v in data]
    d = data[0]
    for v in data[1:]:
        d = d.join(v, on="key", how="inner")
    return d.drop_in_place("key")


def recursive_concatenate(data, dim=0, check_equal_batch_length=False):
    d0 = data[0]
    if isinstance(d0, dict):
        return {k: recursive_concatenate([di[k] for di in data], dim=dim,
                                         check_equal_batch_length=check_equal_batch_length) for k in d0.keys()}
    elif isinstance(d0, list) or isinstance(d0, tuple):
        return [recursive_concatenate([di[n] for di in data], dim=dim,
                                      check_equal_batch_length=check_equal_batch_length) for n in range(len(d0))]
    else:
        minor_type = check_minor_type(d0)

        if minor_type == Types.tensor:
            func = torch.cat
            kwargs = {'dim': dim}

            if check_equal_batch_length and dim == 0:
                if len(set([len(d) for d in data])) != 1:
                    func = PackedTensor
                    kwargs = {}

        elif minor_type == Types.pandas:
            func = pd.concat
            data = [pd.Series(v.values) if isinstance(v, pd.Index) else v for v in data]
            kwargs = {'axis': dim}
        elif minor_type == Types.polars:
            if dim == 0:
                import polars as pl
                func = pl.concat
                kwargs = {'axis': dim}
            else:
                func = concat_polars_horizontally
        elif minor_type == Types.PackedSet:
            func = d0.concat
            kwargs = {}

        elif minor_type == Types.cudf:
            import cudf
            func = cudf.concat
            data = [cudf.Series(v.values) if isinstance(v, cudf.Index) else v for v in data]
            kwargs = {'axis': dim}
        elif minor_type == Types.numpy:
            func = np.concatenate
            kwargs = {'axis': dim}

            if check_equal_batch_length and dim == 0:
                if len(set([len(d) for d in data])) != 1:
                    func = PackedArray
                    kwargs = {}

        else:
            raise ValueError(f"Concatenation not implemented for {minor_type}, returning the original data")

        return func(data, **kwargs)


def batch_augmentation_(x, augmentations):
    return torch.stack([augmentations(xi) for xi in x])


def batch_augmentation(augmentations):
    ba = partial(batch_augmentation_, augmentations=augmentations)
    from torchvision import transforms
    return transforms.Lambda(ba)


def hash_tensor(x, fast=False, coarse=False):
    """
    This  function returns a deterministic hash of the tensor content
    @param x: the tensor to hash
    @param fast: whether to consider only the first and last elements of the tensor for hashing
    @param coarse: whether to apply coarse hashing where the tensor is quantized into low resolution (16bit) tensor
    @return: an integer representing the hash value
    """
    if torch.numel(x) < 10000:
        fast = False

    if coarse and 'float' in str(x.dtype):
        x = (x / x.max() * (2 ** 15)).half()

    x = as_numpy(x)

    if fast:
        x = str(x).encode('utf-8')
    else:
        x.flags.writeable = False
        x = x.data

    return int(hashlib.sha1(x).hexdigest(), 16)


def set_seed(seed=-1, constant=0, increment=False, deterministic=False):
    '''
    :param seed: set -1 to avoid change, set 0 to randomly select seed, set [1, 2**31) to get new seed
    :param constant: a constant to be added to the seed
    :param increment: whether to generate incremental seeds
    :param deterministic: whether to set torch to be deterministic
    :return: None
    '''

    if 'cnt' not in set_seed.__dict__:
        set_seed.cnt = 0
    set_seed.cnt += 1

    if increment:
        constant += set_seed.cnt

    if seed == 0:
        seed = np.random.randint(1, 2 ** 31 - constant) + constant
    else:
        seed += constant

    if seed > 0:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True


def divide_chunks(x, chunksize=None, n_chunks=None, partition=None, squeeze=False, dim=0, x_type=None,
                  chunksize_policy='round'):

    if chunksize_policy == 'round':
        round_func = np.round
    elif chunksize_policy == 'ceil':
        round_func = np.ceil
    elif chunksize_policy == 'floor':
        round_func = np.floor
    else:
        raise ValueError(f"Unsupported chunksize_policy: {chunksize_policy}")

    assert ((chunksize is None) != (n_chunks is None)), "divide_chunks requires only one of chunksize|n_chunks"
    x_type = x_type or check_type(x, element=False)

    # assert x_type.major in [Types.array, Types.other], "divide_chunks supports only array types"

    if n_chunks is not None and hasattr(x, '__len__'):
        n_chunks = min(len(x), n_chunks)
        chunksize = len(x) // n_chunks

    if x_type.major == Types.array:

        l = len(x)

        if chunksize is None:
            chunksize = l // n_chunks

        if n_chunks is None:
            n_chunks = max(int(round_func(l / chunksize)), 1)

        if x_type.minor == Types.tensor:
            for i, c in enumerate(torch.tensor_split(x, n_chunks, dim=dim)):
                if squeeze and len(c) == 1:
                    c = c.squeeze()

                yield i, c

        elif x_type.minor in [Types.pandas, Types.cudf] and partition != None:

            grouped = x.groupby(partition, sort=True)
            for k, g in grouped:
                yield k, g

        elif x_type.minor == Types.numpy:

            for i, c in enumerate(np.array_split(x, n_chunks, axis=dim)):
                if squeeze and len(c) == 1:
                    c = c.squeeze()
                yield i, c

        elif x_type.minor in [Types.pandas, Types.cudf]:

            if x_type.minor == Types.cudf:
                import cudf as upd
            else:
                upd = pd

            if chunksize == 1 and dim == 0:
                for i, c in x.iterrows():
                    if squeeze:
                        yield i, c
                    yield i, c.to_frame().T
            else:

                index_name = x.index.name or 'index'
                is_series = x.ndim == 1
                x = x.reset_index()
                columns = x.columns

                for i, c in enumerate(np.array_split(x, n_chunks, axis=dim)):

                    if squeeze and len(c) == 1:
                        c = c.squeeze()
                        c = upd.Series(c, index=columns)
                        c.name = c[index_name]
                        c = c.drop(index_name)

                    else:
                        c = upd.DataFrame(data=c, columns=columns)
                        c = c.set_index(index_name)

                        if is_series:
                            c = c.squeeze()

                    yield i, c

        elif x_type.minor == Types.polars:

            for i in range(n_chunks):
                c = x.slice(i * chunksize, (i + 1) * chunksize)
                if squeeze and len(c) == 1:
                    c = c[0]
                yield i, c

        else:
            for j, i in enumerate(np.array_split(np.arange(l), n_chunks)):

                v = x[i[0]:i[-1] + 1]
                if squeeze and len(v) == 1:
                    v = v[0]
                yield j, v

    elif x_type.major == Types.container and x_type.minor == Types.dict:

        if chunksize == 1:
            for k, v in x.items():
                yield k, v
        else:
            items = list(x.items())
            chunks = [items[i:i + chunksize] for i in range(0, len(items), n_chunks)]
            for i, c in enumerate(chunks):
                yield i, dict(c)

    else:

        if hasattr(x, '__len__') and chunksize_policy != 'tail':
            l = len(x)

            if chunksize is None:
                chunksize = l // n_chunks

            if n_chunks is None:
                n_chunks = max(int(round_func(l / chunksize)), 1)

            effective_chunksize = l // n_chunks

        else:
            effective_chunksize = chunksize

        c = []
        i = 0
        for xi in iter(x):

            c.append(xi)
            if len(c) == effective_chunksize:

                if squeeze and len(c) == 1:
                    c = c[0]
                yield i, c

                c = []
                i += 1

        if len(c):
            yield i, c


def recursive_chunks(x, chunksize=None, n_chunks=None, partition=None, squeeze=False, dim=0,
                     x_type=None, chunksize_policy='round'):
    x_type = x_type or check_type(x)

    try:

        if dim is None:
            for k, c in divide_chunks(x, chunksize=chunksize, n_chunks=n_chunks, partition=partition,
                                      squeeze=squeeze, dim=0, x_type=x_type, chunksize_policy=chunksize_policy):
                yield k, c

        elif (x_type.major == Types.container) and (x_type.minor == Types.dict):
            gen = {k: recursive_chunks(v, chunksize=chunksize, n_chunks=n_chunks, chunksize_policy=chunksize_policy,
                                       partition=partition, squeeze=squeeze, dim=dim) for k, v in x.items()}

            for i in itertools.count():
                d = {}
                for k, v in gen.items():
                    i, v = next(v)
                    d[k] = v

                yield i, d

        elif x_type.major == Types.container:

            gen = [recursive_chunks(s, chunksize=chunksize, n_chunks=n_chunks, partition=partition,
                                    squeeze=squeeze, dim=dim, chunksize_policy=chunksize_policy) for s in x]
            for i in itertools.count():
                # yield [next(s) for s in gen]
                l = []
                for k, s in enumerate(gen):
                    i, s = next(s)
                    l.append(s)

                yield i, l

        elif x is None:
            for i in itertools.count():
                yield i, None
        else:
            for k, c in divide_chunks(x, chunksize=chunksize, n_chunks=n_chunks, partition=partition,
                                      squeeze=squeeze, dim=dim, x_type=x_type, chunksize_policy=chunksize_policy):
                yield k, c

    except StopIteration:
        return


def recursive_collate_chunks(*xs, dim=0, on='index', how='outer', method='tree'):
    x_type = check_type(xs[0])
    if x_type.major == Types.container:

        values = []
        keys = []

        for k, _ in iter_container(xs[0]):
            values.append(recursive_collate_chunks(*[xi[k] for xi in xs], dim=dim, on=on, how=how, method=method))
            keys.append(k)

        if x_type.minor == Types.dict:
            values = dict(zip(keys, values))

        return values

    else:
        return collate_chunks(*xs, dim=dim, on=on, how=how, method=method)


def collate_chunks(*xs, keys=None, dim=0, on='index', how='outer', method='tree', squeeze=True, logger=None):
    if len(xs) == 0:
        return []

    if len(xs) == 1:
        if squeeze:
            return xs[0]
        return [xs[0]]

    x = list(xs)

    x_type = check_type(x[0], element=False)

    if keys is not None:
        if x_type.minor not in [Types.pandas, Types.cudf] or dim != 0:
            msg = f"Cannot collate with keys for {x_type.minor}, or dim={dim}"
            if logger is not None:
                logger.warning(f"{msg}, returning the data as a dictionary (keys, values)")
                return {k: v for k, v in zip(keys, x)}
            else:
                raise ValueError(msg)

    if x_type.major == Types.container and x_type.minor == Types.dict:
        dictionary = {}
        for xi in x:
            dictionary.update(xi)
        return dictionary

    if x_type.major == Types.container and x_type.minor == Types.list and dim == 0:
        l = []
        for xi in x:
            l.extend(xi)
        return l

    if (x_type.major not in [Types.array, Types.other]) or (dim == 1 and not x_type.is_data_array):
        return x

    if x_type.minor == Types.tensor:
        return torch.cat(x, dim=dim)

    elif x_type.minor == Types.numpy:
        return np.concatenate(x, axis=dim)

    elif x_type.minor == Types.scipy_sparse:

        if dim == 0:
            return scipy.sparse.vstack(x)
        return scipy.sparse.hstack(x)

    elif x_type.minor in [Types.pandas, Types.cudf]:

        if x_type.minor == Types.cudf:
            import cudf as upd
        else:
            upd = pd

        if dim == 0:
            if len(x[0].shape) == 1:
                x = [upd.Series(xi) for xi in x]
            if keys is not None:
                df = upd.concat(x, axis=dim, ignore_index=True)
                df.index = keys
            else:
                df = upd.concat(x, axis=dim)
            return df

        elif on == 'index':
            return recursive_merge(x, method=method, how=how, left_index=True, right_index=True)
        else:
            return recursive_merge(x, method=method, how=how, on=on)

    elif x_type.minor == Types.polars:
        if dim == 1:
            return concat_polars_horizontally(x)
        import polars as pl
        df = pl.concat(x)
        return df
    else:

        xc = []
        for xi in iter(x):
            xc.extend(xi)
        return xc


def recursive_merge(dfs, method='tree', **kwargs):
    if len(dfs) == 1:
        return dfs[0]
    if len(dfs) == 2:
        return pd.merge(dfs[0], dfs[1], **kwargs)
    if method == 'series':
        return recursive_merge([dfs[0], recursive_merge(dfs[1:], method='series', **kwargs)], method='series', **kwargs)
    if method == 'tree':
        return recursive_merge([recursive_merge(dfs[:len(dfs) // 2], method='tree', **kwargs),
                                recursive_merge(dfs[len(dfs) // 2:], method='tree', **kwargs)], method='tree', **kwargs)
    raise ValueError('Unknown method type')


def iter_container(x):
    if hasattr(x, 'items'):
        return iter(x.items())
    return enumerate(x)


def get_chunks(x, chunksize=None, n_chunks=None, partition=None, dim=0):
    keys = []
    values = []
    for k, v in recursive_chunks(x, chunksize=chunksize, n_chunks=n_chunks, partition=partition, dim=dim):
        keys.append(k)
        values.append(v)

    argsort, isarange = is_arange(keys)
    if not isarange:
        values = dict(zip(keys, values))
    else:
        values = [values[i] for i in argsort]

    return values


def recursive_size(x):
    x_type = check_type(x)
    if x_type.major == Types.container:

        keys = []
        values = []

        for k, v in iter_container(x):
            keys.append(k)
            values.append(recursive_size(v))

        if x_type.minor == Types.dict:
            values = dict(zip(keys, values))

        return values

    else:

        return object_size(x, x_type=x_type)


def object_size(x, x_type=None):
    if x_type is None:
        x_type = check_type(x)
    if x_type.minor == Types.tensor:
        return x.element_size() * x.nelement()
    elif x_type.minor in [Types.numpy, Types.scipy_sparse]:
        return x.size * x.dtype.itemsize
    elif x_type.minor in [Types.pandas, Types.cudf]:
        try:
            return np.sum(x.memory_usage(index=True, deep=True))
        except:
            return x.size * x.dtype.itemsize
    elif x_type.minor == Types.polars:
        return x.estimated_size()
    elif x_type.minor == Types.list:
        if len(x) <= 1000:
            return np.sum([sys.getsizeof(i) for i in x])
        ind = np.random.randint(len(x), size=(1000,))
        return len(x) * np.mean([sys.getsizeof(x[i]) for i in ind])
    else:
        return sys.getsizeof(x)


def recursive_elementwise(func):
    def apply_recursively(x, *args, **kwargs):

        if isinstance(x, dict):
            return {k: apply_recursively(v, *args, **kwargs) for k, v in x.items()}
        if isinstance(x, list):
            return [apply_recursively(v, *args, **kwargs) for v in x]
        if isinstance(x, tuple):
            return tuple(apply_recursively(v, *args, **kwargs) for v in x)
        return func(x, *args, **kwargs)

    return apply_recursively


def recursive(func):
    def apply_recursively(x, *args, in_place=False, **kwargs):

        if is_container(x):

            keys = []
            values = []

            for k, v in iter_container(x):
                keys.append(k)
                values.append(apply_recursively(v, *args, **kwargs))

            if isinstance(x, dict):
                values = dict(zip(keys, values))

            if isinstance(x, tuple):
                values = tuple(values)

            return values

        elif in_place and check_minor_type(x) == Types.other:
            if hasattr(x, '__dict__'):
                for k, v in x.__dict__.items():
                    setattr(x, k, apply_recursively(v, *args, **kwargs))
                return x
            else:
                return func(x, *args, **kwargs)
        else:
            return func(x, *args, **kwargs)

    return apply_recursively


def recursive_yield(func, keys=True, values=True, level=-1):
    def apply_recursively(x, *args, _keys=True, _values=True, _level=-1, **kwargs):

        def _apply(_k, _v):
            for item in apply_recursively(_v, *args, _keys=_keys, _values=_values, _level=_level-1, **kwargs):

                if _keys and _values:
                    kk, vv = item
                    kk = (_k,) + kk
                    yield kk, vv
                elif _keys:
                    item = (_k,) + item
                    yield item
                else:
                    yield item

        if _level and is_container(x):
            for k, v in iter_container(x):
                for vv in _apply(k, v):
                    yield vv
            return

        elif _level and check_minor_type(x) == Types.other:
            if hasattr(x, '__dict__'):
                for k, v in x.__dict__.items():
                    for vv in _apply(k, v):
                        yield vv
                return

        if _keys and _values:
            yield tuple(), func(x, *args, **kwargs)
        elif _keys:
            yield tuple()
        else:
            yield func(x, *args, **kwargs)

    return partial(apply_recursively, _keys=keys, _values=values, _level=level)


def recursive_values(x, level=1):
    return recursive_yield(lambda y: y, keys=False, values=True, level=level)(x)


def recursive_items(x, level=1):
    for k, v in recursive_yield(lambda y: y, keys=True, values=True, level=level)(x):
        if level == 1:
            yield k[0], v
        else:
            yield k, v


def recursive_keys(x, level=1):
    for k in recursive_yield(lambda y: y, keys=True, values=False, level=level)(x):
        if level == 1:
            yield k[0]
        else:
            yield k


@recursive
def recursive_clone(x):
    x_minor = check_minor_type(x)
    if x_minor in [Types.tensor, Types.polars]:
        return x.clone()
    elif x_minor == Types.numpy:
        return x.copy()
    elif x_minor in [Types.pandas, Types.cudf]:
        return x.copy(deep=True)
    elif x_minor == Types.scipy_sparse:
        return x.copy()
    else:
        return copy.deepcopy(x)


@recursive
def recursive_devices(x):
    x_minor = check_minor_type(x)
    if x_minor == Types.tensor:
        return str(x.device)
    else:
        return 'none'


def recursive_same_device(x):
    devices = recursive_devices(x)
    devices = recursive_flatten(devices)
    devices = set(devices)
    return len(devices) == 1 and 'none' not in devices


@recursive
def recursive_detach(x):
    x_minor = check_minor_type(x)
    if x_minor == Types.tensor:
        return x.detach()
    return x


@recursive
def recursive_to_cpu(x):
    x_minor = check_minor_type(x)
    if x_minor == Types.tensor:
        return x.cpu()
    return x


def beam_hash(x, bytes_threshold=int(1e6), fast=True):
    h = hashlib.sha1()
    _beam_hash(x, h, bytes_threshold=bytes_threshold, fast=fast)
    return h.hexdigest()


@recursive
def _beam_hash(x, h, bytes_threshold=int(1e6), fast=True):
    if object_size(x) > bytes_threshold and fast:
        h.update(big_array_representation(x))
    else:
        h.update(pickle.dumps(x))


@recursive
def recursive_batch(x, index):
    return slice_array(x, index, x_type=None, indices_type=None)


@recursive
def recursive_len(x, data_array_only=False):
    x_type = BeamType.check(x)

    if x is None:
        return 0

    if x_type.minor == Types.scipy_sparse:
        return x.shape[0]

    if data_array_only:
        if x_type.is_data_array:
            return len(x)
        if empty_element(x, x_type=x_type):
            return 0
        return 1
    else:
        if hasattr(x, '__len__'):
            try:
                return len(x)
            except TypeError:
                print(jupyter_like_traceback())
                return 1

    if x_type.element == Types.none:
        return 1

    return 1


@recursive
def recursive_types(x):
    x_type = check_type(x)
    return f'{x_type.major}.{x_type.minor}.{x_type.element}'


@recursive
def recursive_shape(x):
    if hasattr(x, 'shape'):
        return x.shape
    if hasattr(x, '__len__'):
        return len(x)
    return None


@recursive
def recursive_slice(x, s):
    if x is None:
        return None
    return x.__getitem__(s)


@recursive
def recursive_slice_columns(x, columns, columns_index):
    x_type = check_type(x)

    if x is None:
        return None
    elif x_type.minor in [Types.pandas, Types.cudf]:
        return x[columns]
    elif x_type.minor == Types.polars:
        return x.select(columns)
    else:
        return x[:, columns_index]


def recursive_device(x):
    if isinstance(x, dict):
        for xi in x.values():
            try:
                return recursive_device(xi)
            except AttributeError:
                # case of None
                pass
    elif isinstance(x, list) or isinstance(x, tuple):
        for xi in x:
            try:
                return recursive_device(xi)
            except AttributeError:
                # case of None
                pass

    elif check_minor_type(x) == Types.other:
        for k, v in x.__dict__.items():
            try:
                return recursive_device(v)
            except AttributeError:
                # case of None
                pass

    if hasattr(x, 'device'):
        return x.device
    return None


def container_len(x):
    if isinstance(x, dict):
        for xi in x.values():
            try:
                return container_len(xi)
            except TypeError:
                # case of None
                pass

    elif isinstance(x, list):
        for xi in x:
            try:
                return container_len(xi)
            except TypeError:
                # case of None
                pass

    return len(x)


def big_array_representation(x):
    n = 100
    nl = 1000
    seed = 42

    metadata = None
    minor_type = check_minor_type(x)
    if minor_type in [Types.pandas, Types.cudf]:
        metadata = x.columns
        x = x.values
    elif minor_type == Types.scipy_sparse:
        metadata = x.shape
        x = x.data
    if minor_type in [Types.numpy, Types.tensor, Types.pandas, Types.scipy_sparse, Types.cudf]:
        ind = tuple(slice(0, i, i // n) if i > n else slice(None) for i in x.shape)
        x = x.__getitem__(ind)

    if minor_type == Types.polars:
        metadata = x.columns
        import polars as pl
        x = pl.concat([x.head(n // 2), x.sample(n=min(n, len(x) // nl), seed=seed), x.tail(n // 2)])

    if minor_type in [Types.list, Types.tuple, Types.set]:
        x = list(x)[::len(x) // nl]

    return str((minor_type, metadata, x)).encode('utf-8')


def recursive_hierarchical_keys(x):
    x_type = check_type(x)
    if x_type.major == Types.container:

        keys = []
        values = []

        for k, v in iter_container(x):
            keys.append(k)
            values.append(recursive_hierarchical_keys(v))

        if all([v is None for v in values]):
            return keys

        if x_type.minor == Types.dict:

            argsort, isarange = is_arange(keys)
            if not isarange:
                values = dict(zip(keys, values))
            else:
                values = [values[i] for i in argsort]

        return values

    return None


def recursive_size_summary(x, mode='sum'):
    x_type = check_type(x)

    if x_type.minor == Types.dict:

        if mode == 'sum':
            return sum([recursive_size_summary(v, mode=mode) for v in x.values()])
        elif mode == 'max':
            return max([recursive_size_summary(v, mode=mode) for v in x.values()])
        else:
            raise NotImplementedError

    elif (x_type.minor in [Types.list, Types.tuple]) and x_type.element in [Types.object, Types.unknown, Types.other]:

        if mode == 'sum':
            return sum([recursive_size_summary(s, mode=mode) for s in x])
        elif mode == 'max':
            return max([recursive_size_summary(s, mode=mode) for s in x])
        else:
            raise NotImplementedError

    elif x is None:
        return 0
    else:
        if x_type.minor == Types.tensor:
            return x.element_size() * x.nelement()
        elif x_type.minor in [Types.numpy, Types.scipy_sparse]:
            return x.size * x.dtype.itemsize
        elif x_type.minor == Types.pandas:
            return np.sum(x.memory_usage(index=True, deep=True))
        elif x_type.minor == Types.polars:
            return x.esitmate_size()
        else:
            return sys.getsizeof(x)


@recursive
def recursive_squeeze(x):
    x_type = check_type(x, element=False)
    if x_type.minor == Types.tensor:
        return x.squeeze(0)
    elif x_type.minor == Types.numpy:
        return np.squeeze(x, axis=0)
    elif x_type.minor == Types.pandas:
        if isinstance(x, pd.Index) and len(x) == 1:
            return x[0]
        return x.squeeze('index')
    elif x_type.minor == Types.cudf:
        import cudf
        if isinstance(x, cudf.Index) and len(x) == 1:
            return x[0]
        return x.squeeze('index')
    elif x_type.minor in [Types.scipy_sparse, Types.polars]:
        ValueError("Cannot squeeze scipy sparse matrix")
    elif x_type.minor == Types.list and len(x) == 1:
        return x[0]
    else:
        return x


def empty_element(x, x_type=None):
    if x_type is None:
        x_type = check_type(x)
    if x_type.minor in [Types.numpy, Types.pandas, Types.tensor, Types.scipy_sparse, Types.cudf]:
        return x.size == 0
    if x_type.minor in [Types.list, Types.tuple, Types.set, Types.dict, Types.polars]:
        return len(x) == 0

    if x_type.minor == Types.native:
        return x is None

    if hasattr(x, '__len__'):
        return x.__len__() == 0

    return False


@recursive
def recursive_empty_elements(x):
    return empty_element(x)


def is_empty(x):
    for v in recursive_values(x):
        if not empty_element(v):
            return False
    return True


def is_chunk(path, chunk_pattern='_chunk'):
    return path.is_file() and bool(re.search(rf'\d{6}{chunk_pattern}\.', str(path.name)))


def recursive_flatten(x, flat_array=False, x_type=None, tolist=True, _root=True, depth=-1):

    if depth == 0:
        return [x]

    if x_type is None:
        x_type = check_type(x)

    if x_type.major == Types.container:
        l = []
        for i, xi in iter_container(x):
            l.extend(recursive_flatten(xi, flat_array=flat_array, _root=False, tolist=tolist, depth=depth-1))
        return l
    else:
        if _root:
            if flat_array and x_type.major == Types.array:
                return recursive_flat_array(x, x_type=x_type, tolist=tolist)
            else:
                return [x]
        if isinstance(x, DataObject):
            return [x.data]
        elif not flat_array or x_type.major == Types.scalar:
            return [x]
        else:
            return recursive_flat_array(x, x_type=x_type, tolist=tolist)


def recursive_flatten_with_keys(x):
    x_type = check_type(x)

    if x_type.major == Types.container:
        d = {}
        for i, xi in iter_container(x):
            di = recursive_flatten_with_keys(xi)
            di = {(i, *k): v for k, v in di.items()}
            d.update(di)
        return d
    else:
        return {tuple(): x}


def recursive_flat_array(x, x_type=None, tolist=True):
    if x_type is None:
        x_type = check_type(x)

    if x_type.minor in [Types.numpy, Types.tensor]:
        x = x.flatten()
        if tolist:
            x = x.tolist()
        return x
    elif x_type.minor == Types.pandas:
        x = x.values.flatten()
        if tolist:
            x = x.tolist()
        return x
    elif x_type.minor == Types.scipy_sparse:
        x = x.toarray().flatten()
        if tolist:
            x = x.tolist()
        return x
    elif x_type.minor in [Types.list, Types.tuple]:
        if x_type.element != Types.array:
            return list(x)

        l = []
        for xi in x:
            l.extend(recursive_flat_array(xi, tolist=tolist))
        return l

    elif x_type.minor == Types.native:
        return [x]

    else:
        return [x]


def check_nvlink():
    try:
        # Running the nvidia-smi command
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Parsing the output
        output = result.stdout
        # Check for NVLink in the output
        if "NVLink" in output:
            return True
        else:
            return False
    except FileNotFoundError:
        return False


class GPUManager:
    @staticmethod
    def physical_devices(logical_devices=None):
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            devices = list(range(torch.cuda.device_count()))
        else:
            devices = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        if logical_devices is None:
            return devices
        else:
            return [devices[i] for i in logical_devices]

    @staticmethod
    def logical_devices(physical_devices):
        local_physical_devices = GPUManager.physical_devices()
        return [local_physical_devices.index(d) for d in physical_devices]


class PackedSet:
    def __init__(self, data, length=None):
        raise NotImplementedError("PackedSet is a base class and cannot be instantiated directly.")

    def __len__(self):
        return len(self.offset)

    def clone(self):
        raise NotImplementedError

    def aggregate(self, func):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __repr__(self):
        return repr(self.data)

    @property
    def beam_class_name(self):
        return self.__class__.__name__


class PackedTensor(PackedSet):
    def __init__(self, data, length=None, device=None):

        if length is None:
            self.data = torch.cat(data, dim=0)
            self.length = torch.tensor([0] + [len(x) for x in data], device=self.data.device)
        else:
            self.data = data
            self.length = torch.tensor([0] + list(length), device=self.data.device)

        self._offset = self.length.cumsum(dim=0)
        self.length = self.length[1:]
        self.offset = self._offset[:-1]
        self.index = torch.arange(len(self.offset), device=self.data.device)

        if device is not None:
            self.to(device)

    def clone(self):
        return PackedTensor(self.data.clone(), self.length.clone())

    def to(self, device):
        self.data = self.data.to(device)
        self._offset = self._offset.to(device)
        self.length = self.length.to(device)
        self.offset = self.offset.to(device)
        self.index = self.index.to(device)
        return self

    def aggregate(self, func):
        return torch.stack([func(self.data[self._offset[i]:self._offset[i + 1]]) for i in range(len(self))])

    def __getitem__(self, index):

        if isinstance(index, int):
            return self.data[self._offset[index]:self._offset[index + 1]]

        if isinstance(index, slice):
            index = self.index[index]

        if isinstance(index, torch.Tensor):
            if index.dtype == torch.bool:
                index = self.index[index]
            shape = index.shape
            if len(shape) == 1:
                return PackedTensor([self.data[self._offset[i]:self._offset[i + 1]] for i in index])
            else:
                raise NotImplementedError

        elif isinstance(index, tuple):
            assert len(index) == 2
            a, b = index
            return self.data[b + self._offset[a]]

        else:
            raise NotImplementedError

    @classmethod
    def concat(cls, *args):
        return cls(torch.cat([a.data for a in args], dim=0),
                            torch.cat([a.length for a in args], dim=0))


class PackedArray(PackedSet):
    def __init__(self, data, length=None):

        if length is None:
            self.data = np.concatenate(data, axis=0)
            self.length = np.array([0] + [len(x) for x in data])
        else:
            self.data = data
            self.length = np.array([0] + list(length))

        self._offset = np.cumsum(self.length)
        self.length = self.length[1:]
        self.offset = self._offset[:-1]
        self.index = np.arange(len(self.offset))

    def clone(self):
        return PackedArray(self.data.copy(), self.length.copy())

    def aggregate(self, func):
        return np.stack([func(self.data[self._offset[i]:self._offset[i + 1]]) for i in range(len(self))])

    def __getitem__(self, index):

        if isinstance(index, slice):
            index = np.arange(len(self))[index]

        if isinstance(index, np.ndarray):
            if index.dtype == bool:
                index = self.index[index]
        elif isinstance(index, int):
            return self.data[self._offset[index]:self._offset[index + 1]]

        if isinstance(index, (np.ndarray, list)):
            if np.isscalar(index):
                return self.data[self._offset[index]:self._offset[index + 1]]
            elif len(np.shape(index)) == 1:
                return PackedArray([self.data[self._offset[i]:self._offset[i + 1]] for i in index])
            else:
                raise NotImplementedError

        elif isinstance(index, tuple):
            assert len(index) == 2
            a, b = index
            return self.data[b + self._offset[a]]

        else:
            raise NotImplementedError

    @classmethod
    def concat(cls, *args):
        return cls(np.concatenate([a.data for a in args], axis=0),
                           np.concatenate([a.length for a in args], axis=0))
