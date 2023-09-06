import os, sys
from collections import defaultdict
import numpy as np

import random
import torch
import pandas as pd

import pickle
from .utils_all import check_element_type
from torchvision import transforms
import hashlib
from functools import partial
import itertools
import scipy
import re
from .utils_all import check_type, check_minor_type


def slice_to_index(s, l=None, arr_type='tensor', sliced=None):
    if isinstance(s, slice):

        f = torch.arange if arr_type == 'tensor' else np.arange

        if s == slice(None):
            if sliced is not None:
                return sliced
            elif l is not None:
                return f(l)
            else:
                return ValueError(f"Cannot slice: {s} without length info")

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

        return f(start, stop, step)
    return s


def as_numpy(x):
    if isinstance(x, dict):
        return {k: as_numpy(v) for k, v in x.items()}
    elif isinstance(x, list):
        x_type = check_type(x, check_minor=False, check_element=False)
        if x_type.major != 'container':
            # take care of the case of a list of scalar tensors
            if isinstance(x[0], torch.Tensor):
                return torch.stack(x).detach().cpu().numpy()
            return np.array(x)
        return [as_numpy(s) for s in x]

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.array(x)

    if x.size == 1:
        str_type = str(x.dtype)
        if 'float' in str_type:
            x = float(x)
        elif 'int' in str_type:
            x = int(x)
        elif 'complex' in str_type:
            x = complex(x)

    return x


def as_tensor(x, device=None, dtype=None, return_vector=False):
    x_type = check_type(x)
    if x_type.major == 'container' and x_type.minor == 'dict':
        return {k: as_tensor(v, device=device, return_vector=return_vector) for k, v in x.items()}
    elif x_type.major == 'container' and x_type.minor in ['list', 'tuple']:
        return [as_tensor(s, device=device, return_vector=return_vector) for s in x]
    elif x is None:
        return None

    if dtype is None and hasattr(x, 'dtype'):
        if 'int' in str(x.dtype):
            dtype = torch.int64
        elif 'float' in str(x.dtype):
            dtype = torch.float32
        elif 'complex' in str(x.dtype):
            dtype = torch.complex64

    if check_type(x, check_element=False).minor == 'pandas':
        x = x.values

    x = torch.as_tensor(x, device=device, dtype=dtype)
    if return_vector:
        if not len(x.shape):
            x = x.unsqueeze(0)

    return x


def recursive_concatenate(data, dim=0):
    d0 = data[0]
    if isinstance(d0, dict):
        return {k: recursive_concatenate([di[k] for di in data], dim=dim) for k in d0.keys()}
    elif isinstance(d0, list) or isinstance(d0, tuple):
        return [recursive_concatenate([di[n] for di in data], dim=dim) for n in range(len(d0))]
    else:
        minor_type = check_minor_type(d0)

        if minor_type == 'tensor':
            func = torch.cat
            kwargs = {'dim': dim}
        elif minor_type == 'pandas':
            func = pd.concat
            data = [pd.Series(v.values) if isinstance(v, pd.Index) else v for v in data]
            kwargs = {'axis': dim}
        elif minor_type == 'numpy':
            func = np.concatenate
            kwargs = {'axis': dim}
        else:
            raise ValueError(f"Concatenation not implemented for {minor_type}, returning the original data")

        return func(data, **kwargs)


def batch_augmentation_(x, augmentations):
    return torch.stack([augmentations(xi) for xi in x])


def batch_augmentation(augmentations):
    ba = partial(batch_augmentation_, augmentations=augmentations)
    return transforms.Lambda(ba)


def finite_iterations(iterator, n):
    for i, out in enumerate(iterator):
        yield out
        if i + 1 == n:
            break


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




def setup_distributed(rank, world_size, port='7463', backend='gloo'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    import torch.distributed as dist
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup(rank, world_size):
    import torch.distributed as dist
    dist.destroy_process_group()


def set_seed(seed=-1, constant=0, increment=False, deterministic=False):
    '''
    :param seed: set -1 to avoid change, set 0 to randomly select seed, set [1, 2**32) to get new seed
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
        seed = np.random.randint(1, 2 ** 32 - constant) + constant

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


def to_device(data, device='cuda', half=False):
    if isinstance(data, dict):
        return {k: to_device(v, device=device, half=half) for k, v in data.items()}
    elif isinstance(data, list) or isinstance(data, tuple):
        return [to_device(s, device=device, half=half) for s in data]
    elif isinstance(data, torch.Tensor):
        if half and data.dtype in [torch.float32, torch.float64]:
            data = data.half()
        if half and data.dtype in [torch.complex64, torch.complex128]:
            data = data.astype(torch.complex32)
        return data.to(device)
    else:
        return data


def beam_device(device):
    if isinstance(device, torch.device) or device is None:
        return device
    device = str(device)
    if device == 'cuda':
        device = '0'
    return torch.device(int(device) if device.isnumeric() else device)



def divide_chunks(x, chunksize=None, n_chunks=None, partition=None, squeeze=False, dim=0):
    assert ((chunksize is None) != (n_chunks is None)), "divide_chunks requires only one of chunksize|n_chunks"
    x_type = check_type(x, check_element=False)

    assert x_type.major in ['array', 'other'], "divide_chunks supports only array types"

    if n_chunks is not None and hasattr(x, '__len__'):
        n_chunks = min(len(x), n_chunks)

    if x_type.major == 'array':

        l = len(x)

        if chunksize is None:
            chunksize = l // n_chunks

        if n_chunks is None:
            n_chunks = int(np.round(l / chunksize))

        if x_type.minor == 'tensor':
            for i, c in enumerate(torch.tensor_split(x, n_chunks, dim=dim)):
                if squeeze and len(c) == 1:
                    c = c.squeeze()

                yield i, c

        elif x_type.minor == 'pandas' and partition != None:

            grouped = x.groupby(partition, sort=True)
            for k, g in grouped:
                yield k, g

        elif x_type.minor in ['numpy', 'pandas']:

            for i, c in enumerate(np.array_split(x, n_chunks, axis=dim)):
                if squeeze and len(c) == 1:
                    c = c.squeeze()
                yield i, c

        else:
            for j, i in enumerate(np.array_split(np.arange(l), n_chunks)):

                v = x[i[0]:i[-1] + 1]
                if squeeze and len(v) == 1:
                    v = v[0]
                yield j, v

    else:

        c = []
        i = 0
        for i, xi in enumerate(iter(x)):

            c.append(xi)
            if len(c) == chunksize:

                if squeeze and len(c) == 1:
                    c = c[0]
                yield i, c

                c = []

        i = i + 1 if i > 0 else i
        yield i, c


def stack_batched_results(results, batch_size=None):
    # this step is done in order to be able to pickle the results
    stacked_results = defaultdict(dict)
    for n, res in results.items():
        for k, v in res.items():
            if type(v) == list:
                v_minor = check_type(v[0]).minor
                if v_minor == 'tensor':
                    oprs = {'cat': torch.cat, 'stack': torch.stack}
                elif v_minor == 'numpy':
                    oprs = {'cat': np.concatenate, 'stack': np.stack}
                elif v_minor == 'pandas':
                    oprs = {'cat': pd.concat, 'stack': pd.concat}
                elif v_minor == 'native':
                    oprs = {'cat': torch.tensor, 'stack': torch.tensor}
                else:
                    oprs = {'cat': lambda x: x, 'stack': lambda x: x}

                opr = oprs['cat']
                if batch_size is not None and hasattr(v[0], '__len__') \
                        and len(v[0]) != batch_size and len(v[0]) == len(v[-1]):
                    opr = oprs['stack']

                stacked_results[n][k] = opr(results[n][k])

            else:
                stacked_results[n][k] = results[n][k]

    return stacked_results




def recursive_chunks(x, chunksize=None, n_chunks=None, partition=None, squeeze=False, dim=0):
    x_type = check_type(x)

    try:
        if (x_type.major == 'container') and (x_type.minor == 'dict'):
            gen = {k: recursive_chunks(v, chunksize=chunksize, n_chunks=n_chunks,
                                       partition=partition, squeeze=squeeze, dim=dim) for k, v in x.items()}

            for i in itertools.count():
                d = {}
                for k, v in gen.items():
                    i, v = next(v)
                    d[k] = v

                yield i, d

        elif x_type.major == 'container':

            gen = [recursive_chunks(s, chunksize=chunksize, n_chunks=n_chunks, partition=partition,
                                    squeeze=squeeze, dim=dim) for s in x]
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
                                      squeeze=squeeze, dim=dim):
                yield k, c

    except StopIteration:
        return




def recursive_collate_chunks(*xs, dim=0, on='index', how='outer', method='tree'):
    x_type = check_type(xs[0])
    if x_type.major == 'container':

        values = []
        keys = []

        for k, _ in iter_container(xs[0]):
            values.append(recursive_collate_chunks(*[xi[k] for xi in xs], dim=dim, on=on, how=how, method=method))
            keys.append(k)

        if x_type.minor == 'dict':
            values = dict(zip(keys, values))

        return values

    else:
        return collate_chunks(*xs, dim=dim, on=on, how=how, method=method)


def collate_chunks(*xs, keys=None, dim=0, on='index', how='outer', method='tree'):
    if len(xs) == 0:
        return []

    if len(xs) == 1:
        return xs[0]

    x = list(xs)

    x_type = check_type(x[0], check_element=False)

    if x_type.major == 'container' and x_type.minor == 'dict':
        dictionary = {}
        for xi in x:
            dictionary.update(xi)
        return dictionary

    if (x_type.major not in ['array', 'other']) or (dim == 1 and x_type.minor not in ['tensor', 'numpy', 'pandas']):
        return x

    if x_type.minor == 'tensor':
        return torch.cat(x, dim=dim)

    elif x_type.minor == 'numpy':
        return np.concatenate(x, axis=dim)

    elif x_type.minor == 'scipy_sparse':

        if dim == 0:
            return scipy.sparse.vstack(x)
        return scipy.sparse.hstack(x)

    elif x_type.minor == 'pandas':
        if on is None or dim == 0:
            if len(x[0].shape) == 1:
                x = [pd.Series(xi) for xi in x]
            return pd.concat(x, axis=dim)
        elif on == 'index':
            return recursive_merge(x, method=method, how=how, left_index=True, right_index=True)
        else:
            return recursive_merge(x, method=method, how=how, on=on)
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



def is_arange(x):
    x_type = check_type(x)

    if x_type.element in ['array', 'object', 'empty', 'none', 'unknown']:
        return False

    if x_type.element == 'str':
        pattern = re.compile(r'^(?P<prefix>.*?)(?P<number>\d+)(?P<suffix>.*?)$')
        df = []
        for xi in x:
            match = pattern.match(xi)
            if match:
                df.append(match.groupdict())
            else:
                return None, False
        df = pd.DataFrame(df)
        if not df['prefix'].nunique() == 1 or not df['suffix'].nunique() == 1:
            return None, False

        arr_x = df['number'].astype(int).values
    else:
        arr_x = np.array(x)

    try:
        arr_x = arr_x.astype(int)
        argsort = np.argsort(arr_x)
        arr_x = arr_x[argsort]
    except (ValueError, TypeError):
        return None, False

    isa = np.issubdtype(arr_x.dtype, np.number) and (np.abs(np.arange(len(x)) - arr_x).sum() == 0)

    if not isa:
        argsort = None

    return argsort, isa


def recursive_size(x):
    x_type = check_type(x)
    if x_type.major == 'container':

        keys = []
        values = []

        for k, v in iter_container(x):
            keys.append(k)
            values.append(recursive_size(v))

        if x_type.minor == 'dict':
            values = dict(zip(keys, values))

        return values

    else:

        return object_size(x, x_type=x_type)



def object_size(x, x_type=None):
    if x_type is None:
        x_type = check_type(x)
    if x_type.minor == 'tensor':
        return x.element_size() * x.nelement()
    elif x_type.minor in ['numpy', 'scipy_sparse']:
        return x.size * x.dtype.itemsize
    elif x_type.minor == 'pandas':
        try:
            return np.sum(x.memory_usage(index=True, deep=True))
        except:
            return x.size * x.dtype.itemsize
    elif x_type.minor == 'list':
        if len(x) <= 1000:
            return np.sum([sys.getsizeof(i) for i in x])
        ind = np.random.randint(len(x), size=(1000,))
        return len(x) * np.mean([sys.getsizeof(x[i]) for i in ind])
    else:
        return sys.getsizeof(x)


def recursive(func):
    def apply_recursively(x, *args, **kwargs):

        if is_container(x):

            keys = []
            values = []

            for k, v in iter_container(x):
                keys.append(k)
                values.append(apply_recursively(v, *args, **kwargs))

            if isinstance(x, dict):
                values = dict(zip(keys, values))

            return values

        else:

            return func(x, *args, **kwargs)

    return apply_recursively




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



def is_container(x):
    if isinstance(x, dict):
        return True
    if isinstance(x, list):

        if len(x) < 100:
            sampled_indices = range(len(x))
        else:
            sampled_indices = np.random.randint(len(x), size=(100,))

        elt0 = None
        for i in sampled_indices:
            elt = check_element_type(x[i])

            if elt0 is None:
                elt0 = elt

            if elt != elt0:
                return True

            if elt in ['array', 'none']:
                return True

    return False



@recursive
def recursive_batch(x, index):
    if x is None:
        return None
    elif hasattr(x, 'iloc'):
        return x.iloc[index]
    else:
        return x[index]


@recursive
def recursive_len(x):
    x_type = check_type(x)

    if x_type.minor == 'scipy_sparse':
        return x.shape[0]

    if x_type.element == 'none':
        return 0

    if hasattr(x, '__len__'):
        try:
            return len(x)
        except TypeError:
            return 1

    if x is None:
        return 0

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
    elif x_type.minor == 'pandas':
        return x[columns]
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
    elif isinstance(x, list):
        for xi in x:
            try:
                return recursive_device(xi)
            except AttributeError:
                # case of None
                pass
    return x.device


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

    metadata = None
    minor_type = check_minor_type(x)
    if minor_type == 'pandas':
        metadata = x.columns
        x = x.values
    if minor_type in ['numpy', 'tensor', 'pandas', 'modin']:
        ind = tuple(slice(0, i, i // n) if i > n else slice(None) for i in x.shape)
        x = x.__getitem__(ind)
    if minor_type in ['list', 'tuple', 'set']:
        x = list(x)[::len(x) // nl]

    return str((minor_type, metadata, x)).encode('utf-8')


def recursive_keys(x):
    x_type = check_type(x)
    if x_type.major == 'container':

        keys = []
        values = []

        for k, v in iter_container(x):
            keys.append(k)
            values.append(recursive_keys(v))

        if all([v is None for v in values]):
            return keys

        if x_type.minor == 'dict':

            argsort, isarange = is_arange(keys)
            if not isarange:
                values = dict(zip(keys, values))
            else:
                values = [values[i] for i in argsort]

        return values

    return None


def recursive_size_summary(x, mode='sum'):
    x_type = check_type(x)

    if x_type.minor == 'dict':

        if mode == 'sum':
            return sum([recursive_size_summary(v, mode=mode) for v in x.values()])
        elif mode == 'max':
            return max([recursive_size_summary(v, mode=mode) for v in x.values()])
        else:
            raise NotImplementedError

    elif (x_type.minor in ['list', 'tuple']) and x_type.element in ['object', 'unknown', 'other']:

        if mode == 'sum':
            return sum([recursive_size_summary(s, mode=mode) for s in x])
        elif mode == 'max':
            return max([recursive_size_summary(s, mode=mode) for s in x])
        else:
            raise NotImplementedError

    elif x is None:
        return 0
    else:
        if x_type.minor == 'tensor':
            return x.element_size() * x.nelement()
        elif x_type.minor in ['numpy', 'scipy_sparse']:
            return x.size * x.dtype.itemsize
        elif x_type.minor == 'pandas':
            return np.sum(x.memory_usage(index=True, deep=True))
        else:
            return sys.getsizeof(x)


@recursive
def empty_elements(x):
    x_type = check_type(x)
    if x_type.minor in ['numpy', 'pandas', 'tensor', 'scipy_sparse']:
        return x.size == 0

    if x_type.minor in ['list', 'tuple', 'set', 'dict']:
        return len(x) == 0

    if x_type.minor == 'native':
        return x is None

    if hasattr(x, '__len__'):
        return x.__len__() == 0

    return False


def is_empty(x):
    x = empty_elements(x)
    x = recursive_flatten(x)
    return all(x)


def is_chunk(path, chunk_pattern='_chunk'):
    return path.is_file() and bool(re.search(rf'\d{6}{chunk_pattern}\.', str(path.name)))


def recursive_flatten(x, flat_array=False, x_type=None, tolist=True):

    if x_type is None:
        x_type = check_type(x)

    if x_type.major == 'container':
        l = []
        for i, xi in iter_container(x):
            l.extend(recursive_flatten(xi, flat_array=flat_array))
        return l
    else:
        if not flat_array or x_type.major == 'scalar':
            return [x]
        else:
            return recursive_flat_array(x, x_type=x_type, tolist=tolist)


def recursive_flatten_with_keys(x):
    x_type = check_type(x)

    if x_type.major == 'container':
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

    if x_type.minor in ['numpy', 'tensor']:
        x = x.flatten()
        if tolist:
            x = x.tolist()
        return x
    elif x_type.minor == 'pandas':
        x = x.values.flatten()
        if tolist:
            x = x.tolist()
        return x
    elif x_type.minor == 'scipy_sparse':
        x = x.toarray().flatten()
        if tolist:
            x = x.tolist()
        return x
    elif x_type.minor in ['list', 'tuple']:
        if x_type.element != 'array':
            return list(x)

        l = []
        for xi in x:
            l.extend(recursive_flat_array(xi, tolist=tolist))
        return l

    elif x_type.minor == 'native':
        return [x]

    else:
        return [x]