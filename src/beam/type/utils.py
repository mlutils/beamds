from collections import namedtuple
import random
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import PurePath

from ..path import PureBeamPath

try:
    import modin.pandas as mpd

    has_modin = True
except ImportError:
    mpd = None
    has_modin = False

try:
    import torch

    has_torch = True
except ImportError:
    has_torch = False

try:
    import scipy

    has_scipy = True
except ImportError:
    has_scipy = False

try:
    import polars as pl
    has_polars = True
except ImportError:
    has_polars = False

# try:
#     import cudf
#     has_cudf = True
# except ImportError:
#     has_cudf = False

TypeTuple = namedtuple('TypeTuple', 'major minor element')


def check_element_type(x):
    unknown = (check_minor_type(x) == 'other')

    if not unknown and not np.isscalar(x) and (has_torch and not (torch.is_tensor(x) and (not len(x.shape)))):
        return 'array'

    if pd.isna(x):
        return 'none'

    if hasattr(x, 'dtype'):
        # this case happens in custom classes that have a dtype attribute
        if unknown:
            return 'other'

        t = str(x.dtype).lower()
    else:
        t = str(type(x)).lower()

    if 'int' in t:
        return 'int'
    if 'bool' in t:
        return 'bool'
    if 'float' in t:
        return 'float'
    if 'str' in t:
        return 'str'
    if 'complex' in t:
        return 'complex'

    return 'object'


def check_minor_type(x):
    if has_torch and isinstance(x, torch.Tensor):
        return 'tensor'
    if isinstance(x, np.ndarray):
        return 'numpy'
    if isinstance(x, pd.core.base.PandasObject):
        return 'pandas'
    if isinstance(x, dict):
        return 'dict'
    if isinstance(x, list):
        return 'list'
    if isinstance(x, tuple):
        return 'tuple'
    if isinstance(x, set):
        return 'set'
    if has_modin and isinstance(x, mpd.base.BasePandasDataset):
        return 'modin'
    if has_scipy and scipy.sparse.issparse(x):
        return 'scipy_sparse'
    if has_polars and isinstance(x, pl.DataFrame):
        return 'polars'
    if isinstance(x, PurePath) or isinstance(x, PureBeamPath):
        return 'path'
    elif is_scalar(x):
        return 'scalar'
    else:
        return 'other'


def elt_of_list(x):
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
            return 'object'

    return elt0


def is_scalar(x):
    return np.isscalar(x) or (has_torch and torch.is_tensor(x) and (not len(x.shape)))


def check_type(x, check_minor=True, check_element=True):
    '''

    returns:

    <major type>, <minor type>, <elements type>

    major type: container, array, scalar, none, other
    minor type: dict, list, tuple, set, tensor, numpy, pandas, scipy_sparse, native, none
    elements type: array, int, float, complex, bool, str, object, empty, none, unknown

    '''

    if is_scalar(x):
        mjt = 'scalar'
        if check_minor:
            if type(x) in [int, float, str, complex, bool]:
                mit = 'native'
            else:
                mit = check_minor_type(x)
        else:
            mit = 'na'
        elt = check_element_type(x) if check_element else 'na'

    elif isinstance(x, dict):
        mjt = 'container'
        mit = 'dict'

        if check_element:
            if len(x):
                elt = check_element_type(next(iter(x.values())))
            else:
                elt = 'empty'
        else:
            elt = 'na'

    elif x is None:
        mjt = 'none'
        mit = 'none'
        elt = 'none'

    elif isinstance(x, slice):
        mjt = 'slice'
        mit = 'slice'
        elt = 'slice'

    elif isinstance(x, Counter):
        mjt = 'counter'
        mit = 'counter'
        elt = 'counter'

    else:

        elt = 'unknown'

        if hasattr(x, '__len__'):
            mjt = 'array'
        else:
            mjt = 'other'
        if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, set):
            if not len(x):
                elt = 'empty'
            else:

                if len(x) < 20:
                    elts = [check_element_type(xi) for xi in x]

                else:

                    sample_size = 20
                    try:
                        ind = np.random.randint(len(x), size=(sample_size,))
                        elts = [check_element_type(x[i]) for i in ind]
                    except TypeError:
                        # assuming we are in the case of a set
                        random.sample(list(x), sample_size)

                set_elts = set(elts)
                if len(set_elts) == 1:
                    elt = elts[0]
                elif set_elts == {'int', 'float'}:
                    elt = 'float'
                else:
                    elt = 'object'

            if elt in ['array', 'object', 'none']:
                mjt = 'container'

        mit = check_minor_type(x) if check_minor else 'na'

        if elt:
            if mit in ['numpy', 'tensor', 'pandas', 'scipy_sparse']:
                if mit == 'pandas':
                    dt = str(x.values.dtype)
                else:
                    dt = str(x.dtype)
                if 'float' in dt:
                    elt = 'float'
                elif 'int' in dt:
                    elt = 'int'
                else:
                    elt = 'object'

        if mit == 'other':
            mjt = 'other'
            elt = 'other'

    return TypeTuple(major=mjt, minor=mit, element=elt)
