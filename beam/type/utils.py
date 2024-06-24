from enum import Enum
from collections import namedtuple
import random
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import PurePath
from functools import cached_property

from ..importer.lazy_importer import lazy_importer as lzi


TypeTuple = namedtuple('TypeTuple', 'major minor element')


class Types(Enum):
    array = 'array'
    scalar = 'scalar'
    container = 'container'
    none = 'none'
    other = 'other'
    native = 'native'
    dict = 'dict'
    list = 'list'
    tuple = 'tuple'
    set = 'set'
    tensor = 'tensor'
    numpy = 'numpy'
    pandas = 'pandas'
    scipy_sparse = 'scipy_sparse'
    unknown = 'unknown'
    slice = 'slice'
    counter = 'counter'
    object = 'object'
    empty = 'empty'
    path = 'path'
    beam_data = 'beam_data'
    beam_processor = 'beam_processor'
    polars = 'polars'
    cudf = 'cudf'
    pil = 'pil'


def is_cached_property(obj, attribute_name):
    # Access the class attribute directly without triggering the property
    attr = getattr(type(obj), attribute_name, None)
    return isinstance(attr, cached_property)


def is_polars(x):
    pl = lzi.polars
    return pl and isinstance(x, pl.DataFrame)


def is_tensor(x):
    torch = lzi.torch
    return torch and torch.is_tensor(x)


def is_torch_scalar(x):
    return is_tensor(x) and (not len(x.shape))


def is_scipy_sparse(x):
    scipy = lzi.scipy
    return scipy and scipy.sparse.issparse(x)


def is_cudf(x):
    cudf = lzi.cudf
    return cudf and isinstance(x, cudf.DataFrame)


# def is_pil(x):
#     pil = lzi.PIL
#     return pil and isinstance(x, pil.Image.Image)

def is_pil(x):
    pil = lzi.pil_image
    return pil and isinstance(x, pil.Image)


def check_element_type(x, minor=None):

    if minor is None:
        minor = check_minor_type(x)
    unknown = (minor == 'other')

    if not unknown and not np.isscalar(x) and not is_torch_scalar(x):
        if minor == 'path':
            return 'path'
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
    if isinstance(x, np.ndarray):
        return 'numpy'
    if isinstance(x, pd.core.base.PandasObject):
        return 'pandas'
    if is_tensor(x):
        return 'tensor'
    if isinstance(x, dict):
        return 'dict'
    if isinstance(x, list):
        return 'list'
    if isinstance(x, tuple):
        return 'tuple'
    if isinstance(x, set):
        return 'set'
    if isinstance(x, slice):
        return 'slice'
    if isinstance(x, Counter):
        return 'counter'
    elif is_scalar(x):
        return 'scalar'
    if is_polars(x):
        return 'polars'
    if is_scipy_sparse(x):
        return 'scipy_sparse'
    elif is_pil(x):
        return 'pil'
    elif isinstance(x, PurePath) or is_beam_path(x):
        return 'path'
    elif is_beam_data(x):
        return 'beam_data'
    elif is_beam_processor(x):
        return 'beam_processor'
    if is_cudf(x):
        return 'cudf'
    else:
        return 'other'


def elt_of_list(x, sample_size=20):

    if isinstance(x, set):
        # assuming we are in the case of a set
        elements = random.sample(list(x), sample_size)
    else:
        if len(x) < sample_size:
            ind = list(range(len(x)))
        else:
            ind = np.random.randint(len(x), size=(sample_size,))
        elements = [x[i] for i in ind]

    elt = None
    t0 = type(elements[0])
    for e in elements[1:]:
        if type(e) != t0:
            elt = 'object'
            break

    if elt is None:
        elt = check_element_type(elements[0])

    return elt


def is_scalar(x):
    return np.isscalar(x) or is_torch_scalar(x)


def _check_type(x, minor=True, element=True):
    '''

    returns:

    <major type>, <minor type>, <elements type>

    major type: container, array, scalar, none, other
    minor type: dict, list, tuple, set, tensor, numpy, pandas, scipy_sparse, native, none, slice, counter, other
    elements type: array, int, float, complex, bool, str, object, empty, none, unknown

    '''

    if is_scalar(x):
        mjt = 'scalar'
        if minor:
            if type(x) in [int, float, str, complex, bool]:
                mit = 'native'
            else:
                mit = check_minor_type(x)
        else:
            mit = 'na'
        elt = check_element_type(x, minor=mit if mit != 'na' else None) if element else 'na'

    elif isinstance(x, dict):

        if isinstance(x, Counter):
            mjt = 'counter'
            mit = 'counter'
            elt = 'counter'
        else:
            mjt = 'container'
            mit = 'dict'

            if element:
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

    elif isinstance(x, PurePath) or is_beam_path(x):
        mjt = 'path'
        mit = 'path'
        elt = 'path'

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
                elt = elt_of_list(x)

            if elt in ['array', 'object', 'none']:
                mjt = 'container'

        mit = check_minor_type(x) if minor else 'na'

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


def is_container(x):

    if isinstance(x, dict):
        if isinstance(x, Counter):
            return False
        return True
    if isinstance(x, list) or isinstance(x, tuple):

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

            # path is needed here since we want to consider a list of paths as a container
            if elt in ['array', 'none', 'object', 'path']:
                return True

    return False


def is_beam_data(x):
    if hasattr(x, 'beam_class_name') and 'BeamData' in x.beam_class_name:
        return True
    return False


def is_beam_processor(x):
    if hasattr(x, 'beam_class_name') and 'Processor' in x.beam_class_name:
        return True
    return False


def is_beam_path(x):
    if hasattr(x, 'beam_class_name') and 'PureBeamPath' in x.beam_class_name:
        return True
    return False


def is_beam_config(x):
    if hasattr(x, 'beam_class_name') and 'BeamConfig' in x.beam_class_name:
        return True
    return False


def is_beam_resource(x):
    if hasattr(x, 'beam_class_name') and 'BeamResource' in x.beam_class_name:
        return True
    return False
