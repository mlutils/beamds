from collections import namedtuple
import random
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import PurePath
from functools import cached_property
import types


from ..importer import lazy_importer as lzi

function_types = (types.FunctionType, types.BuiltinFunctionType, types.MethodType, types.BuiltinMethodType)
TypeTuple = namedtuple('TypeTuple', 'major minor element')


# TODO: switch to class Types(Enum)
class Types:
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
    int = 'int'
    float = 'float'
    complex = 'complex'
    bool = 'bool'
    bytes = 'bytes'
    str = 'str'
    na = 'na'
    cls = 'class'
    function = 'function'
    method = 'method'
    PackedTensor = 'PackedTensor'
    PackedArray = 'PackedArray'

def is_nan(x):
    return (pd.isna(x) or (isinstance(x, float) and np.isnan(x)) or (isinstance(x, np.ndarray) and np.isnan(x).all())
            or (isinstance(x, pd.Series) and x.isna().all()) or (isinstance(x, pd.DataFrame) and x.isna().all().all()))

def is_function(obj, include_class=False):
    """
    Returns True if obj is a function, method, or callable.
    If include_class is True, callable class instances are included.
    """
    if include_class:
        return callable(obj) and not is_class_instance(obj)
    return isinstance(obj, function_types)

def is_class_instance(obj):
    """
    Returns True if obj is an instance of a user-defined class.
    """
    return not is_class_type(obj) and hasattr(obj, '__class__') and not isinstance(obj, function_types)

def is_class_type(obj):
    """
    Returns True if obj is a class type, excluding function types.
    """
    return isinstance(obj, type) and not isinstance(obj, function_types) and not isinstance(obj, type(None))

def is_class_method(obj):
    """
    Returns True if obj is a method bound to an instance of a class.
    """
    return isinstance(obj, types.MethodType) and hasattr(obj, '__self__') and is_class_instance(obj.__self__)


def is_cached_property(obj, attribute_name):
    # Access the class attribute directly without triggering the property
    attr = getattr(type(obj), attribute_name, None)
    return isinstance(attr, cached_property)


def is_polars(x):
    if not lzi.is_loaded('polars'):
        return False
    pl = lzi.polars
    return pl and isinstance(x, pl.DataFrame)


def is_tensor(x):
    if not lzi.is_loaded('torch'):
        return False
    torch = lzi.torch
    return torch and torch.is_tensor(x)


def is_torch_scalar(x):
    if not lzi.is_loaded('torch'):
        return False
    return is_tensor(x) and (not len(x.shape))


def is_scipy_sparse(x):
    if not lzi.is_loaded('scipy'):
        return False
    scipy = lzi.scipy
    return scipy and scipy.sparse.issparse(x)


def is_cudf(x):
    if not lzi.is_loaded('cudf'):
        return False
    cudf = lzi.cudf
    return cudf and isinstance(x, cudf.DataFrame)


def is_pandas(x):
    return isinstance(x, pd.core.base.PandasObject)


def is_dataframe(x):
    return is_pandas(x) or is_polars(x) or is_cudf(x)


def is_pandas_dataframe(x):
    return isinstance(x, pd.DataFrame)


def is_pandas_series(x):
    return isinstance(x, pd.Series)


# def is_pil(x):
#     pil = lzi.PIL
#     return pil and isinstance(x, pil.Image.Image)

def is_pil(x):
    if not lzi.is_loaded('PIL'):
        return False
    pil = lzi.pil_image
    return pil and isinstance(x, pil.Image)


def check_element_type(x, minor=None):

    if minor is None:
        minor = check_minor_type(x)
    unknown = (minor == Types.other)

    if not unknown and not np.isscalar(x) and not is_torch_scalar(x):
        if minor == Types.path:
            return Types.path
        return Types.array

    if pd.isna(x):
        return Types.none

    if hasattr(x, 'dtype'):
        # this case happens in custom classes that have a dtype attribute
        if unknown:
            return Types.other

        t = str(x.dtype).lower()
    else:
        t = str(type(x)).lower()

    if 'int' in t:
        return Types.int
    if 'bool' in t:
        return Types.bool
    if 'float' in t:
        return Types.float
    if 'str' in t:
        return Types.str
    if '<u3' in t:
        return Types.str
    if 's3' in t:
        return Types.bytes
    if 'bytes' in t:
        return Types.bytes
    if 'complex' in t:
        return Types.complex

    return Types.object


def check_minor_type(x):
    if isinstance(x, np.ndarray):
        return Types.numpy
    if isinstance(x, pd.core.base.PandasObject):
        return Types.pandas
    if is_tensor(x):
        return Types.tensor
    if isinstance(x, dict):
        return Types.dict
    if isinstance(x, list):
        return Types.list
    if isinstance(x, tuple):
        return Types.tuple
    if isinstance(x, set):
        return Types.set
    if isinstance(x, slice):
        return Types.slice
    if isinstance(x, Counter):
      return Types.counter
    if is_scalar(x):
      return Types.scalar
    if is_polars(x):
      return Types.polars
    if is_scipy_sparse(x):
      return Types.scipy_sparse
    if is_pil(x):
      return Types.pil
    if isinstance(x, PurePath) or is_beam_path(x):
      return Types.path
    if is_beam_data(x):
      return Types.beam_data
    if is_packed_tensor(x):
      return Types.PackedTensor
    if is_packed_array(x):
      return Types.PackedArray
    if is_beam_processor(x):
      return Types.beam_processor
    if is_cudf(x):
        return Types.cudf
    return Types.other


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
            elt = Types.object
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
        mjt = Types.scalar
        if minor:
            if type(x) in [int, float, str, complex, bool]:
                mit = Types.native
            else:
                mit = check_minor_type(x)
        else:
            mit = Types.na
        elt = check_element_type(x, minor=mit if mit != Types.na else None) if element else Types.na

    elif isinstance(x, dict):

        if isinstance(x, Counter):
            mjt = Types.counter
            mit = Types.counter
            elt = Types.counter
        else:
            mjt = Types.container
            mit = Types.dict

            if element:
                if len(x):
                    elt = check_element_type(next(iter(x.values())))
                else:
                    elt = Types.empty
            else:
                elt = Types.na

    elif x is None:
        mjt = Types.none
        mit = Types.none
        elt = Types.none

    elif isinstance(x, slice):
        mjt = Types.slice
        mit = Types.slice
        elt = Types.slice

    elif isinstance(x, PurePath) or is_beam_path(x):
        mjt = Types.path
        mit = Types.path
        elt = Types.path

    elif is_class_method(x):
        mjt = Types.method
        mit = Types.method
        elt = Types.method

    elif is_function(x, include_class=False):
        mjt = Types.function
        mit = Types.function
        elt = Types.function

    else:

        elt = Types.unknown

        if hasattr(x, '__len__'):
            mjt = Types.array
        else:
            mjt = Types.other
        if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, set):
            if not len(x):
                elt = Types.empty
            else:
                elt = elt_of_list(x)

            if elt in [Types.array, Types.object, Types.none]:
                mjt = Types.container

        mit = check_minor_type(x) if minor else Types.na

        if elt:
            if mit in [Types.numpy, Types.tensor, Types.pandas, Types.scipy_sparse]:
                if mit == Types.pandas:
                    dt = str(x.values.dtype)
                else:
                    dt = str(x.dtype)
                if 'float' in dt:
                    elt = Types.float
                elif 'int' in dt:
                    elt = Types.int
                else:
                    elt = Types.object

        if mit == Types.other:
            mjt = Types.other
            elt = Types.other

    return TypeTuple(major=mjt, minor=mit, element=elt)


def is_container(x):

    def check_elements(xi, valid_types):
        if len(xi) < 100:
            sampled_indices = range(len(xi))
        else:
            sampled_indices = np.random.randint(len(xi), size=(100,))

        elt0 = None
        for i in sampled_indices:
            elt = check_element_type(xi[i])

            if elt0 is None:
                elt0 = elt

            if elt != elt0:
                return True

            # path is needed here since we want to consider a list of paths as a container
            if elt in valid_types:
                return True

        return False

    if isinstance(x, dict):
        if isinstance(x, Counter):
            return False
        return check_elements(list(x.values()), valid_types=[Types.array, Types.none, Types.object, Types.path,
                                                             Types.str])
    if isinstance(x, list) or isinstance(x, tuple):
        return check_elements(x, valid_types=[Types.array, Types.none, Types.object, Types.path])

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

def is_packed_tensor(x):
    if hasattr(x, 'beam_class_name') and 'PackedTensor' == x.beam_class_name:
        return True
    return False

def is_packed_array(x):
    if hasattr(x, 'beam_class_name') and 'PackedArray' == x.beam_class_name:
        return True
    return False
