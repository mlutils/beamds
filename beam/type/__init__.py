
from .core import BeamType

if len([]):
    from .utils import (check_minor_type, check_element_type, is_scalar, is_container, is_beam_data, is_beam_path,
                        is_beam_processor, is_cached_property, Types, is_beam_resource, is_beam_config, is_pandas,
                        is_cudf, is_dataframe, is_pandas_dataframe, is_polars, is_tensor, is_pil,
                        is_torch_scalar, is_scipy_sparse, is_pandas_series)


__all__ = ['BeamType', 'check_minor_type', 'check_element_type', 'is_scalar', 'is_container', 'is_beam_data',
           'is_beam_path', 'is_beam_processor', 'is_cached_property', 'Types', 'is_beam_resource', 'is_beam_config',
           'is_pandas', 'is_cudf', 'is_dataframe', 'is_pandas_dataframe', 'is_polars', 'is_tensor', 'is_pil',
           'is_torch_scalar', 'is_scipy_sparse', 'is_pandas_series', 'check_type']


def check_type(x, major=True, minor=True, element=True) -> BeamType:
    return BeamType.check(x, major=major, minor=minor, element=element)


def __getattr__(name):
    if name == 'check_minor_type':
        from .utils import check_minor_type
        return check_minor_type
    elif name == 'check_element_type':
        from .utils import check_element_type
        return check_element_type
    elif name == 'is_scalar':
        from .utils import is_scalar
        return is_scalar
    elif name == 'is_container':
        from .utils import is_container
        return is_container
    elif name == 'is_beam_data':
        from .utils import is_beam_data
        return is_beam_data
    elif name == 'is_beam_path':
        from .utils import is_beam_path
        return is_beam_path
    elif name == 'is_beam_processor':
        from .utils import is_beam_processor
        return is_beam_processor
    elif name == 'is_cached_property':
        from .utils import is_cached_property
        return is_cached_property
    elif name == 'Types':
        from .utils import Types
        return Types
    elif name == 'is_beam_resource':
        from .utils import is_beam_resource
        return is_beam_resource
    elif name == 'is_beam_config':
        from .utils import is_beam_config
        return is_beam_config
    elif name == 'is_pandas':
        from .utils import is_pandas
        return is_pandas
    elif name == 'is_cudf':
        from .utils import is_cudf
        return is_cudf
    elif name == 'is_dataframe':
        from .utils import is_dataframe
        return is_dataframe
    elif name == 'is_pandas_dataframe':
        from .utils import is_pandas_dataframe
        return is_pandas_dataframe
    elif name == 'is_polars':
        from .utils import is_polars
        return is_polars
    elif name == 'is_tensor':
        from .utils import is_tensor
        return is_tensor
    elif name == 'is_pil':
        from .utils import is_pil
        return is_pil
    elif name == 'is_torch_scalar':
        from .utils import is_torch_scalar
        return is_torch_scalar
    elif name == 'is_scipy_sparse':
        from .utils import is_scipy_sparse
        return is_scipy_sparse
    elif name == 'is_pandas_series':
        from .utils import is_pandas_series
        return is_pandas_series
    else:
        AttributeError(f"module {__name__} has no attribute {name}")
