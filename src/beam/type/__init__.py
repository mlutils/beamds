from .core import BeamType
from .utils import (check_minor_type, check_element_type, is_scalar, is_container, is_beam_data, is_beam_path,
                    is_beam_processor, is_cached_property)


def check_type(x, major=True, minor=True, element=True):
    return BeamType.check(x, major=major, minor=minor, element=element)
