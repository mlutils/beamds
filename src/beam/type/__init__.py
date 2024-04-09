from .core import BeamType
from .utils import check_minor_type, check_element_type, is_scalar


def check_type(x, major=True, minor=True, element=True):
    return BeamType.check(x, major=major, minor=minor, element=element)
