from functools import cached_property
from typing import Union, Any

from dataclasses import dataclass
from .utils import check_type, check_element_type, check_minor_type, is_scalar


@dataclass
class BeamType:
    _ref: Any
    _major: Union[str, None] = None
    _minor: Union[str, None] = None
    _element: Union[str, None] = None

    @staticmethod
    def repr_subtype(x):
        if x is None:
            return 'N/A'
        return x

    @cached_property
    def major(self):
        if self._major is None:
            self._major = check_type(self._ref, check_minor=False, check_element=False).major
        return self._major

    @cached_property
    def minor(self):
        if self._minor is None:
            self._minor = check_minor_type(self._ref)
        return self._minor

    @cached_property
    def element(self):
        if self._element is None:
            self._element = check_element_type(self._ref)
        return self._element

    def __str__(self):
        return f"{self.repr_subtype(self._major)}-{self.repr_subtype(self._minor)}-{self.repr_subtype(self._element)}"

    @cached_property
    def is_scalar(self):
        if self._major is None:
            return is_scalar(self._ref)
        return self._major == 'scalar'

    @cached_property
    def is_dataframe(self):
        return self._major == 'container' and self._minor == 'pandas'

    @classmethod
    def check(cls, x, major=True, minor=True, element=True):
        if major:
            x_type = check_type(x, check_minor=minor, check_element=element)
            return cls(_ref=x, _major=x_type.major, _minor=x_type.minor, _element=x_type.element)
        element = check_element_type(x) if element else None
        minor = check_minor_type(x) if minor else None
        return cls(_ref=x, _major=None, _minor=minor, _element=element)

