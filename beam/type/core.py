from functools import cached_property
from typing import Union, Any

from dataclasses import dataclass
from .utils import _check_type, check_element_type, check_minor_type, is_scalar, Types


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
            self._major = _check_type(self._ref, minor=False, element=False).major
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

    def __repr__(self):
        return (f"BeamType(major={self.repr_subtype(self._major)}, minor={self.repr_subtype(self._minor)}, "
                f"element={self.repr_subtype(self._element)})")

    def __str__(self):
        return f"{self.repr_subtype(self._major)}-{self.repr_subtype(self._minor)}-{self.repr_subtype(self._element)}"

    @cached_property
    def is_scalar(self):
        if self._major is None:
            return is_scalar(self._ref)
        return self._major == Types.scalar

    @cached_property
    def is_array(self):
        return self._major == Types.array

    @cached_property
    def is_dataframe(self):
        return self._minor in [Types.pandas, Types.polars, Types.cudf]

    @cached_property
    def is_path(self):
        return self._minor == Types.path

    @cached_property
    def is_data_array(self):
        return self._minor in [Types.numpy, Types.tensor, Types.polars, Types.cudf, Types.pandas, Types.scipy_sparse]

    @cached_property
    def is_torch(self):
        return self._minor in [Types.tensor, Types.PackedTensor]

    @cached_property
    def is_str(self):
        return self._major == Types.scalar and self._element == Types.str

    @classmethod
    def check(cls, x, major=True, minor=True, element=True):
        if major:
            x_type = _check_type(x, minor=minor, element=element)
            return cls(_ref=x, _major=x_type.major, _minor=x_type.minor, _element=x_type.element)
        element = check_element_type(x) if element else None
        minor = check_minor_type(x) if minor else None
        return cls(_ref=x, _major=None, _minor=minor, _element=element)

    @classmethod
    def check_major(cls, x):
        return cls.check(x, major=True, minor=False, element=False)

    @classmethod
    def check_minor(cls, x):
        return cls.check(x, major=False, minor=True, element=False)

    @classmethod
    def check_element(cls, x):
        return cls.check(x, major=False, minor=False, element=True)

    @staticmethod
    def check_if_data_array(x):
        return BeamType.check_minor(x).is_data_array

    @staticmethod
    def check_if_dataframe(x):
        return BeamType.check_minor(x).is_dataframe

    @staticmethod
    def check_if_array(x):
        return BeamType.check_major(x).is_array

    @staticmethod
    def check_if_scalar(x):
        return BeamType.check_major(x).is_scalar

