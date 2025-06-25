from __future__ import annotations
from typing import Any, ClassVar, Mapping, Literal, get_origin, get_args, List
from annotated_types import Interval, Le, Lt, Ge, Gt
from enum import Enum

import numpy as np
import torch
from pydantic import BaseModel, Field, conlist, create_model


class BaseParameters(BaseModel):
    """
    * x_num – float or fixed-length conlist(float, N)
    * x_cat – int, Enum, Literal[…]
    * x_num_col / x_cat_col – ordered column names produced by `encode()`
    """

    _num_fields_w: ClassVar[list[tuple[str, int]]] = []
    _cat_fields: ClassVar[list[str]] = []
    _literal_maps: ClassVar[dict[str, dict[str, dict]]] = {}
    _xnum_len: ClassVar[int] = 0
    _xcat_len: ClassVar[int] = 0
    _num_cols: ClassVar[list[str]] = []
    _cat_cols: ClassVar[list[str]] = []

    # ───── helpers ─────
    @staticmethod
    def _is_literal(t):
        return get_origin(t) is Literal

    @classmethod
    def _fixed_width(cls, field) -> int | None:
        ann = field.annotation
        if get_origin(ann) in (list, List) and get_args(ann) == (float,):
            for m in getattr(field, "metadata", ()):
                if getattr(m, "min_length", None) == getattr(m, "max_length", None) != 0:
                    return m.min_length
        return None

    # ───── metadata (lazy) ─────
    @classmethod
    def _build(cls):
        if cls._num_fields_w:  # already built
            return

        num_w, cat, lit_map, num_cols = [], [], {}, []
        cursor = 0
        for name, field in cls.model_fields.items():
            ann = field.annotation
            width = 1 if ann is float else cls._fixed_width(field) or 0
            if width:  # numeric
                num_w.append((name, width))
                cursor += width
                num_cols.extend(
                    (name,) if width == 1
                    else (f"{name}_{i}" for i in range(width))
                )
            elif ann is int or (isinstance(ann, type) and issubclass(ann, Enum)):
                cat.append(name)
            elif cls._is_literal(ann):  # Literal[…]
                cat.append(name)
                # forward  (value → code)  and reverse (code → value) maps
                lit_map[name] = {
                    "fwd": {v: i for i, v in enumerate(get_args(ann))},
                    "rev": {i: v for i, v in enumerate(get_args(ann))},
                }
            else:
                raise TypeError(f"Unsupported type {ann!r} on field {name!r}")

        cls._num_fields_w = num_w
        cls._cat_fields = cat
        cls._literal_maps = lit_map
        cls._xnum_len = cursor
        cls._xcat_len = len(cat)
        cls._num_cols = num_cols
        cls._cat_cols = cat[:]  # one col per categorical

    # ───── public properties ─────

    @classmethod
    @property
    def x_num_len(cls):
        cls._build(); return cls._xnum_len

    @classmethod
    @property
    def x_cat_len(cls):
        cls._build(); return cls._xcat_len

    @classmethod
    @property
    def x_num_col(cls):
        cls._build(); return cls._num_cols

    @classmethod
    @property
    def x_cat_col(cls):
        cls._build(); return cls._cat_cols

    @classmethod
    @property
    def len_x_num(cls):
        cls._build(); return cls._xnum_len

    @classmethod
    @property
    def len_x_cat(cls):
        cls._build(); return cls._xcat_len

    @classmethod
    @property
    def num_fields_w(cls):
        cls._build(); return cls._num_fields_w

    @classmethod
    @property
    def cat_fields_to_index_map(cls):
        """
        Returns a mapping from categorical field names to their index in the
        encoded categorical vector.
        """
        cls._build()
        return {name: i for i, name in enumerate(cls._cat_fields)}

    @classmethod
    @property
    def num_fields_to_index_map(cls):
        """
        Returns a mapping from numeric field names to their index in the
        encoded numeric vector.
        """
        cls._build()
        return {name: i for i, (name, _) in enumerate(cls._num_fields_w)}

    # ───── encode ─────
    def encode(self, output_type="torch"):
        c = self.__class__
        c._build()
        # numeric
        num_parts = []
        for name, width in c._num_fields_w:
            v = getattr(self, name)
            num_parts.append(
                np.array([v], dtype=np.float64) if width == 1
                else np.asarray(v, dtype=np.float64)
            )
        if output_type == "torch":
            x_num = torch.tensor(np.concatenate(num_parts) if num_parts else np.empty(0, dtype=np.float64), dtype=torch.float64)
        elif output_type == "numpy":
            x_num = np.concatenate(num_parts) if num_parts else np.empty(0, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported output type: {output_type!r}")
        # categorical
        cat_vals = []
        for name in c._cat_fields:
            v = getattr(self, name)
            if isinstance(v, Enum):
                cat_vals.append(int(v.value))
            elif name in c._literal_maps:
                cat_vals.append(c._literal_maps[name]['fwd'][v])
            else:
                cat_vals.append(int(v))

        if output_type == "torch":
            x_cat = torch.tensor(cat_vals, dtype=torch.int64)
        elif output_type == "numpy":
            x_cat = np.asarray(cat_vals, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported output type: {output_type!r}")\

        return x_num, x_cat

    @classmethod
    def get_bounds(cls) -> dict[str, tuple[int | float | None, int | float | None]]:
        """
        Return {field_name: (lower_bound, upper_bound)} for every numeric field
        that carries at least one bound.  Plain floats/ints without constraints
        are omitted.
        """
        bounds: dict[str, tuple[int | float | None, int | float | None]] = {}

        for name, field_info in cls.model_fields.items():
            metadata = field_info.metadata
            lower = upper = None

            for m in metadata:
                if isinstance(m, Interval):
                    # Interval(ge=..., le=...) or similar
                    lower = m.ge if m.ge is not None else m.gt
                    upper = m.le if m.le is not None else m.lt
                    break
                elif isinstance(m, Le):
                    # Le(upper=...)
                    upper = m.le
                elif isinstance(m, Lt):
                    # Lt(upper=...)
                    upper = m.lt
                elif isinstance(m, Ge):
                    # Ge(lower=...)
                    lower = m.ge
                elif isinstance(m, Gt):
                    # Gt(lower=...)
                    lower = m.gt

            # keep only if *something* is constrained
            if lower is not None or upper is not None:
                bounds[name] = (lower, upper)

        return bounds

    # ───── decode ─────

    @classmethod
    def decode_batch(cls, x_num, x_cat) -> List["BaseParameters"]:
        """
        Recreate a batch of model instances from raw numeric / categorical tensors.
        Works with either NumPy or Torch inputs.
        """
        cls._build()

        # make sure we have NumPy arrays
        if isinstance(x_num, torch.Tensor):
            x_num = x_num.detach().cpu().numpy()
        if isinstance(x_cat, torch.Tensor):
            x_cat = x_cat.detach().cpu().numpy()

        if x_num.ndim == 1:
            x_num = x_num.reshape(1, -1)
        if x_cat.ndim == 1:
            x_cat = x_cat.reshape(1, -1)
        if x_num.shape[0] != x_cat.shape[0]:
            raise ValueError("Numeric and categorical tensors must have the same number of rows.")
        if x_num.shape[1] != cls.x_num_len:
            raise ValueError(f"Numeric tensor must have {cls.x_num_len} columns, got {x_num.shape[1]}.")
        if x_cat.shape[1] != cls.x_cat_len:
            raise ValueError(f"Categorical tensor must have {cls.x_cat_len} columns, got {x_cat.shape[1]}.")
        data_list: List[BaseParameters] = []
        for i in range(x_num.shape[0]):
            data = cls.decode(x_num[i], x_cat[i])
            data_list.append(data)
        return data_list

    @classmethod
    def decode(
            cls,
            x_num: np.ndarray | torch.Tensor | list,
            x_cat: np.ndarray | torch.Tensor | list,
    ) -> "BaseParameters":
        """
        Recreate a model instance from raw numeric / categorical tensors.
        Works with either NumPy or Torch inputs.
        """
        cls._build()

        # make sure we have NumPy arrays
        if isinstance(x_num, torch.Tensor):
            x_num = x_num.detach().cpu().numpy()
        if isinstance(x_cat, torch.Tensor):
            x_cat = x_cat.detach().cpu().numpy()

        data: dict[str, Any] = {}
        offset = 0

        # numeric slice-by-slice
        for name, width in cls._num_fields_w:
            if width == 1:
                data[name] = float(x_num[offset])
            else:  # fixed-width array → Python list
                data[name] = x_num[offset: offset + width].astype(float).tolist()
            offset += width

        # categorical reconstruction
        for i, name in enumerate(cls._cat_fields):
            code = int(x_cat[i])
            ann = cls.model_fields[name].annotation

            if cls._is_literal(ann):
                # back-translate code → literal value
                data[name] = cls._literal_maps[name]["rev"][code]
            elif isinstance(ann, type) and issubclass(ann, Enum):
                data[name] = ann(code)  # Enum(value)
            else:  # plain int
                data[name] = code

        return cls(**data)

    # ───── tiny JSON-schema helper (unchanged) ─────
    @classmethod
    def from_json_schema(cls, schema: Mapping[str, Any]):
        title = schema.get("title", "SchemaModel")
        props, req = schema["properties"], set(schema.get("required", []))
        fields = {}
        for n, spec in props.items():
            t = spec["type"]
            if t == "number":
                ann = float
            elif t == "integer":
                ann = int
            elif t == "array" and spec["items"]["type"] == "number":
                m, M = spec.get("minItems"), spec.get("maxItems")
                if m != M: raise ValueError(f"{n}: fixed-length arrays only")
                ann = conlist(float, min_length=m, max_length=M)  # type: ignore
            elif t == "string" and "enum" in spec:
                ann = Literal[tuple(spec["enum"])]  # type: ignore[misc]
            else:
                raise ValueError(f"{n}: unsupported JSON-Schema fragment")
            fields[n] = (ann, Field(... if n in req else None))
        return create_model(title, __base__=cls, **fields)  # type: ignore[return-value]

    @classmethod
    def get_feature_values(cls, key: str, encoded=False) ->List[Any]:
        """
        Returns the values of a specific feature (key) across all instances of the model.
        This is useful for analyzing the distribution of a feature.
        """
        cls._build()
        if key not in cls.model_fields:
            raise ValueError(f"Feature '{key}' does not exist in the model.")

        # assert that the key is integer or enum or literal
        field_info = cls.model_fields[key]
        if not (field_info.annotation is int or
                (isinstance(field_info.annotation, type) and issubclass(field_info.annotation, Enum)) or
                cls._is_literal(field_info.annotation)):
            raise TypeError(f"Feature '{key}' must be of type int, Enum, or Literal.")

        # switch case of literal, enum, and int
        if cls._is_literal(field_info.annotation):
            # For Literal, return enumeration over values
            l = list(get_args(field_info.annotation))
            if encoded:
                l = [cls._literal_maps[key]['fwd'][v] for v in l]
            return l

        elif isinstance(field_info.annotation, type) and issubclass(field_info.annotation, Enum):
            # For Enum, return the enum values
            l = [e.value for e in field_info.annotation]
            if encoded:
                l = [cls._literal_maps[key]['fwd'][v] for v in l]
            return l
        else:
            # For int, use bounds if available, otherwise return None
            bounds = cls.get_bounds().get(key, (None, None))
            lower, upper = bounds
            if lower is None or upper is None:
                return None
            return list(range(lower, upper + 1)) if lower <= upper else None

    @classmethod
    def encode_batch(cls, x: list[dict], output_type="torch") -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of parameters into numeric and categorical tensors.
        :param x: list of dicts with parameters
        :return: (x_num, x_cat) where x_num is a 2D array of numeric features
                 and x_cat is a 1D array of categorical features.
        """
        cls._build()
        x_num_parts = []
        x_cat_parts = []

        for item in x:
            num_part, cat_part = cls(**item).encode(output_type=output_type)
            x_num_parts.append(num_part)
            x_cat_parts.append(cat_part)

        # Concatenate all numeric and categorical parts
        if output_type == "torch":
            x_num = torch.stack(x_num_parts) if x_num_parts else torch.empty((0, cls.x_num_len), dtype=torch.float64)
            x_cat = torch.stack(x_cat_parts) if x_cat_parts else torch.empty((0, cls.x_cat_len), dtype=torch.int64)
        elif output_type == "numpy":
            x_num = np.vstack(x_num_parts) if x_num_parts else np.empty((0, cls.x_num_len), dtype=np.float64)
            x_cat = np.vstack(x_cat_parts) if x_cat_parts else np.empty((0, cls.x_cat_len), dtype=np.int64)
        else:
            raise ValueError(f"Unsupported output type: {output_type!r}")

        return x_num, x_cat

# ────────── demo ──────────
if __name__ == "__main__":
    from enum import Enum
    from typing import Literal
    from pydantic import conlist


    class Mood(Enum): HAPPY = 0; SAD = 1


    class MyFeatures(BaseParameters):
        age: float
        accel: conlist(float, min_length=3, max_length=3)
        label: int
        mood: Mood
        switch: Literal["on", "off"]


    f = MyFeatures(age=2.5, accel=[0.1, 0.2, 9.8], label=4, mood=Mood.HAPPY, switch="off")
    x_num, x_cat = f.encode()
    print("x_num:", x_num)  # [2.5 0.1 0.2 9.8]
    print("x_cat:", x_cat)  # [4 0 1]
    print("x_num_col:", MyFeatures.x_num_col)  # ['age', 'accel_0', 'accel_1', 'accel_2']
    print("x_cat_col:", MyFeatures.x_cat_col)  # ['label', 'mood', 'switch']
