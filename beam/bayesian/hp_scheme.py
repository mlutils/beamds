from __future__ import annotations
from typing import Any, ClassVar, Mapping, Literal, get_origin, get_args, List
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
    _literal_maps: ClassVar[dict[str, dict[Any, int]]] = {}
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
        if cls._num_fields_w:
            return
        num_w, cat, lit_map, num_cols = [], [], {}, []
        cursor = 0
        for name, field in cls.model_fields.items():
            ann = field.annotation
            width = 1 if ann is float else cls._fixed_width(field) or 0
            if width:
                num_w.append((name, width));
                cursor += width
                # expand column names
                if width == 1:
                    num_cols.append(name)
                else:
                    num_cols.extend(f"{name}_{i}" for i in range(width))
            elif ann is int or (isinstance(ann, type) and issubclass(ann, Enum)):
                cat.append(name)
            elif cls._is_literal(ann):
                cat.append(name)
                lit_map[name] = {v: i for i, v in enumerate(get_args(ann))}
            else:
                raise TypeError(f"Unsupported type {ann!r} on field {name!r}")
        cls._num_fields_w = num_w
        cls._cat_fields = cat
        cls._literal_maps = lit_map
        cls._xnum_len = cursor
        cls._xcat_len = len(cat)
        cls._num_cols = num_cols
        cls._cat_cols = cat[:]  # 1-col per categorical

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

    # ───── encode ─────
    def encode(self, output_type="torch"):
        c = self.__class__;
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
                cat_vals.append(c._literal_maps[name][v])
            else:
                cat_vals.append(int(v))

        if output_type == "torch":
            x_cat = torch.tensor(cat_vals, dtype=torch.int64)
        elif output_type == "numpy":
            x_cat = np.asarray(cat_vals, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported output type: {output_type!r}")\

        return x_num, x_cat

    # ───── decode ─────
    def decode(self, x_num: np.ndarray | torch.Tensor, x_cat: np.ndarray | torch.Tensor) -> BaseParameters:
        """
        Decode numeric and categorical features back to the model instance.
        :param x_num: Numeric features as a numpy array.
        :param x_cat: Categorical features as a numpy array.
        :return: An instance of the model with decoded values.
        """
        c = self.__class__
        c._build()
        data = {}
        offset = 0
        for name, width in c._num_fields_w:
            data[name] = x_num[offset:offset + width].tolist() if width > 1 else float(x_num[offset])
            offset += width
        for i, name in enumerate(c._cat_fields):
            data[name] = c._literal_maps[name][x_cat[i]] if name in c._literal_maps else int(x_cat[i])
        return c(**data)

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
