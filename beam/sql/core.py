
import datetime as _dt
import functools as _ft
import json as _json
import os as _os
import re as _re
import typing as _t
from dataclasses import dataclass as _dataclass
from pathlib import PurePosixPath as _PurePosixPath

import ibis
import numpy as _np
import pandas as _pd

__all__ = [
    "BeamIbis",
]


def _now():
    return _dt.datetime.now(tz=_dt.timezone.utc)


# ---------------------------------------------------------------------------
# Path handling & connection cache
# ---------------------------------------------------------------------------

_BQ_CONN_CACHE: dict[str, ibis.backends.bigquery.Backend] = {}


def _get_connection(project: str | None) -> ibis.backends.bigquery.Backend:  # type: ignore[name-defined]
    key = project or "_default_"
    if key not in _BQ_CONN_CACHE:
        _BQ_CONN_CACHE[key] = ibis.bigquery.connect(project_id=project)
    return _BQ_CONN_CACHE[key]


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class BeamIbis:
    """Path‑like BigQuery wrapper with an Elastic‑like fluent API."""

    # ---------------------------------------------------------------------
    # construction helpers
    # ---------------------------------------------------------------------
    _TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S%z"  # match BeamElastic for parity

    def __init__(
        self,
        path: str | _PurePosixPath | None = None,
        *,
        project: str | None = None,
        dataset: str | None = None,
        table: str | None = None,
        expr: ibis.Expr | None = None,
        fields: list[str] | None = None,
        sort_by: str | None = None,
        max_rows: int | None = None,
    ) -> None:
        # Normalise path like "bigquery://project/dataset/table"
        if isinstance(path, str) and path.startswith("bigquery://"):
            parts = _PurePosixPath(path[len("bigquery://") :]).parts  # drop scheme
            if parts:
                project = project or parts[0]
            if len(parts) > 1:
                dataset = dataset or parts[1]
            if len(parts) > 2:
                table = table or parts[2]
        self.project = project
        self.dataset = dataset
        self.table = table
        self._expr = expr  # ibis expression (lazy)
        self._fields = fields  # projection
        self._sort_by = sort_by
        self._max_rows = max_rows or 10_000

        # Connection is established lazily to avoid needless auth in pickled objs
        self._conn: ibis.backends.bigquery.Backend | None = None

    # ------------------------------------------------------------------
    # private utilities
    # ------------------------------------------------------------------
    @property
    def _connection(self):
        if self._conn is None:
            self._conn = _get_connection(self.project)
        return self._conn

    # ------------------------------------------------------------------
    # path helpers (mimic pathlib / BeamElastic behaviour)
    # ------------------------------------------------------------------
    def gen(self, *path_parts, **kwargs):
        """Return *new* instance, cloning current settings and overriding *kwargs*."""
        merged = dict(
            project=self.project,
            dataset=self.dataset,
            table=self.table,
            expr=self._expr,
            fields=self._fields,
            sort_by=self._sort_by,
            max_rows=self._max_rows,
        )
        merged.update(kwargs)
        return type(self)(None, **merged)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # navigation – /, [] and .joinpath() like PurePath
    # ------------------------------------------------------------------
    def __truediv__(self, key: str):  # / operator
        if self.project is None:
            return self.gen(project=key)
        if self.dataset is None:
            return self.gen(dataset=key)
        if self.table is None:
            return self.gen(table=key)
        raise ValueError("Cannot descend deeper than table level – BigQuery has only 3 levels")

    # Allow dict‑like field selection: client["col1"] -> restrict projection
    def __getitem__(self, item: str | list[str]):
        if isinstance(item, str):
            item = [item]
        fields = item
        if self._fields is not None:
            missing = set(fields) - set(self._fields)
            if missing:
                raise ValueError(f"Cannot select unknown fields {missing} not in {self._fields}")
        return self.gen(fields=fields)

    # ------------------------------------------------------------------
    # level inspection (root/dataset/table/query)  – similar to BeamElastic.level
    # ------------------------------------------------------------------
    @property
    def level(self):
        if self.table is not None and self._expr is not None:
            return "query"
        if self.table is not None:
            return "table"
        if self.dataset is not None:
            return "dataset"
        return "root"

    # ------------------------------------------------------------------
    # building ibis expression lazily
    # ------------------------------------------------------------------
    def _base_table_expr(self) -> ibis.Table:
        if self.level in {"root", "dataset"}:
            raise ValueError("Table expression requires table-level path")
        table = self._connection.table(self.table, dataset=self.dataset)
        if self._fields is not None:
            table = table[self._fields]
        if self._sort_by is not None:
            table = table.sort_by(self._sort_by)
        return table

    def _current_expr(self) -> ibis.Expr:
        if self._expr is not None:
            return self._expr
        if self.level == "table":
            return self._base_table_expr()
        raise ValueError("No expression at this level – navigate into a table or set a query")

    # ------------------------------------------------------------------
    # filters (parallel to BeamElastic.filter_* helpers)
    # ------------------------------------------------------------------
    def _with_filter(self, predicate: ibis.Expr):
        base = self._current_expr()
        new_expr = base.filter(predicate)
        return self.gen(expr=new_expr)

    def parse_column(self, field: str | None):
        if field is None:
            # default: first field if projection set, else raise
            if self._fields:
                return self._fields[0]
            raise ValueError("Field must be specified when no default context")
        return field

    # Equality / membership
    def filter_term(self, value, field: str | None = None):
        col = self.parse_column(field)
        return ibis.field(col) == value

    def filter_terms(self, values, field: str | None = None):
        col = self.parse_column(field)
        return ibis.field(col).isin(list(values))

    # Range filters
    def filter_gte(self, value, field: str | None = None):
        col = self.parse_column(field)
        return ibis.field(col) >= value

    def filter_gt(self, value, field: str | None = None):
        col = self.parse_column(field)
        return ibis.field(col) > value

    def filter_lte(self, value, field: str | None = None):
        col = self.parse_column(field)
        return ibis.field(col) <= value

    def filter_lt(self, value, field: str | None = None):
        col = self.parse_column(field)
        return ibis.field(col) < value

    # Time range filter similar to TimeFilter => accept kwargs like start/end/period
    def filter_time_range(
        self,
        *,
        field: str | None = None,
        start: _dt.datetime | str | None = None,
        end: _dt.datetime | str | None = None,
        period: _dt.timedelta | str | None = None,
    ):
        field = self.parse_column(field)

        if start and isinstance(start, str):
            if start == "now":
                start = _now()
            else:
                start = _dt.datetime.fromisoformat(start)
        if end and isinstance(end, str):
            if end == "now":
                end = _now()
            else:
                end = _dt.datetime.fromisoformat(end)
        if period is not None and isinstance(period, str):
            _period_re = _re.compile(r"(\d+)([smhdw])")
            m = _period_re.fullmatch(period.strip())
            if not m:
                raise ValueError("Invalid period string – use e.g. '5d', '12h'")
            qty, unit = m.groups()
            qty = int(qty)
            delta_map = {
                "s": _dt.timedelta(seconds=qty),
                "m": _dt.timedelta(minutes=qty),
                "h": _dt.timedelta(hours=qty),
                "d": _dt.timedelta(days=qty),
                "w": _dt.timedelta(weeks=qty),
            }
            period = delta_map[unit]

        # Resolve start & end from period
        if period is not None:
            if start is None and end is None:
                end = _now()
                start = end - period
            elif start is None:
                start = end - period
            elif end is None:
                end = start + period
        predicate = True
        if start is not None:
            predicate = predicate & (ibis.field(field) >= start)
        if end is not None:
            predicate = predicate & (ibis.field(field) <= end)
        return predicate

    # Fluent wrappers like with_filter_term etc.
    def with_filter_term(self, value, field: str | None = None):
        return self._with_filter(self.filter_term(value, field))

    def with_filter_terms(self, values, field: str | None = None):
        return self._with_filter(self.filter_terms(values, field))

    def with_filter_gte(self, value, field: str | None = None):
        return self._with_filter(self.filter_gte(value, field))

    def with_filter_gt(self, value, field: str | None = None):
        return self._with_filter(self.filter_gt(value, field))

    def with_filter_lte(self, value, field: str | None = None):
        return self._with_filter(self.filter_lte(value, field))

    def with_filter_lt(self, value, field: str | None = None):
        return self._with_filter(self.filter_lt(value, field))

    def with_filter_time_range(self, **kwargs):
        return self._with_filter(self.filter_time_range(**kwargs))

    # Operator overloads for & / |
    def __and__(self, other: "BeamIbis"):
        if not isinstance(other, BeamIbis):
            raise TypeError("& expects another BeamBigQuery instance")
        if (self.project, self.dataset, self.table) != (other.project, other.dataset, other.table):
            raise ValueError("Cannot combine queries from different tables")
        combined = self._current_expr().filter(other._current_expr())  # this will fail; easier: & over preds unsupported
        # Simpler: convert to ibis.bool exprs and combine. We'll treat _expr as predicate only if not table.
        raise NotImplementedError("Chaining two BeamBigQuery queries is not yet implemented – use ._with_filter")

    def __or__(self, other: "BeamIbis"):
        raise NotImplementedError("OR combination not yet supported – use ibis.boolean_or explicitly")

    # Comparison overloads – produce predicate (like BeamElastic)
    def __eq__(self, other):  # noqa: D401, E743
        return self.filter_term(other)

    def __ge__(self, other):
        return self.filter_gte(other)

    def __gt__(self, other):
        return self.filter_gt(other)

    def __le__(self, other):
        return self.filter_lte(other)

    def __lt__(self, other):
        return self.filter_lt(other)

    def groupby(self, fields: str | list[str]):
        if isinstance(fields, str):
            fields = [fields]
        return BeamIbis.GroupByHelper(self, fields)

    # ------------------------------------------------------------------
    # materialisers – as_df(), as_dict(), etc.
    # ------------------------------------------------------------------
    def as_df(self, limit: int | None = None):
        expr = self._current_expr()
        lim = limit or self._max_rows
        return expr.execute(limit=lim)

    def as_dict(self, limit: int | None = None):
        return self.as_df(limit=limit).to_dict(orient="records")

    def as_pl(self, limit: int | None = None):
        import polars as pl
        return pl.from_pandas(self.as_df(limit))

    def as_cudf(self, limit: int | None = None):
        import cudf
        return cudf.from_pandas(self.as_df(limit))

    def head(self, n: int = 5):
        return self.as_df(limit=n)

    # ------------------------------------------------------------------
    # Writing helpers (only DataFrame -> table append for now)
    # ------------------------------------------------------------------
    def write(self, df: _pd.DataFrame, *, if_exists: str = "append", **load_kwargs):
        if self.level != "table":
            raise ValueError("Write only supported at table level")
        tmp_uri = None
        try:
            tmp_uri = f"gs://{_os.environ.get('TEMP_GCS_BUCKET')}/beam_tmp_{_now().timestamp()}.parquet"
        except Exception as exc:  # pragma: no cover – env may not exist
            raise RuntimeError("Set TEMP_GCS_BUCKET env var for staging") from exc
        df.to_parquet("/tmp/_beam_tmp.parquet")
        from google.cloud import storage  # lazy import

        client = storage.Client()
        bucket_name, blob_name = tmp_uri[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename("/tmp/_beam_tmp.parquet")

        tbl = self._connection.load_data(
            _pd.read_parquet("/tmp/_beam_tmp.parquet"),
            table_name=self.table,
            dataset=self.dataset,
            project=self.project,
            if_exists=if_exists,
            **load_kwargs,
        )
        return tbl

    # ------------------------------------------------------------------
    # misc utils
    # ------------------------------------------------------------------
    def count(self):
        return int(self._current_expr().count().execute())

    def __repr__(self):
        parts = ["bigquery://"]
        if self.project:
            parts.append(self.project)
        if self.dataset:
            parts.append("/" + self.dataset)
        if self.table:
            parts.append("/" + self.table)
        s = "".join(parts)
        if self._expr is not None and self.level == "query":
            s += " | expr=[…]"
        if self._fields:
            s += f" | fields={self._fields}"
        if self._sort_by:
            s += f" | sort={self._sort_by}"
        return s
