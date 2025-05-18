
import datetime as _dt
import os as _os
import re as _re
import typing as _t

import ibis
from ..path import PureBeamPath, BeamPath
import pandas as _pd


def _now():
    return _dt.datetime.now(tz=_dt.timezone.utc)


class BeamIbis(PureBeamPath):
    """Path‑like Ibis wrapper pandas+lazy query API."""

    _TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S%z"  # match BeamElastic for parity

    def __init__(
        self,
        *args,
        hostname=None, port=None, username=None, password=None, verify=False, fragment=None, client=None,
        expr: ibis.Expr | None = None,
        columns: list[str] | None = None,
        backend: str | None = None,
        backend_kwargs: dict[str, _t.Any] | None = None, **kwargs,
    ) -> None:

        super().__init__(*args, hostname=hostname, port=port, username=username, password=password,
                            fragment=fragment, client=client, **kwargs)

        self._expr = expr  # ibis expression (lazy)
        self._columns = columns  # projection

        # Connection is established lazily to avoid needless auth in pickled objs
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.verify = verify

        self._database = None
        self._table = None

    @property
    def project(self):
        if self.backend == "bigquery":
            return self.parts[0] if len(self.parts) > 0 else None
        # assume sqlite
        raise ValueError(f"Project not supported for {self.backend} backend")

    @property
    def dataset(self):
        if self.backend == "bigquery":
            return self.parts[1] if len(self.parts) > 1 else None
        raise ValueError(f"Dataset not supported for {self.backend} backend")

    @property
    def database(self):
        if self._database is None:
            path = BeamPath(*self.parts[:-1])
            if path.is_file() or any(path.parts[-1].endswith(ext) for ext in [".db", ".sqlite"]):
                d = str(path)
                self._table = self.parts[-1]
            else:
                d = self.path
                self._table = None

            self._database = d
        return self._database

    @property
    def table(self):

        if self._table is not None:
            return self._table

        if self.backend == "bigquery":
            return self.parts[2] if len(self.parts) > 2 else None
        elif self.backend == "sqlite":
            _ = self.database  # force database resolution
            return self._table

        raise ValueError(f"Table not supported for {self.backend} backend")

    def get_client(self):
        if self.backend == 'bigquery':
            c = ibis.bigquery.connect(host=self.hostname, port=self.port,
                username=self.username, password=self.password, verify=self.verify,
                project_id=self.project,
                dataset_id=self.dataset,
                **self.backend_kwargs,
            )
        else:
            # assume sqlite
            c = ibis.sqlite.connect(
                database=self.database,
                **self.backend_kwargs,
            )
        return c

    @property
    def client(self):
        if self._client is None:
            self._client = self.get_client()
        return self._client

    def gen(self, *path_parts, **kwargs):
        """Return *new* instance, cloning current settings and overriding *kwargs*."""
        merged = dict(
            expr=self._expr,
            columns=self._columns,
        )
        merged.update(kwargs)
        return type(self)(None, **merged)

    def gen(self, path, **kwargs):
        hostname = kwargs.pop('hostname', self.hostname)
        port = kwargs.pop('port', self.port)
        username = kwargs.pop('username', self.username)
        password = kwargs.pop('password', self.password)
        fragment = kwargs.pop('fragment', self.fragment)
        params = kwargs.pop('params', self.params)
        doc_cls = kwargs.pop('document', self._doc_cls)
        query = kwargs.pop('query', {})
        fields = kwargs.pop('fields', self.fields)
        sort_by = kwargs.pop('sort_by', self.sort_by)
        llm = kwargs.pop('llm', self.llm)
        q = kwargs.pop('q', self.q)

        # must be after extracting all other kwargs
        query = {**query, **kwargs}
        PathType = type(self)
        return PathType(path, client=self.client, hostname=hostname, port=port, username=username, fields=fields,
                        password=password, fragment=fragment, params=params, document=doc_cls, q=q, sort_by=sort_by,
                        llm=llm, **query)

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
        if self._columns is not None:
            missing = set(fields) - set(self._columns)
            if missing:
                raise ValueError(f"Cannot select unknown fields {missing} not in {self._columns}")
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
        table = self.client.table(self.table, dataset=self.dataset)
        if self._columns is not None:
            table = table[self._columns]
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
            if self._columns:
                return self._columns[0]
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

        tbl = self.client.load_data(
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
        if self._columns:
            s += f" | fields={self._columns}"
        if self._sort_by:
            s += f" | sort={self._sort_by}"
        return s
