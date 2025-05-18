
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
        q=None, llm=None,
        columns: list[str] | None = None,
        backend: str | None = None,
        backend_kwargs: dict[str, _t.Any] | None = None, **kwargs,
    ) -> None:

        super().__init__(*args, hostname=hostname, port=port, username=username, password=password,
                            fragment=fragment, client=client, **kwargs)

        self.columns = columns  # projection

        # Connection is established lazily to avoid needless auth in pickled objs
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.verify = verify

        self._database = None
        self._table_name = None
        self._table = None

        self.q = q

    @property
    def table(self):
        if self._table is None:
            table_args = {}
            if self.backend == "bigquery":
                table_args = dict(
                    database=self.database,
                )
            self._table = self.client.table(self.table_name, **table_args)
        return self._table

    @property
    def project(self):
        if self.backend == "bigquery":
            return self.parts[0] if len(self.parts) > 0 else None
        # assume sqlite
        raise ValueError(f"Project not supported for {self.backend} backend")

    @property
    def database(self):
        if self._database is None:
            if self.backend == "bigquery":
                d = self.parts[1] if len(self.parts) > 1 else None
            elif self.backend == "sqlite":
                path = BeamPath(*self.parts[:-1])
                if path.is_file() or any(path.parts[-1].endswith(ext) for ext in [".db", ".sqlite"]):
                    d = str(path)
                    self._table_name = self.parts[-1]
                else:
                    d = self.path
                    self._table_name = None
            else:
                raise ValueError(f"Database not supported for {self.backend} backend")

            self._database = d
        return self._database

    @property
    def table_name(self):

        if self._table_name is not None:
            return self._table_name

        if self.backend == "bigquery":
            return self.parts[2] if len(self.parts) > 2 else None
        elif self.backend == "sqlite":
            _ = self.database  # force database resolution
            return self._table_name

        raise ValueError(f"Table not supported for {self.backend} backend")

    def get_client(self):
        if self.backend == 'bigquery':
            c = ibis.bigquery.connect(host=self.hostname, port=self.port,
                username=self.username, password=self.password, verify=self.verify,
                project_id=self.project,
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

    def gen(self, path, **kwargs):
        hostname = kwargs.pop('hostname', self.hostname)
        port = kwargs.pop('port', self.port)
        username = kwargs.pop('username', self.username)
        password = kwargs.pop('password', self.password)
        fragment = kwargs.pop('fragment', self.fragment)
        params = kwargs.pop('params', self.params)
        query = kwargs.pop('query', {})
        columns = kwargs.pop('columns', self.columns)
        llm = kwargs.pop('llm', self.llm)
        q = kwargs.pop('q', self.q)

        # must be after extracting all other kwargs
        query = {**query, **kwargs}
        PathType = type(self)
        return PathType(path, client=self.client, hostname=hostname, port=port, username=username, columns=columns,
                        password=password, fragment=fragment, params=params, llm=llm, q=q, **query)

    @property
    def query_table(self):
        if self.q is not None:
            return self.q
        return self.table

    # Allow dict‑like field selection: client["col1"] -> restrict projection
    def __getitem__(self, item: str | list[str]):
        q = self.query_table[item]
        return self.gen(self.path, q=q)

    def order_by(self, field: str | list[str]):
        q = self.query_table.order_by(field)
        return self.gen(self.path, q=q)

    def filter(self, field: str | list[str], value: _t.Any):
        q = self.query_table.filter(field, value)
        return self.gen(self.path, q=q)

    def iterdir(self):
        """Iterate over the directory contents."""
        if self.level == "root":
            return [self.gen(self.path, q=self.client.table(t)) for t in self.client.list_tables()]
        elif self.level == "dataset":
            return [self.gen(self.path, q=self.client.table(t)) for t in self.client.list_tables(self.database)]
        raise ValueError("Iterdir only supported at root or dataset level")

    def mkdir(self):
        """Create a new directory."""
        if self.level == "root":
            raise ValueError("Cannot create root directory")
        elif self.level == "dataset":
            raise ValueError("Cannot create dataset directory")
        elif self.level == "table":
            # create table
            if self.backend == "bigquery":
                self.client.create_table(self.table_name, database=self.database)
            elif self.backend == "sqlite":
                self.client.create_table(self.table_name, database=self.database)
            else:
                raise ValueError(f"Create table not supported for {self.backend} backend")
        else:
            raise ValueError("Cannot create table in query mode")

    @property
    def level(self):
        if self.table_name is not None and self.q is not None:
            return "query"
        if self.table_name is not None:
            return "table"
        if self.database is not None:
            return "dataset"
        return "root"

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
        return self.query_table.execute(limit=limit)

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
    def write(self, data, **kwargs):
        """Write a DataFrame to the table."""
        if self.level == "table":
            if isinstance(data, _pd.DataFrame):
                data.to_sql(self.table_name, self.client, if_exists="append", index=False)
            else:
                raise ValueError("Data must be a pandas DataFrame")
        else:
            raise ValueError("Write only supported at table level")

    def count(self):
        """Count rows in the table."""
        if self.level in ['query', 'table']:
            return self.query_table.count()
        raise ValueError("Count only supported at table or query level")

    def __repr__(self):
        parts = ["bigquery://"]
        if self.project:
            parts.append(self.project)
        if self.database:
            parts.append("/" + self.database)
        if self.table_name:
            parts.append("/" + self.table_name)
        s = "".join(parts)
        if self.expr is not None and self.level == "query":
            s += " | expr=[…]"
        if self.columns:
            s += f" | fields={self.columns}"
        if self._sort_by:
            s += f" | sort={self._sort_by}"
        return s
