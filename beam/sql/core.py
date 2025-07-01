import datetime as _dt
import json
import os as _os
import re as _re
import typing as _t
from argparse import Namespace

import ibis
import numpy as np
import pandas as _pd

from ..path import PureBeamPath, BeamPath, normalize_host
from ..utils import lazy_property as cached_property, recursive_elementwise
from ..type import check_type, Types
from ..utils import divide_chunks, retry
from ..base import Loc

from .queries import BeamIbisQuery, TimeFilter
from .groupby import Groupby


def _now():
    return _dt.datetime.now(tz=_dt.timezone.utc)


class LLMQueryResponse:
    """Simple response structure for LLM queries."""
    def __init__(self, query, description, table=None):
        self.query = query
        self.description = description
        self.table = table


class BeamIbis(PureBeamPath):
    """Path‑like Ibis wrapper with pandas+lazy query API similar to BeamElastic."""

    _TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S%z"  # match BeamElastic for parity
    date_format = '%Y-%m-%d %H:%M:%S'

    def __init__(
        self,
        *args,
        hostname=None, port=None, username=None, password=None, verify=False, 
        fragment=None, client=None, q=None, llm=None, timeout=None,
        columns: list[str] | None = None,
        backend: str | None = None,
        backend_kwargs: dict[str, _t.Any] | None = None,
        sort_by=None, max_actions=None, keep_alive=None, sleep=None,
        **kwargs,
    ) -> None:

        super().__init__(*args, hostname=hostname, port=port, username=username, password=password,
                            fragment=fragment, **kwargs)

        self.verify = verify
        self.timeout = float(timeout) if timeout is not None else None
        
        # Connection is established lazily to avoid needless auth in pickled objs
        self.backend = backend or 'sqlite'
        self.backend_kwargs = backend_kwargs or {}
        
        self._client = client
        self._database = None
        self._table_name = None
        self._table = None
        self._q = self.parse_query(q)

        # Field projection and ordering
        if columns is not None:
            columns = columns if isinstance(columns, list) else [columns]
        else:
            columns = []

        more_fields = self.fragment.split(',') if bool(self.fragment) else []
        columns = list(set(columns + more_fields))
        self.columns = columns if columns else None
        self.sort_by = sort_by

        # LLM integration
        self._llm = llm

        # Limits and performance settings
        if max_actions is None:
            max_actions = 10000
        self.max_actions = int(max_actions)

        if sleep is None:
            sleep = 0.1
        self.sleep = float(sleep)

        self.keep_alive = keep_alive or '1m'

        # Cache for values and metadata
        self._values = None
        self._metadata = None
        self._schema = None

        # Helper objects
        self.loc = Loc(self)

    @property
    def llm(self):
        from ..llm import beam_llm
        return beam_llm(self._llm)

    @staticmethod
    def parse_query(query) -> _t.Any:
        """Parse various query formats into Ibis expressions."""
        if query is None:
            return None
        if isinstance(query, str):
            # For simple string queries, we could parse them as raw SQL
            # For now, return as-is and handle in query_table
            return query
        return query

    def __repr__(self):
        parts = []
        if self.backend == "bigquery":
            parts.append("bigquery://")
            if self.project:
                parts.append(self.project)
            if self.database:
                parts.append("/" + self.database)
            if self.table_name:
                parts.append("/" + self.table_name)
        elif self.backend == "sqlite":
            parts.append("sqlite://")
            if self.database:
                parts.append(self.database)
            if self.table_name:
                parts.append("/" + self.table_name)
        else:
            parts.append(f"{self.backend}://")
            if self.hostname:
                parts.append(f"{self.hostname}")
                if self.port:
                    parts.append(f":{self.port}")
            if self.database:
                parts.append("/" + self.database)
            if self.table_name:
                parts.append("/" + self.table_name)

        s = "".join(parts)
        
        if self._q is not None:
            s += " | query: [...]"
        if self.columns:
            s += f" | fields: {self.columns}"
        if self.sort_by:
            s += f" | sort: {self.sort_by}"
        return s

    @property
    def q(self):
        if self.level == 'query':
            return self._q
        return None

    @property
    def project(self):
        if self.backend == "bigquery":
            return self.parts[0] if len(self.parts) > 0 else None
        return None

    @property
    def database(self):
        if self._database is None:
            if self.backend == "bigquery":
                self._database = self.parts[1] if len(self.parts) > 1 else None
            elif self.backend == "sqlite":
                if len(self.parts) > 0:
                    path = BeamPath(*self.parts[:-1]) if len(self.parts) > 1 else BeamPath(self.parts[0])
                    if path.suffix in [".db", ".sqlite"] or (len(self.parts) == 1 and not self.parts[0].endswith('/')):
                        self._database = str(path)
                        self._table_name = self.parts[-1] if len(self.parts) > 1 else None
                    else:
                        self._database = self.path if self.path != '/' else None
                        self._table_name = None
            elif self.backend in ['postgresql', 'postgres']:
                self._database = self.parts[0] if len(self.parts) > 0 else None
            else:
                self._database = self.parts[0] if len(self.parts) > 0 else None

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
        elif self.backend in ['postgresql', 'postgres']:
            return self.parts[1] if len(self.parts) > 1 else None
        else:
            return self.parts[1] if len(self.parts) > 1 else None

    def get_client(self):
        if self.backend == 'bigquery':
            kwargs = {
                'project_id': self.project,
                **self.backend_kwargs
            }
            if self.hostname:
                kwargs['host'] = self.hostname
            if self.port:
                kwargs['port'] = self.port
            if self.username:
                kwargs['user'] = self.username
            if self.password:
                kwargs['password'] = self.password
            return ibis.bigquery.connect(**kwargs)
            
        elif self.backend == 'sqlite':
            return ibis.sqlite.connect(database=self.database, **self.backend_kwargs)
            
        elif self.backend in ['postgresql', 'postgres']:
            kwargs = {
                'host': self.hostname or 'localhost',
                'port': self.port or 5432,
                'database': self.database,
                **self.backend_kwargs
            }
            if self.username:
                kwargs['user'] = self.username
            if self.password:
                kwargs['password'] = self.password
            return ibis.postgres.connect(**kwargs)
            
        elif self.backend == 'duckdb':
            return ibis.duckdb.connect(database=self.database, **self.backend_kwargs)
            
        elif self.backend == 'mysql':
            kwargs = {
                'host': self.hostname or 'localhost',
                'port': self.port or 3306,
                'database': self.database,
                **self.backend_kwargs
            }
            if self.username:
                kwargs['user'] = self.username
            if self.password:
                kwargs['password'] = self.password
            return ibis.mysql.connect(**kwargs)
            
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    @property
    def client(self):
        if self._client is None:
            self._client = self.get_client()
        return self._client

    @property
    def table(self):
        if self._table is None:
            if self.table_name is None:
                raise ValueError("No table specified")
            table_args = {}
            if self.backend == "bigquery":
                table_args = dict(database=self.database)
            self._table = self.client.table(self.table_name, **table_args)
        return self._table

    @property
    def query_table(self):
        """Get the current query expression or base table."""
        if self._q is not None:
            return self._q
        return self.table

    @property
    def level(self):
        if self.table_name is not None and self._q is not None:
            return "query"
        if self.table_name is not None:
            return "table"
        if self.database is not None:
            return "dataset"
        return "root"

    def gen(self, path, **kwargs):
        """Generate a new BeamIbis instance with updated parameters."""
        hostname = kwargs.pop('hostname', self.hostname)
        port = kwargs.pop('port', self.port)
        username = kwargs.pop('username', self.username)
        password = kwargs.pop('password', self.password)
        fragment = kwargs.pop('fragment', self.fragment)
        params = kwargs.pop('params', self.params)
        query = kwargs.pop('query', {})
        columns = kwargs.pop('columns', self.columns)
        llm = kwargs.pop('llm', self.llm)
        q = kwargs.pop('q', self._q)
        sort_by = kwargs.pop('sort_by', self.sort_by)
        backend = kwargs.pop('backend', self.backend)
        backend_kwargs = kwargs.pop('backend_kwargs', self.backend_kwargs)

        # must be after extracting all other kwargs
        query = {**query, **kwargs}
        PathType = type(self)
        return PathType(path, client=self._client, hostname=hostname, port=port, username=username, columns=columns,
                        password=password, fragment=fragment, params=params, llm=llm, q=q, sort_by=sort_by,
                        backend=backend, backend_kwargs=backend_kwargs, **query)

    # Field selection and projection
    def __getitem__(self, item: str | list[str]):
        """Select columns like df['col'] or df[['col1', 'col2']] or navigate paths."""
        if self.level == 'root':
            return self.gen(f"/{item}")
        else:
            if isinstance(item, str):
                item = [item]
            
            if self.columns is not None:
                if set(item) - set(self.columns):
                    raise ValueError(f"Cannot select fields {list(set(item) - set(self.columns))} not in {self.columns}")
            
            q = self.query_table.select(item)
            return self.gen(self.path, q=q, columns=item)

    def select(self, *columns):
        """Select specific columns."""
        q = self.query_table.select(list(columns))
        return self.gen(self.path, q=q, columns=list(columns))

    # Ordering
    def order_by(self, *fields):
        """Order by one or more fields."""
        q = self.query_table.order_by(list(fields))
        return self.gen(self.path, q=q, sort_by=list(fields))

    def sort_values(self, field, ascending=True):
        """Sort by a field (pandas-like interface)."""
        if ascending:
            return self.order_by(field)
        else:
            return self.order_by(ibis.desc(field))

    # Query composition (like BeamElastic)
    def __and__(self, other):
        """Combine queries with AND logic."""
        if isinstance(other, BeamIbis):
            other_pred = other._q
        else:
            other_pred = other
            
        current_q = self._q or self.table
        if other_pred is not None:
            if hasattr(current_q, 'filter'):
                q = current_q.filter(other_pred)
            else:
                q = other_pred
        else:
            q = current_q
            
        return self.gen(self.path, q=q)

    def __or__(self, other):
        """Combine queries with OR logic."""
        if isinstance(other, BeamIbis):
            other_pred = other._q
        else:
            other_pred = other
            
        current_q = self._q or self.table
        # For OR operations, we need to combine predicates at the filter level
        # This is more complex in Ibis and may require restructuring the query
        if other_pred is not None and hasattr(current_q, 'filter'):
            # For now, we'll use a simple approach
            q = current_q.filter(other_pred)
        else:
            q = current_q
            
        return self.gen(self.path, q=q)

    # Filtering methods
    def parse_column(self, field: str | None = None):
        """Parse column name, using default if needed."""
        if field is None:
            if self.columns and len(self.columns) == 1:
                return self.columns[0]
            else:
                raise ValueError("Must specify field name or have exactly one column selected")
        return field

    def filter_term(self, value, field: str | None = None):
        """Filter for exact term match."""
        col = self.parse_column(field)
        return self.query_table[col] == value

    def filter_terms(self, values, field: str | None = None):
        """Filter for multiple term matches (IN clause)."""
        col = self.parse_column(field)
        return self.query_table[col].isin(list(values))

    def filter_gte(self, value, field: str | None = None):
        """Filter for values >= threshold."""
        col = self.parse_column(field)
        return self.query_table[col] >= value

    def filter_gt(self, value, field: str | None = None):
        """Filter for values > threshold."""
        col = self.parse_column(field)
        return self.query_table[col] > value

    def filter_lte(self, value, field: str | None = None):
        """Filter for values <= threshold."""
        col = self.parse_column(field)
        return self.query_table[col] <= value

    def filter_lt(self, value, field: str | None = None):
        """Filter for values < threshold."""
        col = self.parse_column(field)
        return self.query_table[col] < value

    def filter_time_range(
        self,
        *,
        field: str | None = None,
        start: _dt.datetime | str | None = None,
        end: _dt.datetime | str | None = None,
        period: _dt.timedelta | str | None = None,
    ):
        """Filter for time range similar to BeamElastic TimeFilter."""
        return TimeFilter(
            backend=self.backend,
            table=self.query_table,
            field=field,
            start=start,
            end=end,
            period=period
        )

    # "with_filter" methods that return new instances
    def _with_filter(self, predicate):
        """Apply a filter predicate and return new instance."""
        if hasattr(predicate, 'to_expr'):
            # Handle custom filter objects like TimeFilter
            q = predicate.to_expr()
        else:
            # Handle Ibis expressions
            q = self.query_table.filter(predicate)
        return self.gen(self.path, q=q)

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

    # Comparison operators (like BeamElastic)
    def __eq__(self, other):
        return self.with_filter_term(other)

    def __ge__(self, other):
        return self.with_filter_gte(other)

    def __gt__(self, other):
        return self.with_filter_gt(other)

    def __le__(self, other):
        return self.with_filter_lte(other)

    def __lt__(self, other):
        return self.with_filter_lt(other)

    # Directory operations
    def iterdir(self, wildcard=None, hidden=False):
        """Iterate over contents (tables, etc.)."""
        if self.level == "root":
            # List all databases/projects
            if self.backend == "bigquery":
                # For BigQuery, we'd need to list projects, but this requires special permissions
                raise NotImplementedError("BigQuery project listing not implemented")
            elif self.backend == "sqlite":
                # For SQLite, list database files in current directory
                import glob
                pattern = wildcard or "*.db"
                for db_file in glob.glob(pattern):
                    yield self.gen(f"/{db_file}")
            else:
                # For other backends, we'd need backend-specific logic
                raise NotImplementedError(f"Database listing not implemented for {self.backend}")
                 
        elif self.level == "dataset":
            # List tables in database
            try:
                tables = self.client.list_tables(database=self.database)
                for table in tables:
                    if not hidden and table.startswith('_'):
                        continue
                    if wildcard and not table.match(wildcard):
                        continue
                    yield self.gen(f"{self.path}/{table}")
            except Exception:
                # Some backends don't support database parameter
                tables = self.client.list_tables()
                for table in tables:
                    if not hidden and table.startswith('_'):
                        continue
                    if wildcard and not table.match(wildcard):
                        continue
                    yield self.gen(f"{self.path}/{table}")
        else:
            raise ValueError("iterdir not supported at table/query level")

    def is_file(self):
        """Check if this is a 'file' (table in database context)."""
        return self.level == "table"

    def is_dir(self):
        """Check if this is a 'directory' (database/root in database context)."""
        return self.level in ["root", "dataset"]

    def exists(self):
        """Check if the path exists."""
        if self.level == "root":
            try:
                return bool(self.client)
            except Exception:
                return False
        elif self.level == "dataset":
            try:
                self.client.list_tables(database=self.database)
                return True
            except Exception:
                return False
        elif self.level == "table":
            try:
                self.table
                return True
            except Exception:
                return False
        else:
            # Query level - check if query returns any results
            try:
                return self.count() > 0
            except Exception:
                return False

    def count(self):
        """Count rows in table/query."""
        if self.level in ['query', 'table']:
            return int(self.query_table.count().execute())
        elif self.level == 'dataset':
            # Count tables in dataset
            return len(list(self.iterdir()))
        else:
            return 0

    def __len__(self):
        return self.count()

    # Schema operations
    @cached_property
    def schema(self):
        """Get table schema."""
        if self.level in ['table', 'query']:
            return dict(self.query_table.schema())
        return {}

    # Aggregation methods
    def unique(self, field_name=None, size=None):
        """Get unique values in a field."""
        field_name = self.parse_column(field_name)
        q = self.query_table.select(field_name).distinct()
        if size is not None:
            q = q.limit(size)
        result = q.execute()
        return result[field_name].tolist()

    def nunique(self, field_name=None):
        """Count unique values in a field."""
        field_name = self.parse_column(field_name)
        return int(self.query_table[field_name].nunique().execute())

    def value_counts(self, field_name=None, sort=True, normalize=False):
        """Get value counts for a field (pandas-like)."""
        field_name = self.parse_column(field_name)
        q = (self.query_table
             .group_by(field_name)
             .aggregate(count=ibis.literal(1).count())
             .select(field_name, 'count'))
        
        if sort:
            q = q.order_by(ibis.desc('count'))
            
        df = q.execute()
        series = df.set_index(field_name)['count']
        
        if normalize:
            series = series / series.sum()
            
        return series

    def agg(self, agg_func, field_name=None, **kwargs):
        """Apply aggregation function."""
        field_name = self.parse_column(field_name)
        col = self.query_table[field_name]
        
        if agg_func == 'sum':
            result = col.sum()
        elif agg_func == 'mean' or agg_func == 'avg':
            result = col.mean()
        elif agg_func == 'min':
            result = col.min()
        elif agg_func == 'max':
            result = col.max()
        elif agg_func == 'std':
            result = col.std()
        elif agg_func == 'var':
            result = col.var()
        elif agg_func == 'count':
            result = col.count()
        else:
            raise ValueError(f"Unsupported aggregation function: {agg_func}")
            
        return result.execute()

    def sum(self, field_name=None):
        return self.agg('sum', field_name)

    def mean(self, field_name=None):
        return self.agg('mean', field_name)

    def min(self, field_name=None):
        return self.agg('min', field_name)

    def max(self, field_name=None):
        return self.agg('max', field_name)

    def std(self, field_name=None):
        return self.agg('std', field_name)

    def var(self, field_name=None):
        return self.agg('var', field_name)

    # GroupBy operations
    def groupby(self, field_names, size=None, **kwargs):
        """Create a GroupBy object for aggregations."""
        return Groupby(self, field_names, size=size, **kwargs)

    # Data retrieval methods
    def as_df(self, limit: int | None = None, add_ids=False, add_metadata=False):
        """Execute query and return pandas DataFrame."""
        q = self.query_table
        if limit is not None:
            q = q.limit(limit)
        return q.execute()

    def as_dict(self, limit: int | None = None, add_metadata=False):
        """Execute query and return list of dictionaries."""
        df = self.as_df(limit=limit)
        records = df.to_dict(orient="records")
        if add_metadata:
            # For now, return empty metadata
            return records, [{}] * len(records)
        return records

    def as_pl(self, limit: int | None = None):
        """Execute query and return Polars DataFrame."""
        import polars as pl
        return pl.from_pandas(self.as_df(limit=limit))

    def as_cudf(self, limit: int | None = None):
        """Execute query and return cuDF DataFrame."""
        import cudf
        return cudf.from_pandas(self.as_df(limit=limit))

    def head(self, n: int = 5):
        """Get first n rows."""
        return self.as_df(limit=n)

    def tail(self, n: int = 5):
        """Get last n rows (requires ordering)."""
        # This is tricky without knowing the natural order
        # For now, we'll just reverse any existing order and take head
        q = self.query_table
        if self.sort_by:
            # Reverse the sort order
            reverse_sorts = []
            for sort_field in self.sort_by:
                if isinstance(sort_field, str):
                    reverse_sorts.append(ibis.desc(sort_field))
                else:
                    reverse_sorts.append(sort_field)  # Assume it's already ordered
            q = q.order_by(reverse_sorts).limit(n)
        else:
            q = q.limit(n)
        return q.execute()

    # Data writing methods
    def write(self, data, if_exists="append", **kwargs):
        """Write data to table."""
        if self.level != "table":
            raise ValueError("Write only supported at table level")
            
        if isinstance(data, _pd.DataFrame):
            # Use pandas to_sql if available, otherwise convert to records
            try:
                if hasattr(self.client, 'raw_sql'):
                    # For backends that support raw SQL
                    data.to_sql(self.table_name, self.client, if_exists=if_exists, index=False, **kwargs)
                else:
                    # For Ibis backends, we might need to create table first then insert
                    if if_exists == "replace" or not self.exists():
                        # Create table from DataFrame
                        table_expr = ibis.memtable(data, name=self.table_name)
                        self.client.create_table(self.table_name, table_expr)
                    else:
                        # Insert data
                        records = data.to_dict(orient='records')
                        # This is backend-specific and might not work for all backends
                        raise NotImplementedError("Insert operation not implemented for this backend")
            except Exception as e:
                raise ValueError(f"Failed to write data: {e}")
        else:
            raise ValueError("Data must be a pandas DataFrame")

    # PureBeamPath API compatibility methods
    def read(self, as_df=False, as_dict=False, as_iter=True, limit=None, add_ids=False, add_score=False,
             add_index_name=False, add_metadata=False, **kwargs):
        """
        Read data from table/query with multiple output formats.
        Similar to BeamElastic's read method.
        """
        if self.level == 'root':
            # Return list of databases/datasets
            return list(self.iterdir())
        elif self.level == 'dataset':
            # Return list of tables
            return list(self.iterdir())
        
        if as_df:
            return self.as_df(limit=limit, add_ids=add_ids, add_metadata=add_metadata)
        
        if as_dict:
            return self.as_dict(limit=limit, add_metadata=add_metadata)
        
        if as_iter:
            # Return iterator over records
            df = self.as_df(limit=limit)
            for _, row in df.iterrows():
                yield row.to_dict()

    def items(self):
        """Iterate over key-value pairs (similar to BeamElastic)."""
        if self.level == 'root':
            # Return (name, BeamIbis) pairs for each database
            for db in self.iterdir():
                yield db.name, db
        elif self.level == 'dataset':
            # Return (name, BeamIbis) pairs for each table
            for table in self.iterdir():
                yield table.name, table
        else:
            # For table/query level, iterate over rows with index as key
            df = self.as_df()
            for idx, row in df.iterrows():
                yield idx, row.to_dict()

    @property
    def values(self):
        """
        Get all values (similar to BeamElastic).
        Returns different things depending on level.
        """
        if self.level == 'root':
            return [str(db) for db in self.iterdir()]
        elif self.level == 'dataset':
            return [str(table) for table in self.iterdir()]
        else:
            return self._get_all_values()

    def _get_all_values(self, **kwargs):
        """Get all values from table/query."""
        if self._values is None:
            self._values = self.as_dict(**kwargs)
        return self._values

    # File operations (path-like interface)
    def mkdir(self, parents=True, exist_ok=True):
        """Create table/database (conceptually like creating a directory)."""
        if self.level == "table":
            if not exist_ok and self.exists():
                raise FileExistsError(f"Table {self.table_name} already exists")
            # Creating an empty table requires a schema, which we don't have here
            # This would typically be done when writing data
            pass
        elif self.level == "dataset":
            # For some backends, databases are created automatically
            # For others, this might require special permissions
            pass
        else:
            raise ValueError("mkdir only supported at table/dataset level")

    def rmdir(self):
        """Remove directory (database)."""
        if self.level == "dataset":
            # Drop all tables in database (dangerous!)
            for table in self.iterdir():
                table.delete()
        else:
            raise ValueError("rmdir only supported at dataset level")

    def rmtree(self, ignore=None, include=None):
        """Remove tree recursively."""
        if self.level == "table":
            self.delete()
        elif self.level == "dataset":
            # Delete all tables
            for table in self.iterdir():
                ext = table.suffix  # In database context, this might be table type
                if ignore is not None:
                    if isinstance(ignore, str):
                        ignore = [ignore]
                    if ext in ignore:
                        continue
                if include is not None:
                    if isinstance(include, str):
                        include = [include]
                    if ext not in include:
                        continue
                table.delete()
            self.rmdir()
        else:
            raise ValueError("rmtree not supported at this level")

    def delete(self):
        """Delete table/database."""
        if self.level == "table":
            self.client.drop_table(self.table_name)
        elif self.level == "dataset":
            # Drop all tables in database (dangerous!)
            for table in self.iterdir():
                table.delete()
        else:
            raise ValueError("Delete not supported at this level")

    def unlink(self, **kwargs):
        """Alias for delete (path-like interface)."""
        return self.delete()

    def touch(self, mode=0o666, exist_ok=True):
        """Create empty table (like touching a file)."""
        if self.level == "table":
            if not exist_ok and self.exists():
                raise FileExistsError(f"Table {self.table_name} already exists")
            # Create empty table with minimal schema
            import pandas as pd
            empty_df = pd.DataFrame({'_placeholder': [1]})  # Minimal schema
            self.write(empty_df, if_exists='replace')
            # Remove the placeholder data
            try:
                # Try to delete all rows
                if hasattr(self.client, 'raw_sql'):
                    self.client.raw_sql(f"DELETE FROM {self.table_name}")
            except:
                pass  # Some backends might not support this
        else:
            raise ValueError("touch only supported at table level")

    def rename(self, target):
        """Rename table."""
        if self.level != "table":
            raise ValueError("rename only supported at table level")
        
        if isinstance(target, str):
            target_name = target
        else:
            target_name = target.table_name
        
        # This is backend-specific and might not work for all backends
        try:
            if hasattr(self.client, 'raw_sql'):
                self.client.raw_sql(f"ALTER TABLE {self.table_name} RENAME TO {target_name}")
                return self.gen(f"{self.path.parent}/{target_name}")
            else:
                raise NotImplementedError("Rename not supported for this backend")
        except Exception as e:
            raise ValueError(f"Failed to rename table: {e}")

    def replace(self, target):
        """Replace table (rename with overwrite)."""
        if isinstance(target, BeamIbis) and target.exists():
            target.delete()
        return self.rename(target)

    def copy(self, dst, **kwargs):
        """Copy data to another table."""
        if isinstance(dst, str):
            dst = self.gen(dst)
        
        if self.level in ['table', 'query']:
            data = self.as_df()
            dst.write(data, **kwargs)
        elif self.level == 'dataset':
            # Copy all tables
            dst.mkdir(exist_ok=True)
            for table in self.iterdir():
                table.copy(dst.joinpath(table.name), **kwargs)
        else:
            raise ValueError("Copy not supported at this level")

    def walk(self):
        """Walk directory tree (similar to os.walk)."""
        if self.level == "root":
            # Walk through databases
            for db in self.iterdir():
                if db.is_dir():
                    yield from db.walk()
        elif self.level == "dataset":
            dirs = []
            files = []
            
            for item in self.iterdir():
                if item.is_dir():
                    dirs.append(item.name)
                else:
                    files.append(item.name)
            
            yield self, dirs, files
            
            for dir_name in dirs:
                yield from self.joinpath(dir_name).walk()

    # LLM Integration (similar to BeamElastic)
    def ask(self, question, llm=None, execute=False, answer=True, **kwargs):
        """Ask natural language questions about the data using LLM."""
        if llm is None:
            llm = self.llm

        if llm is None:
            raise ValueError("LLM resource not set")

        schema_info = str(self.schema) if self.schema else "Schema not available"
        table_info = f"Table: {self.table_name}\n" if self.table_name else ""
        
        prompt = (f"You are an agent that interacts with a SQL database using Ibis. "
                  f"You are required to answer the users' questions based on the data in the database. "
                  f"You can generate SQL queries to retrieve the relevant data.\n\n"
                  f"The dataset schema is:\n"
                  f"{table_info}"
                  f"{schema_info}\n\n"
                  f"User's question: {question}\n\n"
                  f"Please provide a SQL query to answer this question.")

        if not hasattr(llm, 'chat'):
            raise ValueError("LLM must have a chat method")

        llm.reset_chat()
        response = llm.chat(prompt, **kwargs)
        
        # Extract SQL from response (this is simplified)
        sql_query = response.text if hasattr(response, 'text') else str(response)
        
        query_result = None
        text_answer = None
        
        if execute:
            try:
                # Execute the SQL query
                expr = self.client.sql(sql_query)
                df = expr.execute()
                query_result = df
                
                if answer and not df.empty:
                    info = f"Query returned {len(df)} rows with columns: {list(df.columns)}\n"
                    info += df.head().to_string()
                    
                    answer_prompt = (f"Based on the data retrieved from the database:\n"
                                   f"{info}\n\n"
                                   f"Please provide a text answer to the user's question: {question}")
                    
                    text_answer = llm.chat(answer_prompt, **kwargs).text
            except Exception as e:
                query_result = f"Error executing query: {e}"
        
        return Namespace(
            query=sql_query,
            description="Generated SQL query",
            df=query_result,
            text_answer=text_answer
        )

    # Utility methods
    def sql(self):
        """Get the SQL representation of the current query."""
        if self.level in ['table', 'query']:
            return ibis.to_sql(self.query_table)
        raise ValueError("SQL only available for table/query level")

    def info(self):
        """Get information about the table/query."""
        if self.level in ['table', 'query']:
            schema_info = self.schema
            count = self.count()
            return {
                'table': self.table_name,
                'row_count': count,
                'columns': list(schema_info.keys()) if schema_info else [],
                'schema': schema_info
            }
        return {}

    # Additional utility methods
    def ping(self):
        """Test connection (similar to BeamElastic)."""
        try:
            return bool(self.client)
        except Exception:
            return False

    def not_empty(self, filter_pattern=None):
        """Check if not empty."""
        if self.is_dir():
            for item in self.iterdir():
                if item.not_empty():
                    return True
                if item.is_file():
                    if filter_pattern is not None:
                        if not re.match(filter_pattern, item.name):
                            return True
                    else:
                        return True
        elif self.is_file():
            return self.count() > 0
        return False
