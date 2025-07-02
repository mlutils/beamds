import datetime as _dt
import re
import typing as _t
from argparse import Namespace
import ibis

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
        
        # Flatten columns if it contains nested lists
        flattened_columns = []
        for item in columns:
            if isinstance(item, list):
                flattened_columns.extend(item)
            else:
                flattened_columns.append(item)
        
        # Combine and deduplicate
        all_columns = flattened_columns + more_fields
        columns = list(dict.fromkeys(all_columns))  # Preserves order while removing duplicates
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
            return self.parts[1] if len(self.parts) > 0 else None
        return None

    @property
    def database(self):
        if self._database is None:
            if self.backend == "bigquery":
                self._database = self.parts[2] if len(self.parts) > 2 else None
            elif self.backend == "sqlite":
                # For SQLite: determine if we have a database file or database + table
                if len(self.parts) == 0:
                    self._database = None
                else:
                    # Check if the full path or last part looks like a database file
                    full_path = '/'.join(self.parts)
                    last_part = self.parts[-1]
                    
                    # If the full path ends with .db/.sqlite, it's all database
                    if full_path.endswith(('.db', '.sqlite', '.sqlite3')):
                        self._database = full_path
                    # If last part ends with .db/.sqlite, everything up to and including it is database
                    elif last_part.endswith(('.db', '.sqlite', '.sqlite3')):
                        self._database = full_path
                    # Otherwise, assume last part is table name
                    else:
                        if len(self.parts) == 1:
                            # Single part without extension - treat as database
                            self._database = self.parts[0]
                        else:
                            # Multiple parts - last is table, rest is database
                            self._database = '/'.join(self.parts[:-1])
            elif self.backend in ['postgresql', 'postgres']:
                self._database = self.parts[1] if len(self.parts) > 1 else None
            else:
                self._database = self.parts[1] if len(self.parts) > 1 else None

        return self._database

    @property
    def table_name(self):
        if self._table_name is not None:
            return self._table_name

        if self.backend == "bigquery":
            return self.parts[3] if len(self.parts) > 3 else None
        elif self.backend == "sqlite":
            # For SQLite: table is only the last part if it's not a database file
            if len(self.parts) <= 1:
                return None
            
            full_path = '/'.join(self.parts)
            last_part = self.parts[-1]
            
            # If the full path or last part looks like a database file, no table
            if (full_path.endswith(('.db', '.sqlite', '.sqlite3')) or 
                last_part.endswith(('.db', '.sqlite', '.sqlite3'))):
                return None
            else:
                # Last part is table name
                return last_part
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
            # For SQLite, database comes from path parsing only, not backend_kwargs
            # Remove any 'database' key from backend_kwargs to avoid conflicts
            sqlite_kwargs = {k: v for k, v in self.backend_kwargs.items() if k != 'database'}
            return ibis.sqlite.connect(database=self.database, **sqlite_kwargs)
            
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
        llm = kwargs.pop('llm', self._llm)
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
        # For now, disable composition of BeamIbis instances and suggest alternative API
        if isinstance(other, BeamIbis):
            raise NotImplementedError(
                "Query composition with & operator is not yet fully implemented. "
                "Please use chained method calls instead:\n"
                "Instead of: db.with_filter_term('a', 'field1') & db.with_filter_gte(10, 'field2')\n"
                "Use: db.with_filter_term('a', 'field1').with_filter_gte(10, 'field2')"
            )
        else:
            # Handle direct predicate (Ibis expression)
            current_q = self._q if self._q is not None else self.table
            if other is not None:
                q = current_q.filter(other)
            else:
                q = current_q
            return self.gen(self.path, q=q)

    def __or__(self, other):
        """Combine queries with OR logic."""
        if isinstance(other, BeamIbis):
            other_pred = other._q
        else:
            other_pred = other
            
        current_q = self._q if self._q is not None else self.table
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
             .aggregate(count=ibis._.count())
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

    # Data writing methods - Enhanced version using only BeamIbis API
    def write_table(self, data, table_name=None, if_exists="append", **kwargs):
        """
        Write data to create a new table or append to existing table.
        
        Args:
            data: pandas.DataFrame, pyarrow.Table, or Ibis expression to write
            table_name: Name of table to create/write to (defaults to self.table_name)
            if_exists: "replace", "append", or "fail" (default: "append")
            **kwargs: Additional backend-specific options
            
        Returns:
            BeamIbis: New instance pointing to the written table
        """
        if self.level not in ["table", "dataset"]:
            raise ValueError("write_table only supported at table or dataset level")
            
        target_table_name = table_name or self.table_name
        if not target_table_name:
            raise ValueError("Must specify table_name or be at table level")
            
        # Handle different data types
        if hasattr(data, 'to_pyarrow'):  # Ibis expression
            arrow_data = data.to_pyarrow()
        elif hasattr(data, 'to_pandas'):  # Other dataframe types
            arrow_data = data.to_pandas()
        else:
            arrow_data = data
            
        # Check if table exists
        target_path = f"{self.path.parent}/{target_table_name}" if self.level == "table" else f"{self.path}/{target_table_name}"
        target_beam = self.gen(target_path)
        table_exists = target_beam.exists()
        
        if table_exists and if_exists == "fail":
            raise ValueError(f"Table {target_table_name} already exists")
        
        if not table_exists or if_exists == "replace":
            # Create new table
            if hasattr(data, 'schema'):  # Ibis expression
                schema = data.schema()
            else:
                # Infer schema from data
                import pandas as pd
                if isinstance(arrow_data, pd.DataFrame):
                    # Let Ibis handle schema inference by creating a memtable first
                    schema = None  # Let client.create_table infer the schema
                else:
                    schema = None
                    
            if if_exists == "replace" and table_exists:
                target_beam.delete()
                
            # Prepare create_table arguments based on backend
            create_kwargs = {}
            if self.backend == "bigquery" and self.database:
                create_kwargs['database'] = self.database
            create_kwargs.update(kwargs)
            
            self.client.create_table(
                target_table_name,
                obj=arrow_data,
                schema=schema,
                **create_kwargs
            )
        else:
            # Append to existing table
            # Prepare insert arguments based on backend
            insert_kwargs = {}
            if self.backend == "bigquery" and self.database:
                insert_kwargs['database'] = self.database
            insert_kwargs.update(kwargs)
            
            self.client.insert(
                target_table_name,
                obj=arrow_data,
                **insert_kwargs
            )
            
        return self.gen(target_path)
    
    def append_batch(self, data, **kwargs):
        """
        Append batch data to this table.
        
        Args:
            data: pandas.DataFrame, pyarrow.Table, or Ibis expression
            **kwargs: Additional backend-specific options
            
        Returns:
            BeamIbis: Self for method chaining
        """
        if self.level != "table":
            raise ValueError("append_batch only supported at table level")
            
        self.write_table(data, if_exists="append", **kwargs)
        return self
        
    def append_row(self, row_data, **kwargs):
        """
        Append a single row to this table.
        
        Args:
            row_data: dict with column names as keys
            **kwargs: Additional backend-specific options
            
        Returns:
            BeamIbis: Self for method chaining
        """
        if self.level != "table":
            raise ValueError("append_row only supported at table level")
            
        import pandas as pd
        
        # Convert single row to DataFrame
        df = pd.DataFrame([row_data])
        self.write_table(df, if_exists="append", **kwargs)
        return self
        
    def create_table_from_data(self, data, table_name, **kwargs):
        """
        Create a new table from data.
        
        Args:
            data: pandas.DataFrame, pyarrow.Table, or Ibis expression
            table_name: Name of the new table
            **kwargs: Additional backend-specific options
            
        Returns:
            BeamIbis: New instance pointing to the created table
        """
        return self.write_table(data, table_name=table_name, if_exists="replace", **kwargs)
    
    def create_table_from_schema(self, schema, table_name, **kwargs):
        """
        Create an empty table from a schema definition.
        
        Args:
            schema: BeamIbisSchema class, Ibis schema, dict, or legacy schema instance
            table_name: Name of the new table
            **kwargs: Additional backend-specific options
            
        Returns:
            BeamIbis: New instance pointing to the created table
        """
        if self.level not in ["dataset", "root"]:
            raise ValueError("create_table_from_schema only supported at dataset/root level")
            
        # Handle different schema types
        if hasattr(schema, 'to_ibis_schema') and callable(schema.to_ibis_schema):
            # New schema class or legacy schema instance
            ibis_schema = schema.to_ibis_schema()
        elif isinstance(schema, dict):
            ibis_schema = ibis.schema(schema)
        else:
            # Assume it's already an Ibis schema
            ibis_schema = schema
            
        target_path = f"{self.path}/{table_name}"
        
        # Prepare create_table arguments based on backend
        create_kwargs = {}
        if self.backend == "bigquery" and self.database:
            create_kwargs['database'] = self.database
        create_kwargs.update(kwargs)
        
        self.client.create_table(
            table_name,
            schema=ibis_schema,
            **create_kwargs
        )
        
        return self.gen(target_path)

    # Enhanced write method (main API)
    def write(self, data, schema=None, if_exists="append", **kwargs):
        """
        Smart write method that routes to appropriate operations based on context.
        
        This is the main write API that automatically determines the best write strategy:
        - Routes to create_table_from_data/schema for new tables
        - Routes to append_batch for multiple rows
        - Routes to append_row for single rows
        
        Args:
            data: Data to write (DataFrame, list of dicts, single dict, etc.)
            schema: Optional BeamIbisSchema class or instance to enforce
            if_exists: "append", "replace", or "fail" (default: "append")
            **kwargs: Additional backend-specific options
            
        Returns:
            BeamIbis: Instance pointing to the written table
        """
        if self.level not in ["table", "dataset"]:
            raise ValueError("write only supported at table or dataset level")
        
        # Check data type using beam type checking
        data_type = check_type(data)
        
        # Determine target table name
        target_table_name = self.table_name if self.level == "table" else None
        if not target_table_name:
            raise ValueError("Must specify table name or be at table level")
        
        # Check if table exists
        table_exists = self.exists() if self.level == "table" else False
        
        # Handle different data input types
        processed_data = self._prepare_data_for_write(data, data_type)
        is_single_row = self._is_single_row_data(processed_data, data_type)
        
        # Route to appropriate method based on context
        if not table_exists:
            # Table doesn't exist - create it
            if schema is not None:
                # Create from schema first, then insert data
                if hasattr(schema, 'to_ibis_schema') and callable(schema.to_ibis_schema):
                    # Schema class or instance - need to create at dataset level
                    if self.level == "table":
                        # Navigate to parent dataset level for creation
                        dataset_level = self.gen(str(self.path.parent))
                        result = dataset_level._create_table_from_schema_then_insert(schema, target_table_name, processed_data, **kwargs)
                    else:
                        result = self._create_table_from_schema_then_insert(schema, target_table_name, processed_data, **kwargs)
                else:
                    # Dictionary or Ibis schema
                    if self.level == "table":
                        dataset_level = self.gen(str(self.path.parent))
                        result = dataset_level.create_table_from_schema(schema, target_table_name, **kwargs)
                        if processed_data is not None:
                            result.write(processed_data, if_exists="append", **kwargs)
                    else:
                        result = self.create_table_from_schema(schema, target_table_name, **kwargs)
                        if processed_data is not None:
                            result.write(processed_data, if_exists="append", **kwargs)
            else:
                # Create from data
                if self.level == "table":
                    dataset_level = self.gen(str(self.path.parent))
                    result = dataset_level.create_table_from_data(processed_data, target_table_name, **kwargs)
                else:
                    result = self.create_table_from_data(processed_data, target_table_name, **kwargs)
        else:
            # Table exists - append or replace
            if if_exists == "fail":
                raise ValueError(f"Table {target_table_name} already exists")
            elif if_exists == "replace":
                # Replace entire table
                result = self.write_table(processed_data, if_exists="replace", **kwargs)
            else:
                # Append data
                if is_single_row:
                    result = self.append_row(processed_data, **kwargs)
                else:
                    result = self.append_batch(processed_data, **kwargs)
        
        return result
    
    def _prepare_data_for_write(self, data, data_type):
        """Prepare data for writing based on its type."""
        
        # Handle different input types
        if data_type.is_dataframe:
            # pandas, polars, cudf DataFrames
            return data
        elif data_type.minor == 'list':
            # List of dictionaries
            if len(data) == 0:
                return None  # Empty list - no data to insert
            # Convert to DataFrame for consistency
            import pandas as pd
            return pd.DataFrame(data)
        elif data_type.minor == 'dict':
            # Single dictionary (single row)
            return data
        elif hasattr(data, 'to_pandas'):
            # Other dataframe-like objects
            return data.to_pandas()
        else:
            # Try to handle as-is
            return data
    
    def _is_single_row_data(self, data, data_type):
        """Determine if data represents a single row."""
        
        if data is None:
            return False
        
        if data_type.minor == 'dict':
            return True
        elif data_type.is_dataframe and hasattr(data, '__len__'):
            return len(data) == 1
        elif data_type.minor == 'list':
            if hasattr(data, '__len__'):
                return len(data) == 1
            else:
                return False
        return False
    
    def _create_table_from_schema_then_insert(self, schema, table_name, data, **kwargs):
        """Create table from schema then insert data if provided."""
        
        # Create empty table from schema
        result = self.create_table_from_schema(schema, table_name, **kwargs)
        
        # Insert data if provided
        if data is not None:
            if self._is_single_row_data(data, check_type(data)):
                result.append_row(data, **kwargs)
            else:
                result.append_batch(data, **kwargs)
        
        return result
    
    def get_schema(self):
        """
        Retrieve the schema of an existing table (like BeamElastic).
        
        Returns:
            dict: Schema dictionary mapping column names to Ibis data types
        """
        if self.level != "table":
            raise ValueError("get_schema only supported at table level")
        
        if not self.exists():
            raise ValueError(f"Table {self.table_name} does not exist")
        
        return self.schema
    
    def describe_schema(self):
        """
        Get a human-readable description of the table schema.
        
        Returns:
            str: Formatted schema description
        """
        if self.level != "table":
            raise ValueError("describe_schema only supported at table level")
        
        schema = self.get_schema()
        
        lines = [f"Table: {self.table_name}"]
        lines.append(f"Columns: {len(schema)}")
        lines.append("Schema:")
        
        for col_name, col_type in schema.items():
            lines.append(f"  {col_name}: {col_type}")
        
        return "\n".join(lines)
    
    def infer_schema_class(self, class_name=None):
        """
        Create a BeamIbisSchema class from an existing table's schema.
        
        Args:
            class_name: Name for the generated schema class
            
        Returns:
            type: BeamIbisSchema subclass representing the table schema
        """
        if self.level != "table":
            raise ValueError("infer_schema_class only supported at table level")
        
        from .schema import BeamIbisSchema
        import ibis.expr.datatypes as dt
        
        schema = self.get_schema()
        
        if not class_name:
            class_name = f"{self.table_name.title().replace('_', '')}Schema"
        
        # Create class attributes dictionary
        class_attrs = {}
        
        # Map Ibis types back to Python types for annotations
        type_mapping = {
            dt.Int64: int,
            dt.Int32: int,
            dt.Int16: int,
            dt.Int8: int,
            dt.Float64: float,
            dt.Float32: float,
            dt.String: str,
            dt.Boolean: bool,
            dt.Timestamp: 'datetime',
            dt.Date: 'date',
            dt.Time: 'time',
            dt.JSON: dict,
            dt.Binary: bytes,
        }
        
        annotations = {}
        for col_name, col_type in schema.items():
            # Find the best Python type annotation
            python_type = str  # default fallback
            for ibis_type, py_type in type_mapping.items():
                if isinstance(col_type, ibis_type):
                    python_type = py_type
                    break
            
            annotations[col_name] = python_type
        
        # Create the class dynamically
        class_attrs['__annotations__'] = annotations
        class_attrs['__doc__'] = f"Auto-generated schema for table {self.table_name}"
        
        # Create the class
        schema_class = type(class_name, (BeamIbisSchema,), class_attrs)
        
        return schema_class

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
