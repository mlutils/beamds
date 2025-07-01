from datetime import datetime
import ibis
import pandas as pd


class Groupby:
    """GroupBy helper for BeamIbis, similar to BeamElastic's Groupby class."""

    # Mapping of aggregation names to Ibis methods
    agg_name_mapping = {
        'mean': 'mean', 
        'sum': 'sum', 
        'min': 'min', 
        'max': 'max', 
        'nunique': 'nunique',
        'count': 'count',
        'std': 'std',
        'var': 'var'
    }

    def __init__(self, beam_ibis, gb_field_names, size=10000, max_items=None, max_buckets=None):
        """
        Initialize GroupBy helper.
        
        Args:
            beam_ibis: BeamIbis instance
            gb_field_names: Field name(s) to group by
            size: Maximum number of groups to return
            max_items: Maximum number of items to process
            max_buckets: Maximum number of buckets allowed
        """
        self.beam_ibis = beam_ibis
        self.gb_field_names = gb_field_names if isinstance(gb_field_names, list) else [gb_field_names]
        self.size = size
        self.max_items = max_items
        self.max_buckets = max_buckets
        
        # Storage for aggregations
        self.aggregations = {}
        self.date_fields = set()  # Track date fields for special formatting

    def _add_aggregation(self, field_name, agg_type, alias=None):
        """Add an aggregation to the group by."""
        if alias is None:
            alias = f"{field_name.replace('.', '_')}_{agg_type}"
        
        # Check if it's a date field for special handling
        try:
            schema = self.beam_ibis.schema
            if schema and field_name in schema:
                field_type = str(schema[field_name]).lower()
                if 'timestamp' in field_type or 'datetime' in field_type or 'date' in field_type:
                    if agg_type in ['min', 'max', 'mean']:
                        self.date_fields.add(alias)
        except:
            pass  # Ignore schema errors
        
        self.aggregations[alias] = (field_name, agg_type)
        return self

    def sum(self, field_name):
        """Add sum aggregation."""
        return self._add_aggregation(field_name, 'sum')

    def mean(self, field_name):
        """Add mean aggregation."""
        return self._add_aggregation(field_name, 'mean')

    def avg(self, field_name):
        """Add average aggregation (alias for mean)."""
        return self.mean(field_name)

    def min(self, field_name):
        """Add min aggregation."""
        return self._add_aggregation(field_name, 'min')

    def max(self, field_name):
        """Add max aggregation."""
        return self._add_aggregation(field_name, 'max')

    def nunique(self, field_name):
        """Add nunique (cardinality) aggregation."""
        return self._add_aggregation(field_name, 'nunique')

    def count(self, field_name="*"):
        """Add count aggregation."""
        if field_name == "*":
            alias = "count"
        else:
            alias = f"{field_name.replace('.', '_')}_count"
        self.aggregations[alias] = (field_name, 'count')
        return self

    def std(self, field_name):
        """Add standard deviation aggregation."""
        return self._add_aggregation(field_name, 'std')

    def var(self, field_name):
        """Add variance aggregation."""
        return self._add_aggregation(field_name, 'var')

    def first(self, field_name, sort_field=None):
        """Add first value aggregation (using min of sort field)."""
        if sort_field is None:
            # Try to find a timestamp field
            schema = self.beam_ibis.schema
            timestamp_fields = []
            if schema:
                for fname, ftype in schema.items():
                    if 'timestamp' in str(ftype).lower() or 'datetime' in str(ftype).lower():
                        timestamp_fields.append(fname)
            
            if timestamp_fields:
                sort_field = timestamp_fields[0]
            else:
                # Fallback to any field that can be sorted
                sort_field = field_name
        
        # For first, we'll use a custom aggregation that's backend-specific
        alias = f"{field_name.replace('.', '_')}_first"
        self.aggregations[alias] = (field_name, 'first', sort_field)
        return self

    def last(self, field_name, sort_field=None):
        """Add last value aggregation (using max of sort field)."""
        if sort_field is None:
            # Try to find a timestamp field
            schema = self.beam_ibis.schema
            timestamp_fields = []
            if schema:
                for fname, ftype in schema.items():
                    if 'timestamp' in str(ftype).lower() or 'datetime' in str(ftype).lower():
                        timestamp_fields.append(fname)
            
            if timestamp_fields:
                sort_field = timestamp_fields[0]
            else:
                sort_field = field_name
        
        alias = f"{field_name.replace('.', '_')}_last"
        self.aggregations[alias] = (field_name, 'last', sort_field)
        return self

    def median(self, field_name):
        """Add median aggregation (50th percentile)."""
        alias = f"{field_name.replace('.', '_')}_median"
        self.aggregations[alias] = (field_name, 'median')
        return self

    def percentiles(self, field_name, percentiles=(25, 50, 75, 90)):
        """Add percentiles aggregation."""
        for p in percentiles:
            alias = f"{field_name.replace('.', '_')}_p{p}"
            self.aggregations[alias] = (field_name, 'percentile', p)
        return self

    def agg(self, agg_dict):
        """
        Add multiple aggregations at once.
        
        Args:
            agg_dict: Dictionary where keys are field names and values are 
                     aggregation functions (string or list of strings)
        """
        for field_name, agg_funcs in agg_dict.items():
            if isinstance(agg_funcs, str):
                agg_funcs = [agg_funcs]
            
            for agg_func in agg_funcs:
                if hasattr(self, agg_func):
                    getattr(self, agg_func)(field_name)
                else:
                    self._add_aggregation(field_name, agg_func)
        
        return self

    def circuit_breaker(self):
        """Check limits before executing query."""
        if self.max_items is not None:
            count = self.beam_ibis.count()
            if count > self.max_items:
                raise ValueError(f"Number of documents in the table exceeds the limit of {self.max_items}")
        
        if self.max_buckets is not None:
            for field in self.gb_field_names:
                nunique = self.beam_ibis.nunique(field)
                if nunique > self.max_buckets:
                    raise ValueError(f"Number of unique values in field {field} exceeds the limit of {self.max_buckets}")

    def _apply(self):
        """
        Execute the group by query and return results as list of dictionaries.
        """
        self.circuit_breaker()
        
        # Get the base query table
        table = self.beam_ibis.query_table
        
        # Group by the specified fields
        grouped = table.group_by(self.gb_field_names)
        
        # Build aggregations dictionary
        agg_exprs = {}
        
        # Always include count
        agg_exprs['count'] = ibis._.count()
        
        # Add user-defined aggregations
        for alias, agg_info in self.aggregations.items():
            field_name = agg_info[0]
            agg_type = agg_info[1]
            
            if field_name == "*" and agg_type == "count":
                continue  # Already handled above
            
            col = table[field_name]
            
            if agg_type == 'sum':
                agg_exprs[alias] = col.sum()
            elif agg_type == 'mean':
                agg_exprs[alias] = col.mean()
            elif agg_type == 'min':
                agg_exprs[alias] = col.min()
            elif agg_type == 'max':
                agg_exprs[alias] = col.max()
            elif agg_type == 'nunique':
                agg_exprs[alias] = col.nunique()
            elif agg_type == 'count':
                agg_exprs[alias] = col.count()
            elif agg_type == 'std':
                agg_exprs[alias] = col.std()
            elif agg_type == 'var':
                agg_exprs[alias] = col.var()
            elif agg_type == 'median':
                # Median is typically percentile 50
                try:
                    agg_exprs[alias] = col.quantile(0.5)
                except:
                    # Fallback if quantile not supported
                    agg_exprs[alias] = col.median() if hasattr(col, 'median') else None
            elif agg_type == 'percentile':
                percentile = agg_info[2] / 100.0  # Convert to 0-1 range
                try:
                    agg_exprs[alias] = col.quantile(percentile)
                except:
                    agg_exprs[alias] = None
            elif agg_type in ['first', 'last']:
                # For first/last, we need more complex logic
                # This is a simplified version - in practice you might need window functions
                if agg_type == 'first':
                    agg_exprs[alias] = col.min()  # Simplified
                else:
                    agg_exprs[alias] = col.max()  # Simplified
            else:
                # Unknown aggregation type
                agg_exprs[alias] = None
        
        # Apply aggregations
        result_expr = grouped.aggregate(**agg_exprs)
        
        # Add ordering and limit
        if self.size is not None:
            result_expr = result_expr.order_by(ibis.desc('count')).limit(self.size)
        
        # Execute the query
        df = result_expr.execute()
        
        # Convert to list of dictionaries
        results = []
        for _, row in df.iterrows():
            row_dict = {}
            
            # Set the index (group keys)
            if len(self.gb_field_names) == 1:
                row_dict['index'] = row[self.gb_field_names[0]]
            else:
                row_dict['index'] = tuple(row[field] for field in self.gb_field_names)
            
            # Add aggregation results
            for col_name, value in row.items():
                if col_name not in self.gb_field_names:
                    # Handle date formatting for date fields
                    if col_name in self.date_fields and pd.notna(value):
                        try:
                            if isinstance(value, (int, float)):
                                # Assume milliseconds timestamp
                                row_dict[col_name] = datetime.fromtimestamp(value / 1000)
                            else:
                                row_dict[col_name] = value
                        except:
                            row_dict[col_name] = value
                    else:
                        row_dict[col_name] = value
            
            results.append(row_dict)
        
        return results

    def as_df(self):
        """Execute and return results as pandas DataFrame."""
        results = self._apply()
        df = pd.DataFrame(results)
        
        if 'index' in df.columns:
            df = df.set_index('index')
            
            # Set appropriate index names
            if len(self.gb_field_names) > 1:
                df.index = pd.MultiIndex.from_tuples(
                    df.index, 
                    names=[field.replace('.', '_') for field in self.gb_field_names]
                )
            else:
                df.index.name = self.gb_field_names[0].replace('.', '_')
        
        return df

    def as_pl(self):
        """Execute and return results as Polars DataFrame."""
        import polars as pl
        results = self._apply()
        df = pl.DataFrame(results)
        
        # Rename index column if present
        if "index" in df.columns:
            if len(self.gb_field_names) == 1:
                df = df.rename({"index": self.gb_field_names[0].replace('.', '_')})
            else:
                # For multi-index, keep as index column
                df = df.rename({"index": "group_keys"})
        
        return df

    def as_cudf(self):
        """Execute and return results as cuDF DataFrame."""
        import cudf
        results = self._apply()
        df = cudf.DataFrame(results)
        
        if 'index' in df.columns:
            df = df.set_index('index')
            
            # Set appropriate index names  
            if len(self.gb_field_names) > 1:
                df.index = cudf.MultiIndex.from_tuples(
                    df.index.to_pandas(),
                    names=[field.replace('.', '_') for field in self.gb_field_names]
                )
            else:
                df.index.name = self.gb_field_names[0].replace('.', '_')
        
        return df

    def as_dict(self):
        """Execute and return results as list of dictionaries."""
        return self._apply()

    @property
    def values(self):
        """Get values without count column."""
        df = self.as_df()
        if 'count' in df.columns:
            df = df.drop(columns=['count'])
        return df
