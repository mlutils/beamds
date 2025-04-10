from datetime import datetime

from elasticsearch_dsl import A
import pandas as pd


class Groupby:

    agg_name_mapping = {'mean': 'avg', 'sum': 'sum', 'min': 'min', 'max': 'max', 'nunique': 'cardinality',
                        'count': 'value_count'}

    def __init__(self, es, gb_field_names: str|list[str], size=10000, max_items=None, max_buckets=None,
                 use_composite=None):
        self.buckets = {}

        self.es = es
        gb_field_names = gb_field_names if isinstance(gb_field_names, list) else [gb_field_names]
        self.gb_field_names = [es.keyword_field(field_name) for field_name in gb_field_names]

        self.use_composite = use_composite if use_composite is not None else (
                    size is None or size > 10000 or len(self.gb_field_names) > 1)

        self.size = size
        # self.groups = [A('terms', field=field_name, size=size)
        #                for field_name in self.gb_field_names]

        self.date_buckets = []
        self.max_items = max_items
        self.max_buckets = max_buckets

    def add_aggregator(self, field_name, agg_type):
        bucket_name = f"{field_name.removesuffix('.keyword')}_{agg_type}"
        if self.es.is_date_field(field_name) and agg_type in ['min', 'max', 'avg']:
            self.date_buckets.append(bucket_name)
        self.buckets[bucket_name] = A(agg_type, field=field_name)

    def sum(self, field_name):
        self.add_aggregator(field_name, 'sum')
        return self

    def mean(self, field_name):
        self.add_aggregator(field_name, 'avg')
        return self

    def min(self, field_name):
        self.add_aggregator(field_name, 'min')
        return self

    def max(self, field_name):
        self.add_aggregator(field_name, 'max')
        return self

    def nunique(self, field_name):
        field_name = self.es.keyword_field(field_name)
        self.add_aggregator(field_name, 'cardinality')
        return self

    def count(self, field_name):
        field_name = self.es.keyword_field(field_name)
        self.add_aggregator(field_name, 'value_count')
        return self

    def list(self, field_name):
        field_name = self.es.keyword_field(field_name)
        self.add_aggregator(field_name, 'top_hits')
        return self

    def first(self, field_name, sort_field=None):
        if not sort_field:
            sort_field = self.es.default_timestamp_field  # Auto-detect if available
        bucket_name = f"{field_name.removesuffix('.keyword')}_first"
        self.buckets[bucket_name] = A('top_hits', size=1, _source=[field_name], sort=[{sort_field: {"order": "asc"}}])
        return self

    def last(self, field_name, sort_field=None):
        if not sort_field:
            sort_field = self.es.default_timestamp_field
        bucket_name = f"{field_name.removesuffix('.keyword')}_last"
        self.buckets[bucket_name] = A('top_hits', size=1, _source=[field_name], sort=[{sort_field: {"order": "desc"}}])
        return self

    def median(self, field_name):
        """
        Returns the median (50th percentile) value of `field_name`.
        """
        bucket_name = f"{field_name.removesuffix('.keyword')}_median"
        self.buckets[bucket_name] = A('percentiles', field=field_name, percents=[50])
        return self

    def percentiles(self, field_name, percents=(25, 50, 75, 90)):
        """
        Returns multiple percentiles for `field_name`.
        """
        percents = list(percents)
        bucket_name = f"{field_name.removesuffix('.keyword')}_percentiles"
        self.buckets[bucket_name] = A('percentiles', field=field_name, percents=percents)
        return self

    def get_group(self, group):
        return self.buckets[group]

    def __getitem__(self, ind):
        return self.get_group(ind)

    def agg(self, d):
        for k, v in d.items():
            if type(v) is list:
                for agg_i in v:
                    agg_func = getattr(self, agg_i)
                    agg_func(k)
            else:
                agg_func = getattr(self, v)
                agg_func(k)
            # self.add_aggregator(k, self.agg_name_mapping.get(v, v))
        return self

    def circuit_breaker(self):
        if self.max_items is not None:
            if self.es.count() > self.max_items:
                raise ValueError(f"Number of documents in the index exceeds the limit of {self.max_items}")
        if self.max_buckets is not None:
            for field in self.gb_field_names:
                if self.es.nunique(field) > self.max_buckets:
                    raise ValueError(f"Number of unique values in the field {field} exceeds the limit of {self.max_buckets}")

    def _apply(self):
        """
        Build the nested Elasticsearch aggregation, execute it,
        and recursively parse the aggregation buckets at each level.
        Returns a list of dicts, where each dict represents one grouped row:
            {
              'index': (group_key_1, group_key_2, ...),
              'count': doc_count_in_that_bucket,
              'fieldName_aggtype': value_of_aggregator,
              ...
            }
        """

        self.circuit_breaker()

        results = []

        if self.use_composite:
            # ---------------------------------------
            # COMPOSITE AGGREGATION IMPLEMENTATION
            # ---------------------------------------

            after_key = None
            while True:

                s = self.es._s  # the underlying Search object
                composite_kwargs = {}
                if after_key:
                    composite_kwargs['after'] = after_key

                sources = [{field: {"terms": {"field": field}}} for field in self.gb_field_names]
                composite_agg = A('composite', sources=sources, size=1000, **composite_kwargs)  # 1000 buckets per batch

                # Attach aggregations to the composite aggregation
                for bucket_name, bucket in self.buckets.items():
                    composite_agg.metric(bucket_name, bucket)

                s.aggs.bucket('composite_group', composite_agg)

                response = s.execute()
                buckets = response.aggregations.composite_group.buckets

                for b in buckets:
                    row = {"index": tuple(b.key[field] for field in self.gb_field_names)}
                    row["count"] = b.doc_count

                    # Collect aggregations
                    for bucket_name in self.buckets.keys():
                        if hasattr(b, bucket_name):
                            if bucket_name in self.date_buckets:
                                row[bucket_name] = datetime.fromtimestamp(b[bucket_name].value / 1000)
                            else:
                                row[bucket_name] = b[bucket_name].value
                        else:
                            row[bucket_name] = None

                    results.append(row)

                if len(buckets) < self.es.maximum_bucket_limit:
                    break
                after_key = response.aggregations.composite_group.after_key

        else:

            # -------------------------------------------------
            # 1) Build the nested aggregation structure
            # -------------------------------------------------
            # Start from the last grouping field and add all the metric buckets
            # g = self.groups[-1]

            s = self.es._s  # the underlying Search object

            g = A('terms', field=self.gb_field_names[-1], size=self.size)

            # attach each aggregator bucket (sum, avg, etc.) at the deepest level
            for bucket_name, bucket in self.buckets.items():
                g.bucket(bucket_name, bucket)

            # nest each higher-level groupby around the previous one
            # for gi in self.groups[:-1][::-1]:
            #     gi.bucket(f'groupby_{g.field}', g)
            #     g = gi
            for field in reversed(self.gb_field_names[:-1]):
                gi = A('terms', field=field, size=self.size)
                gi.bucket(f'groupby_{g.field}', g)
                g = gi

            # Attach top-level aggregator
            top_agg_name = f'groupby_{g.field}'
            s.aggs.bucket(top_agg_name, g)

            # -------------------------------------------------
            # 2) Execute the query
            # -------------------------------------------------
            response = s.execute()

            # -------------------------------------------------
            # 3) Recursively parse the buckets
            # -------------------------------------------------
            results = []

            def recurse_buckets(buckets, depth, parents):
                """
                Traverse the buckets recursively, collecting aggregator results.
                  buckets: the list of ES bucket objects at this depth
                  depth: which group field index we are processing
                  parents: list of group keys collected so far
                """
                # if we are not at the last grouping level, we keep going deeper
                if depth < len(self.gb_field_names) - 1:
                    next_field = self.gb_field_names[depth + 1]
                    next_agg_name = f'groupby_{next_field}'

                    for b in buckets:
                        # accumulate the current bucket key
                        new_parents = parents + [b.key]

                        # recursively traverse the next set of sub-buckets
                        sub_buckets = getattr(b, next_agg_name).buckets
                        recurse_buckets(sub_buckets, depth + 1, new_parents)

                else:
                    # this is the last grouping level, so collect aggregator data
                    for b in buckets:
                        new_parents = parents + [b.key]
                        row = {}

                        # store group keys as a tuple in 'index'
                        if len(new_parents) == 1:
                            row['index'] = new_parents[0]
                        else:
                            row['index'] = tuple(new_parents)

                        # doc_count for this bucket
                        row['count'] = b.doc_count

                        # gather aggregator results
                        for bucket_name in self.buckets.keys():
                            # ES DSL aggregator result is typically b[bucket_name].value
                            # If there's no aggregator result, store None
                            if hasattr(b, bucket_name):
                                if bucket_name in self.date_buckets:
                                    row[bucket_name] = datetime.fromtimestamp(b[bucket_name].value / 1000)
                                else:
                                    row[bucket_name] = b[bucket_name].value
                            else:
                                row[bucket_name] = None

                        results.append(row)

            # Kick off recursion from the top-level aggregator
            first_field = self.gb_field_names[0]
            first_agg_name = f'groupby_{first_field}'
            top_buckets = getattr(response.aggregations, first_agg_name).buckets
            recurse_buckets(top_buckets, 0, [])

        return results

    def as_df(self):
        res = self._apply()
        df = pd.DataFrame(res)
        df = df.set_index('index')
        if len(self.gb_field_names) > 1:
            df.index = pd.MultiIndex.from_tuples(df.index, names=[f.removesuffix('.keyword') for f in self.gb_field_names])
        else:
            df.index.name = self.gb_field_names[0].removesuffix('.keyword')
        return df

    def as_pl(self):

        import polars as pl
        res = self._apply()
        df = pl.DataFrame(res)

        # Set index
        if "index" in df.columns:
            df = df.rename({"index": "_id"})

        # Handle multi-index or single index name
        if len(self.gb_field_names) > 1:
            df = df.with_columns(pl.struct(self.gb_field_names).alias("_id"))
        else:
            df = df.rename({"_id": self.gb_field_names[0].removesuffix('.keyword')})

        return df

    def as_cudf(self):
        import cudf
        res = self._apply()
        df = cudf.DataFrame(res)

        # Set index
        if "index" in df.columns:
            df = df.set_index("index")

        # Handle multi-index or single index name
        if len(self.gb_field_names) > 1:
            df.index = cudf.MultiIndex.from_tuples(df.index.to_pandas(),
                                                   names=[f.removesuffix('.keyword') for f in self.gb_field_names])
        else:
            df.index.name = self.gb_field_names[0].removesuffix('.keyword')

        return df

    @property
    def values(self):
        v = self.as_df()
        v = v.drop(columns='count')
        return v

    def as_dict(self):
        return self._apply()



