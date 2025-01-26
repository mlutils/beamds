from elasticsearch_dsl import A
import pandas as pd


class Groupby:

    agg_name_mapping = {'mean': 'avg', 'sum': 'sum', 'min': 'min', 'max': 'max', 'nunique': 'cardinality',
                        'count': 'value_count'}

    def __init__(self, es, gb_field_names: str|list[str], size=10000):
        self.buckets = {}

        self.es = es
        gb_field_names = gb_field_names if isinstance(gb_field_names, list) else [gb_field_names]
        self.gb_field_names = [es.keyword_field(field_name) for field_name in gb_field_names]

        self.groups = [A('terms', field=field_name.removesuffix('.keyword'), size=size)
                       for field_name in self.gb_field_names]

    def add_aggregator(self, field_name, agg_type):
        bucket_name = f"{field_name.removesuffix('.keyword')}_{agg_type}"
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

    def get_group(self, group):
        return self.buckets[group]

    def __getitem__(self, ind):
        return self.get_group(ind)

    def agg(self, d):
        for k, v in d.items():
            agg_func = getattr(self, k)
            agg_func(v)
            # self.add_aggregator(k, self.agg_name_mapping.get(v, v))
        return self

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

        # -------------------------------------------------
        # 1) Build the nested aggregation structure
        # -------------------------------------------------
        # Start from the last grouping field and add all the metric buckets
        g = self.groups[-1]
        s = self.es._s  # the underlying Search object

        # attach each aggregator bucket (sum, avg, etc.) at the deepest level
        for bucket_name, bucket in self.buckets.items():
            g.bucket(bucket_name, bucket)

        # nest each higher-level groupby around the previous one
        for gi in self.groups[:-1][::-1]:
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
            if depth < len(self.groups) - 1:
                next_field = self.groups[depth + 1].field
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
                            row[bucket_name] = b[bucket_name].value
                        else:
                            row[bucket_name] = None

                    results.append(row)

        # Kick off recursion from the top-level aggregator
        first_field = self.groups[0].field
        first_agg_name = f'groupby_{first_field}'
        top_buckets = getattr(response.aggregations, first_agg_name).buckets
        recurse_buckets(top_buckets, 0, [])

        return results

    def as_df(self):
        res = self._apply()
        # pop index to different column
        index = [r.pop('index') for r in res]
        if len(self.gb_field_names) > 1:
            # construct multi-index
            index = pd.MultiIndex.from_tuples(index, names=[f for f in self.gb_field_names])
        df = pd.DataFrame(res, index=index)
        return df

    @property
    def values(self):
        v = self.as_df()
        v = v.drop(columns='count')
        return v

    def as_dict(self):
        return self._apply()



