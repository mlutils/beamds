from elasticsearch_dsl import A

class Groupby:

    agg_name_mapping = {'mean': 'avg', 'sum': 'sum', 'min': 'min', 'max': 'max', 'nunique': 'cardinality',
                        'count': 'value_count'}

    def __init__(self, es, gb_field_names: str|list[str], size=10000):
        self.buckets = {}

        self.es = es
        gb_field_names = gb_field_names if isinstance(gb_field_names, list) else [gb_field_names]
        self.gb_field_names = [es.keyword_field(field_name) for field_name in gb_field_names]

        self.groups = [A('terms', field=field_name, size=size)
                       for field_name in gb_field_names]

    def add_aggregator(self, field_name, agg_type):
        bucket_name = f"{field_name}_{agg_type}"
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
        self.add_aggregator(field_name, 'cardinality')
        return self

    def count(self, field_name):
        self.add_aggregator(field_name, 'value_count')
        return self

    def get_group(self, group):
        return self.buckets[group]

    def __getitem__(self, ind):
        return self.get_group(ind)

    def agg(self, d):
        for k, v in d.items():
            self.add_aggregator(k, self.agg_name_mapping.get(v, v))
        return self

    def _apply(self):

        g = self.groups[-1]
        s = self.es._s

        for bucket_name, bucket in self.buckets.items():
            g.bucket(bucket_name, bucket)

        for gi in self.groups[:-1][::-1]:
            g = gi.bucket(f'groupby_{gi.field}', g)

        s.aggs.bucket(f'groupby_{g.field}', g)

        response = s.execute()

        # # extract all the buckets from the response
        # for bucket_name, bucket in self.buckets.items():
        #     self.buckets[bucket_name] = response.aggregations[f'groupby_{g.field}'][bucket_name]

        return response





