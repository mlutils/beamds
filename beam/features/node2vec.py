import pandas as pd

from .feature import FeaturesCategories, BeamFeature, ParameterSchema, ParameterType
import tempfile
from beam import resource
from functools import cached_property
from itertools import combinations
from collections import Counter

from ..type.utils import is_pandas_series


# class FeaturesCategories(Enum):
#     numerical = 'numerical'
#     categorical = 'categorical'
#     embedding = 'embedding'
#     text = 'text'
#
#
# class ParameterType(Enum):
#     categorical = 'categorical'
#     linspace = 'linspace'
#     logspace = 'logspace'
#     uniform = 'uniform'
#     loguniform = 'loguniform'


class Node2Vec(BeamFeature):
    def __init__(self, *args, name=None, q=None, p=None, n_workers=1, verbose=False, num_walks=10, walk_length=80,
                vector_size=8, window=3, min_count=0, sg=1, workers=1, epochs=1, **kwargs):
        super().__init__(*args, name=name, kind=FeaturesCategories.embedding, **kwargs)
        self.q = q or self.parameters_schema['q'].default_value
        self.p = p or self.parameters_schema['p'].default_value
        self.num_walks = num_walks or self.parameters_schema['num_walks'].default_value
        self.walk_length = walk_length or self.parameters_schema['walk_length'].default_value
        self.vector_size = vector_size or self.parameters_schema['vector_size'].default_value
        self.window = window or self.parameters_schema['window'].default_value
        self.min_count = min_count or self.parameters_schema['min_count'].default_value
        self.sg = sg or self.parameters_schema['sg'].default_value
        self.verbose = verbose
        self.model = None
        self.n_workers = n_workers or self.parameters_schema['n_workers'].default_value

    @cached_property
    def parameters_schema(self):
        return {
            'q': ParameterSchema(name='q', kind=ParameterType.uniform,
                                 min_value=0.1, max_value=10, default_value=1, description='Node2Vec q parameter'),
            'p': ParameterSchema(name='p', kind=ParameterType.uniform,
                                 min_value=0.1, max_value=10, default_value=1, description='Node2Vec p parameter'),
            'vector_size': ParameterSchema(name='vector_size', kind=ParameterType.linspace, min_value=1, max_value=100,
                                           default_value=8, description='Size of embeddings'),
            'window': ParameterSchema(name='window', kind=ParameterType.linspace, min_value=1, max_value=10,
                                        default_value=3, description='Window size'),
            'min_count': ParameterSchema(name='min_count', kind=ParameterType.linspace, min_value=0, max_value=10,
                                        default_value=0, description='Minimum count'),
            'sg': ParameterSchema(name='sg', kind=ParameterType.categorical, possible_values=[0, 1],
                                        default_value=1, description='Skip-gram'),
            'num_walks': ParameterSchema(name='num_walks', kind=ParameterType.linspace, min_value=1, max_value=100,
                                        default_value=10, description='Number of walks'),
            'walk_length': ParameterSchema(name='walk_length', kind=ParameterType.linspace, min_value=1, max_value=100,
                                        default_value=80, description='Walk length'),

        }

    def _fit(self, x, weighted=False, directed=False):
        from gensim.models.word2vec import Word2Vec
        from pecanpy import pecanpy

        with tempfile.TemporaryDirectory() as tmp_dir:
            file = resource(tmp_dir).joinpath('data.csv')
            file.write(x, index=False, header=False, sep='\t')
            g = pecanpy.SparseOTF(p=self.p, q=self.q, workers=self.n_workers, verbose=self.verbose)
            g.read_edg(file.str, weighted=weighted, directed=directed)

        walks = g.simulate_walks(num_walks=self.num_walks, walk_length=self.walk_length)
        # use random walks to train embeddings
        w2v_model = Word2Vec(walks, vector_size=self.vector_size, window=self.window,
                             min_count=self.min_count, sg=self.sg, workers=self.workers, epochs=self.epochs)

        self.model = w2v_model

    def fit(self, x, source='source', target='target', directed=False, weight='weight', weighted=False):

        x = x[[source, target, weight]] if weighted else x[[source, target]]
        self._fit(x, weighted=weighted, directed=directed)

    def transform(self, x, column=None, index=None):
        assert self.model is not None, 'Model is not trained'
        if column is None:
            assert is_pandas_series(x), 'x should be a pandas Series'
        else:
            x = x[column]
        x = x.values
        index = index or x.index

        return pd.DataFrame([self.model.wv[i] for i in x], index=index)

    def fit_transform(self, x, g=None, source='source', target='target',
                      directed=False, weight='weight', weighted=False, column=None, index=None):
        g = g or x
        self.fit(g, source, target, directed, weight, weighted)
        return self.transform(x, column, index)


class Set2Vec(Node2Vec):

    def __init__(self, *args, aggregation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregation = aggregation or 'mean'

    @cached_property
    def parameters_schema(self):
        return {
            **super().parameters_schema,
            'aggregation': ParameterSchema(name='aggregation', kind=ParameterType.categorical,
                                           possible_values=['mean', 'sum', 'max', 'min'],
                                           default_value='mean', description='Aggregation function')
        }

    def fit(self, x, index=None, **kwargs):
        # Step 1: Generate pairs and count their occurrences
        pair_counts = Counter()

        # Iterate over each list in the Series
        for lst in x:
            # Generate unique pairs from the list
            pairs = combinations(lst, 2)
            # Update the counter with the pairs
            pair_counts.update(pairs)

        # Step 2: Construct the DataFrame
        df = pd.DataFrame([{'n1': n1, 'n2': n2, 'weight': weight} for (n1, n2), weight in pair_counts.items()])
        super().fit(df, source='n1', target='n2', weight='weight', weighted=True, directed=False)

    def transform(self, x, column=None, index=None):
        assert self.model is not None, 'Model is not trained'
        if column is None:
            assert is_pandas_series(x), 'x should be a pandas Series'
        else:
            x = x[column]
        x = x.values
        index = index or x.index

        # aggregate the embeddings of the nodes in the set
        if self.aggregation == 'mean':
            return pd.DataFrame([self.model.wv[i].mean(axis=0) for i in x], index=index)
        elif self.aggregation == 'sum':
            return pd.DataFrame([self.model.wv[i].sum(axis=0) for i in x], index=index)
        elif self.aggregation == 'max':
            return pd.DataFrame([self.model.wv[i].max(axis=0) for i in x], index=index)
        elif self.aggregation == 'min':
            return pd.DataFrame([self.model.wv[i].min(axis=0) for i in x], index=index)
        else:
            raise ValueError(f'Unknown aggregation function: {self.aggregation}')

    def fit_transform(self, x, index=None, **kwargs):
        self.fit(x, index)
        return self.transform(x, index=index)

