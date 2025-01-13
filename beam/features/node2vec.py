import numpy as np
import pandas as pd

from .feature import FeaturesCategories, BeamFeature, ParameterSchema, ParameterType
import tempfile

from ..utils import as_numpy, tqdm_beam as tqdm
from ..resources import resource
from functools import cached_property, wraps
from itertools import combinations
from collections import Counter

from ..type.utils import is_pandas_series


class Node2Vec(BeamFeature):

    @wraps(BeamFeature.__init__)
    def __init__(self, *args, q=None, p=None, n_workers=1, verbose=False, num_walks=10, walk_length=80,
                vector_size=8, window=3, min_count=0, sg=1, epochs=None, **kwargs):
        super().__init__(*args, kind=FeaturesCategories.embedding, **kwargs)
        self.q = q or self.parameters_schema['q'].default
        self.p = p or self.parameters_schema['p'].default
        self.num_walks = num_walks or self.parameters_schema['num_walks'].default
        self.walk_length = walk_length or self.parameters_schema['walk_length'].default
        self.vector_size = vector_size or self.parameters_schema['vector_size'].default
        self.window = window or self.parameters_schema['window'].default
        self.min_count = min_count or self.parameters_schema['min_count'].default
        self.sg = sg or self.parameters_schema['sg'].default
        self.verbose = verbose
        self.model = None
        self.n_workers = n_workers or self.parameters_schema['n_workers'].default
        self.epochs = epochs or self.parameters_schema['epochs'].default

    @cached_property
    def parameters_schema(self):

        return super().parameters_schema | {
            'q': ParameterSchema(name='q', kind=ParameterType.uniform,
                                 start=0.1, end=10, default=1, description='Node2Vec q parameter'),
            'p': ParameterSchema(name='p', kind=ParameterType.uniform,
                                 start=0.1, end=10, default=1, description='Node2Vec p parameter'),
            'vector_size': ParameterSchema(name='vector_size', kind=ParameterType.linspace, start=1, end=100,
                                           default=8, description='Size of embeddings'),
            'window': ParameterSchema(name='window', kind=ParameterType.linspace, start=1, end=10,
                                      default=3, description='Window size'),
            'min_count': ParameterSchema(name='min_count', kind=ParameterType.linspace, start=0, end=10,
                                         default=0, description='Minimum count'),
            'sg': ParameterSchema(name='sg', kind=ParameterType.categorical, choices=[0, 1],
                                  default=1, description='Skip-gram'),
            'num_walks': ParameterSchema(name='num_walks', kind=ParameterType.linspace, start=1, end=100,
                                         default=10, description='Number of walks'),
            'walk_length': ParameterSchema(name='walk_length', kind=ParameterType.linspace, start=1, end=100,
                                           default=80, description='Walk length'),
            'epochs': ParameterSchema(name='epochs', kind=ParameterType.linspace, start=1, end=12,
                                      default=5, description='Number of epochs in WV training'),

        }

    def transform_cell(self, i):
        return self.model.wv[i] if i in self.model.wv.key_to_index else self.na_value

    @cached_property
    def na_value(self):
        return np.zeros(self.model.vector_size)

    def fit_callback(self, x, source='source', target='target', directed=False, weight='weight', weighted=False):

        x = x[[source, target, weight]] if weighted else x[[source, target]]
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
                             min_count=self.min_count, sg=self.sg, workers=self.n_workers, epochs=self.epochs)

        self.model = w2v_model

    def transform_callback(self, x, **kwargs):
        assert self.model is not None, 'Model is not trained'
        return x.applymap(self.transform_cell)

    def fit_transform(self, x, g=None, source='source', target='target',
                      directed=False, weight='weight', weighted=False, column=None, index=None):

        assert g is not None, 'Graph is not provided'
        self.fit(g, source=source, target=target, directed=directed, weight=weight, weighted=weighted)
        return self.transform(x)


class Set2Vec(Node2Vec):

    def __init__(self, *args, aggregation=None, self_loop=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregation = aggregation or 'mean'
        self.self_loop = self_loop

    @cached_property
    def parameters_schema(self):
        return {
            **super().parameters_schema,
            'aggregation': ParameterSchema(name='aggregation', kind=ParameterType.categorical,
                                           choices=['mean', 'sum', 'max', 'min'],
                                           default='mean', description='Aggregation function'),
            'self_loop': ParameterSchema(name='self_loop', kind=ParameterType.categorical,
                                         choices=[True, False],
                                         default=True, description='Include self loops in the graph'),
        }

    def transform_cell(self, i):
        try:
            return self.model.wv[i]
        except KeyError:
            return np.stack([Node2Vec.transform_cell(self, j) for j in i])

    def fit_callback(self, x, **kwargs):
        # Step 1: Generate pairs and count their occurrences
        pair_counts = Counter()

        # Iterate over each list in the Series
        for lst in tqdm(np.squeeze(as_numpy(x))):
            # Generate unique pairs from the list

            lst = set(lst)
            pairs = combinations(lst, 2)

            pairs = [(c1, c2) if c1 < c2 else (c2, c1) for c1, c2 in pairs]
            if self.self_loop:
                pairs = pairs + [(n, n) for n in lst]

            # Update the counter with the pairs
            pair_counts.update(pairs)

        # Step 2: Construct the DataFrame
        df = pd.DataFrame([{'n1': n1, 'n2': n2, 'weight': weight} for (n1, n2), weight in pair_counts.items()])
        Node2Vec.fit_callback(self, df, source='n1', target='n2', weight='weight', weighted=True, directed=False)

    def transform_callback(self, x: pd.DataFrame, **kwargs):
        assert self.model is not None, 'Model is not trained'

        x = x.squeeze()
        # aggregate the embeddings of the nodes in the set
        if self.aggregation == 'mean':
            v = [self.transform_cell(i).mean(axis=0) for i in x.values]
        elif self.aggregation == 'sum':
            v = [self.transform_cell(i).sum(axis=0) for i in x.values]
        elif self.aggregation == 'max':
            v = [self.transform_cell(i).max(axis=0) for i in x.values]
        elif self.aggregation == 'min':
            v = [self.transform_cell(i).min(axis=0) for i in x.values]
        else:
            raise ValueError(f'Unknown aggregation function: {self.aggregation}')

        return pd.Series(v, index=x.index).apply(list).to_frame(name=self.name)

    def fit_transform(self, x, **kwargs):
        return BeamFeature.fit_transform(self, x, **kwargs)

