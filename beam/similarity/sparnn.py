import numpy as np

import scipy.sparse as sp
from .core import BeamSimilarity, Similarities


class SparnnSimilarity(BeamSimilarity):

    def __init__(self, *args, k_clusters=10, matrix_size=None, num_indexes=2, **kwargs):
        super().__init__(*args, k_clusters=k_clusters, matrix_size=matrix_size, num_indexes=num_indexes, **kwargs)
        self.k_clusters = self.get_hparam('k_clusters', k_clusters)
        self.matrix_size = self.get_hparam('matrix_size', matrix_size)
        self.num_indexes = self.get_hparam('num_indexes', num_indexes)

        self.index = None
        self.vectors = None
        self.cluster = None

    def reset(self):
        self.index = None
        self.vectors = None
        self.cluster = None

    @classmethod
    @property
    def special_state_attributes(cls):
        return super(SparnnSimilarity, cls).special_state_attributes.union(['index', 'vectors', 'cluster'])

    def add(self, x, index=None, **kwargs):

        x, index = self.extract_data_and_index(x, index, convert_to='scipy_csr')
        self.add_index(x, index)
        if self.vectors is None:
            self.vectors = x
        else:
            self.vectors = sp.vstack([self.vectors, x])

    def fit(self, x=None, index=None, **kwargs):

        if x is not None:
            self.add(x, index)
        if self.vectors is None:
            raise ValueError('No vectors to fit')

        import pysparnn.cluster_index as ci
        self.cluster = ci.MultiClusterIndex(self.vectors, np.arange(len(self.index)), num_indexes=self.num_indexes,
                                            matrix_size=self.matrix_size)

    def search(self, x, k=1) -> Similarities:
        x, _ = self.extract_data_and_index(x, convert_to='scipy_csr')
        if self.cluster is None:
            self.fit()

        res = self.cluster.search(x, k=k, k_clusters=self.k_clusters, return_distance=True)
        res = np.array(res).transpose(2, 0, 1)
        I = res[1].astype(np.int64)
        D = res[0]

        return Similarities(index=self.index[I], distance=D, metric='cosine', model='sparnn')
