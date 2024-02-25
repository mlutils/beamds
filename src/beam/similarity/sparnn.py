import numpy as np
import pysparnn.cluster_index as ci

import scipy.sparse as sp
from scipy.sparse import csr_matrix

from .. import as_numpy, check_type
from ..similarity.core import BeamSimilarity


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

    def to_sparse(self, x):
        x_type = check_type(x)

        if x_type.minor == 'scipy_sparse':
            x = x.tocsr()

        elif x_type.minor in ['tensor', 'numpy']:
            x = as_numpy(x)

            x = csr_matrix(x)

        elif x_type.minor == 'dict':
            x = csr_matrix((x['val'], (x['row'], x['col'])))

        elif x_type.minor == 'tuple':
            x = csr_matrix((x[2], (x[0], x[1])))

        return x

    def add(self, x, index=None, **kwargs):
        x = self.to_sparse(x)
        if self.vectors is None:
            self.vectors = x
        else:
            self.vectors = sp.vstack([self.vectors, x])

        if index is not None:
            if self.index is None:
                self.index = index
            else:
                self.index = np.concatenate([self.index, index])
        else:
            if self.index is None:
                self.index = np.arange(len(x), device=self.device)
            else:
                index = np.arange(len(x), device=self.device) + self.index.max() + 1
                self.index = np.concatenate([self.index, index])
