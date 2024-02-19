from typing import List, Union

import numpy as np
import torch

from .core import BeamSimilarity
from .. import as_numpy, BeamData
from ..core import Processor
from ..transformer import Transformer
from ..utils import check_type, as_tensor, beam_device, tqdm_beam as tqdm, lazy_property
from ..parallel import BeamParallel
from collections import Counter
import scipy.sparse as sp


class ChunkTF(Transformer):

    def __init__(self, *args, sparse_framework='torch', device='cpu', preprocessor=None, **kwargs):
        self.preprocessor = preprocessor or TFIDF.default_preprocessor
        self.sparse_framework = sparse_framework
        self._device = device
        super().__init__(*args, **kwargs)

    @lazy_property
    def device(self):
        from ..utils import beam_device
        return beam_device(self._device)

    def transform_callback(self, x, tokens=None, binary=False, **kwargs):

        values = []
        ind_ptrs = [0]
        cols = []

        for xi in x:
            xi = self.preprocessor(xi)
            if binary:
                xi = list(set(xi))
            c = Counter(xi)
            c = Counter({k: v for k, v in c.items() if k in tokens})

            ind_ptrs.append(len(c))
            if self.sparse_framework == 'torch':

                values.append(torch.FloatTensor(list(c.values()), device=self.device))
                cols.append(torch.LongTensor(list(c.keys()), device=self.device))

            else:
                cols.append(np.array(list(c.keys())))
                values.append(np.array(list(c.values())))

        if self.sparse_framework == 'torch':
            ind_ptrs = torch.cumsum(torch.tensor(ind_ptrs, device=self.device), dim=0)
            y = torch.sparse_csr_tensor(torch.cat(cols), torch.cat(values), ind_ptrs, device=self.device,
                                        size=(len(x), len(tokens)))
        else:
            ind_ptrs = np.cumsum(ind_ptrs)
            y = sp.csr_matrix((np.concatenate(values), np.concatenate(cols), ind_ptrs),
                              shape=(len(x), len(tokens)))

        return y


class ChunkDF(Transformer):

        def __init__(self, *args, preprocessor=None, **kwargs):
            self.preprocessor = preprocessor or TFIDF.default_preprocessor
            super().__init__(*args, **kwargs)

        def transform_callback(self, x, key=None, is_chunk=False, fit=False, path=None, **kwargs):
            y = Counter()
            y_sum = Counter()
            for xi in x:
                xi = self.preprocessor(xi)
                y.update(set(xi))
                y_sum.update(xi)
            return y, y_sum


class TFIDF(Processor):

    def __init__(self, *args, preprocessor=None, min_df=None, max_df=None, max_features=None, use_idf=True,
                 smooth_idf=True, sublinear_tf=False, n_workers=0, n_chunks=None, chunksize=None, mp_method='joblib',
                 sparse_framework='torch', device='cpu', **kwargs):

        super().__init__(*args, min_df=min_df, max_df=max_df, max_features=max_features, use_idf=use_idf,
                         sparse_framework=sparse_framework, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf,
                         device=device, **kwargs)
        self.df = None
        self.cf = None  # corpus frequency
        self.n_docs = None
        self.tf = None
        self.reset()

        self.preprocessor = preprocessor or TFIDF.default_preprocessor

        self.min_df = self.get_hparam('min_df', min_df)
        self.max_df = self.get_hparam('max_df', max_df)
        self.max_features = self.get_hparam('max_features', max_features)
        self.use_idf = self.get_hparam('use_idf', use_idf)
        self.smooth_idf = self.get_hparam('smooth_idf', smooth_idf)
        self.sublinear_tf = self.get_hparam('sublinear_tf', sublinear_tf)
        self.sparse_framework = self.get_hparam('sparse_framework', sparse_framework)

        # we choose the csr layout for the sparse matrix
        # according to chatgpt it has some advantages over coo:
        # see https://chat.openai.com/share/9028c9f3-9695-4914-a15c-89902efa8837

        self.device = self.get_hparam('device', device)

        self.n_workers = self.get_hparam('n_workers', n_workers)
        n_chunks = self.get_hparam('n_chunks', n_chunks)
        chunksize = self.get_hparam('chunksize', chunksize)
        mp_method = self.get_hparam('mp_method', mp_method)

        self.chunk_tf = ChunkTF(n_workers=self.n_workers, n_chunks=n_chunks, chunksize=chunksize, mp_method=mp_method,
                                squeeze=False)
        self.chunk_df = ChunkDF(n_workers=self.n_workers, n_chunks=n_chunks, chunksize=chunksize, mp_method=mp_method,
                                squeeze=False)

    @staticmethod
    def default_preprocessor(x):

        x_type = check_type(x)

        if x_type.minor == 'torch':
            x = as_numpy(x).tolist()
        elif x_type.minor == 'numpy':
            x = x.tolist()
        else:
            x = list(x)

        return x

    def term_frequencies(self, x, tokens=None, **kwargs):
        if tokens is None:
            tokens = self.tokens
        return self.chunk_tf.transform(x, tokens=tokens, **kwargs)

    def bm25(self, q, k1=1.5, b=0.75):
        q_tf = self.term_frequencies(q, binary=True)
        len_norm = (1 - b) + b * self.doc_len / self.avg_doc_len
        bm25_tf = self.tf * (k1 + 1) / (self.tf + k1 * len_norm)

        idf = self.idf_bm25

        if self.sparse_framework == 'torch':
            idf = torch.cat([idf] * len(q_tf))
        else:
            idf = sp.vstack([idf] * len(q_tf))

        q_idf = q_tf * idf

        scores = q_idf.multiply(bm25_tf)
        return scores

    def transform(self, x: Union[List, List[List], BeamData]):

        x_type = check_type(x)
        if not x_type.major == 'container':
            x = [x]

        tfs = self.term_frequencies(x)
        if self.tf is None:
            self.tf = tfs
        else:
            if self.sparse_framework == 'torch':
                self.tf = torch.cat([self.tf, tfs])
            else:
                self.tf = sp.vstack([self.tf, tfs])

        if self.sparse_framework == 'torch':
            idf = torch.cat([self.idf] * len(tfs))
        else:
            idf = sp.vstack([self.idf] * len(tfs))

        tfidf = tfs * idf
        return tfidf

    def reset(self):
        self.df = Counter()
        self.cf = Counter()
        self.n_docs = 0
        self.tf = None
        self.clear_cache('idf', 'tokens', 'n_tokens', 'avg_doc_len', 'idf_bm25', 'doc_len')

    @lazy_property
    def tokens(self):
        """Build a mapping from tokens to indices based on filtered tokens."""
        return set(self.df.keys())

    @lazy_property
    def avg_doc_len(self):
        return sum(self.cf.values()) / self.n_docs

    @lazy_property
    def n_tokens(self):
        return max(list(self.tokens))

    @lazy_property
    def idf(self):

        if self.use_idf:
            if self.smooth_idf:
                idf_version = 'smooth'
            else:
                idf_version = 'standard'
        else:
            idf_version = 'unary'

        return self.calculate_idf(version=idf_version)

    @lazy_property
    def idf_bm25(self):
        return self.calculate_idf(version='bm25')

    @lazy_property
    def doc_len(self):
        if self.sparse_framework == 'torch':
            doc_lengths = self.tf.sum(dim=1)
        else:
            doc_lengths = self.tf.sum(axis=1).toarray()

        return doc_lengths

    def calculate_idf(self, version='standard'):
        """Calculate the inverse document frequency (IDF) vector.
        version: str, default='standard' [standard, smooth, unary, bm25]
        """
        n_docs = self.n_docs
        indptr = [0, len(self.df)]

        if self.sparse_framework == 'torch':
            col_indices = torch.LongTensor(list(self.df.keys()), device=self.device)
            log = torch.log
            ones_like = torch.ones_like
            framework_kwargs = {'device': self.device}
            array = torch.FloatTensor
        else:
            col_indices = np.array(list(self.df.keys()))
            log = np.log
            ones_like = np.ones_like
            framework_kwargs = {}
            array = np.array

        nq = array(list(self.df.values()), **framework_kwargs)
        if version == 'standard':
            values = log(n_docs / nq)
        elif version == 'smooth':
            values = log(n_docs / (nq + 1)) + 1
        elif version == 'unary':
            values = ones_like(col_indices, **framework_kwargs)
        elif version == 'bm25':
            values = log((n_docs - nq + .5) / (nq + .5)) + 1
        else:
            raise ValueError(f"Unknown version: {version}")

        if self.sparse_framework == 'torch':
            idf = torch.sparse_csr_tensor(indptr, col_indices, values, size=(1, self.n_tokens), device=self.device)
        else:
            idf = sp.csr_matrix((values, col_indices, indptr), shape=(1, self.n_tokens))

        return idf

    def fit(self, x, **kwargs):
        self.reset()
        dfs, cfs = self.chunk_df.transform(x, **kwargs)
        self.n_docs = len(dfs)
        for df, cf in zip(dfs, cfs):
            self.df.update(df)
            self.cf.update(cf)
        self.filter_tokens()

    def fit_transform(self, x):
        self.fit(x)
        self.transform(x)

    def add(self, x, **kwargs):
        dfs = self.chunk_df.transform(x, **kwargs)
        for df in dfs:
            self.df.update(df)
        self.n_docs += len(dfs)

    def fit_termination(self):
        self.filter_tokens(len(self.tfs))

    def filter_tokens(self):

        n = self.n_docs
        if self.min_df is not None:
            min_df = self.min_df
            if self.min_df < 1:
                min_df = int(self.min_df * n)
            self.df = {k: v for k, v in self.df.items() if v >= min_df}

        if self.max_df is not None:
            max_df = self.max_df
            if self.max_df < 1:
                max_df = int(self.max_df * n)
            self.df = {k: v for k, v in self.df.items() if v <= max_df}

        if self.max_features is not None:
            self.df = Counter(dict(sorted(self.df.items(), key=lambda x: x[1], reverse=True)[:self.max_features]))

        self.cf = {k: v for k, v in self.cf.items() if k in self.df}


class SparseSimilarity(BeamSimilarity):
    """
    The `SparseSimilarity` class is a processor that computes similarity between sparse vectors.

    Args:
        metric (str): The similarity metric to use. Possible values are 'cosine', 'prod', 'l2', and 'max'.
                      Default is 'cosine'.
        layout (str): The layout format of the sparse vectors. Possible values are 'coo' and 'csr'. Default is 'coo'.
        vec_size (int): The size of the vectors. Required if the layout is 'csr', otherwise optional.
        device (str): The device to use for computation. Default is None, which means using the default device.
        k (int): The number of nearest neighbors to search for. Default is 1.
        q (float): The quantile value to use for the 'quantile' metric. Default is 0.9.

    Methods:
        reset()
            Reset the state of the processor.

        sparse_tensor(r, c, v)
            Convert coordinate, row, column, and value data into a sparse tensor.

            Args:
                r (Tensor): The row indices.
                c (Tensor): The column indices.
                v (Tensor): The values.

            Returns:
                SparseTensor: The sparse tensor.

        index
            Get the current index tensor.

        scipy_to_row_col_val(x)
            Convert a sparse matrix in the scipy sparse format to row, column, and value data.

            Args:
                x (scipy.sparse.spmatrix): The sparse matrix.

            Returns:
                Tensor: The row indices.
                Tensor: The column indices.
                Tensor: The values.

        to_sparse(x)
            Convert input data to a sparse tensor.

            Args:
                x (Tensor, numpy.ndarray, scipy.sparse.spmatrix, dict, tuple): The input data.

            Returns:
                SparseTensor: The sparse tensor.

        add(x)
            Add a sparse vector to the index.

            Args:
                x (Tensor, numpy.ndarray, scipy.sparse.spmatrix, dict, tuple): The input sparse vector.

        search(x, k=None)
            Search for the nearest neighbors of a sparse vector.

            Args:
                x (SparseTensor, Tensor, numpy.ndarray, scipy.sparse.spmatrix, dict, tuple): The query sparse vector.
                k (int): The number of nearest neighbors to search for. If not specified, use the default value.

            Returns:
                Tensor: The distances to the nearest neighbors.
                Tensor: The indices of the nearest neighbors.
    """
    def __init__(self, *args, metric='cosine', layout='coo', vec_size=None, device=None, k=1, q=.9, **kwargs):

        super().__init__(*args, **kwargs)
        # possible similarity metrics: cosine, prod, l2, max
        self.metric = metric
        self.layout = layout
        self.device = beam_device(device)
        self.vec_size = vec_size
        self.state = {'index': None, 'chunks': []}
        self.k = k
        self.q = q

    def reset(self):
        self.state = {'index': None, 'chunks': []}

    def sparse_tensor(self, r, c, v,):
        device = self.device
        size = (r.max() + 1, self.vec_size)

        r, c, v = as_tensor([r, c, v], device=device)

        if self.layout == 'coo':
            return torch.sparse_coo_tensor(torch.stack([r, c]), v, size=size, device=device)

        if self.layout == 'csr':
            return torch.sparse_csr_tensor(r, c, v, size=size, device=device)

        raise ValueError(f"Unknown format: {self.layout}")

    @property
    def index(self):

        if len(self.state['chunks']):

            if self.state['index'] is None:
                chunks = self.state['chunks']
            else:
                chunks = [self.state['index']] + self.state['chunks']

            self.state['index'] = torch.cat(chunks)
            self.state['chunks'] = []

        return self.state['index']

    @staticmethod
    def scipy_to_row_col_val(x):

        r, c = x.nonzero()
        return r, c, x.data

    def to_sparse(self, x):

        x_type = check_type(x)

        if x_type.minor == 'scipy_sparse':
            r, c, v = self.scipy_to_row_col_val(x)
            x = self.sparse_tensor(r, c, v)

        elif x_type.minor in ['tensor', 'numpy']:

            if x_type.minor == 'numpy':
                x = as_tensor(x)

            if self.layout == 'coo':
                x = x.to_sparse_coo()
            elif self.layout == 'csr':
                x = x.to_sparse_csr()
            else:
                raise ValueError(f"Unknown format: {self.layout}")

        elif x_type.minor == 'dict':
            x = self.sparse_tensor(x['row'], x['col'], x['val'])

        elif x_type.minor == 'tuple':
            x = self.sparse_tensor(x[0], x[1], x[2])

        else:
            raise ValueError(f"Unsupported type: {x_type}")

        return x

    def add(self, x):

        x = self.to_sparse(x)
        self.state['chunks'].append(x)

    def search(self, x, k=None, **kwargs):

        if k is None:
            k = self.k

        x = self.to_sparse(x)

        if self.metric in ['cosine', 'l2', 'prod']:

            if self.layout == 'csr':
                x = x.to_dense()

            ab = self.index @ x.T

            if self.metric in ['l2', 'cosine']:

                a2 = (self.index * self.index).sum(dim=1, keepdim=True)
                b2 = (x * x).sum(dim=1, keepdim=True)

                if self.metric == 'cosine':

                    s = 1 / torch.sqrt(a2 @ b2.T).to_dense()
                    dist = - ab * s
                else:
                    dist = a2 + b2 - 2 * ab

            elif self.metric == 'prod':
                dist = -ab

            dist = dist.to_dense()

        elif self.metric in ['max', 'quantile']:
            x = x.to_dense()

            def metric(x):
                if self.metric == 'max':
                    return x.max()
                elif self.metric == 'quantile':
                    return x.quantile(self.q)
                else:
                    raise ValueError(f"Unknown metric: {self.metric}")

            dist = []
            for xi in x:
                d = self.index * xi.unsqueeze(0)
                i = d._indices()
                v = d._values()

                dist.append(as_tensor([metric(v[i[0] == j]) for j in range(len(self.index))]))

            dist = -torch.stack(dist, dim=1)

        topk = torch.topk(dist, k, dim=0, largest=False, sorted=True)

        return topk.values.T, topk.indices.T
