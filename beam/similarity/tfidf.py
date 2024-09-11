from typing import List, Union, Any

from ..type import Types
from ..utils import cached_property

from collections import Counter
import scipy.sparse as sp
import numpy as np

from ..importer import lazy_importer as lzi
if lzi.has('torch'):
    import torch
    default_sparse_framework = 'torch'
else:
    default_sparse_framework = 'scipy'

from ..data import BeamData
from ..transformer import Transformer
from ..utils import check_type, as_numpy
from .core import Similarities, BeamSimilarity


class ChunkTF(Transformer):

    def __init__(self, *args, sparse_framework=default_sparse_framework, device='cpu', preprocessor=None, **kwargs):
        self.preprocessor = preprocessor or TFIDF.default_preprocessor
        self.sparse_framework = sparse_framework
        self._device = device
        super().__init__(*args, **kwargs)

    @cached_property
    def device(self):
        from ..utils import beam_device
        return beam_device(self._device)

    @staticmethod
    def tf_tfidf_row(counts, idf=None, scheme='term_frequencies', sparse_framework='torch', norm='l2', k=0.5,
                     log_normalization=False):

        log1p = torch.log1p if sparse_framework == 'torch' else np.log1p
        tf = counts
        if scheme == 'counts_squared':
            tf = tf ** 2
        if scheme in ['term_frequencies', 'raw_counts', 'binary']:
            tf = tf / tf.sum()
        elif scheme == 'double_normalization':
            tf = tf + (1 - k) * tf / tf.max()
        if log_normalization:
            tf = log1p(tf)
        if idf is not None:
            tfidf = tf * idf
        else:
            tfidf = tf

        if norm == 'l2' and sparse_framework == 'torch':
            tfidf = tfidf / torch.norm(tfidf, p=2)
        elif norm == 'l2':
            tfidf = tfidf / np.linalg.norm(tfidf, ord=2)
        elif norm == 'l1' and sparse_framework == 'torch':
            tfidf = tfidf / torch.norm(tfidf, p=1)
        elif norm == 'l1':
            tfidf = tfidf / np.linalg.norm(tfidf, ord=1)

        if scheme == 'raw_counts':
            tf = counts
        elif scheme == 'binary':
            tf = torch.ones_like(counts)

        return tf, tfidf

    def transform_callback(self, x, tokens=None, scheme='term_frequencies', max_token=None, k=0.5, idf=None,
                           norm='l2', log_normalization=False, **kwargs):
        """

        Args:
            k:
            x:
            tokens:
            scheme: can be ['binary', 'term_frequencies', 'log_normalization', 'raw_count', 'double_normalization']
            max_token:
            **kwargs:

        Returns:

        """
        if max_token is None:
            max_token = np.max(list(tokens))

        tfidf = []
        tf = []
        ind_ptrs = [0]
        cols = []

        for xi in x:
            xi = self.preprocessor(xi)
            if scheme == 'binary':
                xi = list(set(xi))
            c = Counter(xi)
            c = Counter({k: v for k, v in c.items() if k in tokens})

            ind_ptrs.append(len(c))
            if self.sparse_framework == 'torch':
                ind = torch.tensor(list(c.keys()), dtype=torch.int64, device=self.device)
                counts = torch.tensor(list(c.values()), dtype=torch.float32, device=self.device)
                tfi, tfidfi = self.tf_tfidf_row(counts, idf=idf[ind], scheme=scheme, sparse_framework=self.sparse_framework,
                                            norm=norm, k=k, log_normalization=log_normalization)

                cols.append(ind)
                tf.append(tfi)
                tfidf.append(tfidfi)

            else:
                ind = np.array(list(c.keys()), dtype=int)
                counts = np.array(list(c.values()), dtype=float)
                tfi, tfidfi = self.tf_tfidf_row(counts, idf=idf[ind], scheme=scheme,
                                                sparse_framework=self.sparse_framework,
                                                norm=norm, k=k, log_normalization=log_normalization)

                cols.append(ind)
                tf.append(tfi)
                tfidf.append(tfidfi)

        if self.sparse_framework == 'torch':
            ind_ptrs = torch.cumsum(torch.tensor(ind_ptrs, dtype=torch.int64, device=self.device), dim=0)
            tf = torch.sparse_csr_tensor(ind_ptrs, torch.cat(cols), torch.cat(tf), device=self.device,
                                        size=(len(x), max_token + 1))
            tfidf = torch.sparse_csr_tensor(ind_ptrs, torch.cat(cols), torch.cat(tfidf), device=self.device,
                                         size=(len(x), max_token + 1))
        else:
            ind_ptrs = np.cumsum(ind_ptrs)
            tf = sp.csr_matrix((np.concatenate(tf), np.concatenate(cols), ind_ptrs),
                              shape=(len(x), max_token + 1))
            tfidf = sp.csr_matrix((np.concatenate(tfidf), np.concatenate(cols), ind_ptrs),
                               shape=(len(x), max_token + 1))

        return tf, tfidf


class ChunkDF(Transformer):

        def __init__(self, *args, preprocessor=None, **kwargs):
            self.preprocessor = preprocessor or TFIDF.default_preprocessor
            super().__init__(*args, **kwargs)

        def transform_callback(self, x, _key=None, _is_chunk=False, _fit=False, path=None, **kwargs):
            y = Counter()
            y_sum = Counter()
            for xi in x:
                xi = self.preprocessor(xi)
                y.update(set(xi))
                y_sum.update(xi)
            return y, y_sum


class TFIDF(BeamSimilarity):

    def __init__(self, *args, preprocessor=None, max_df=1.0, min_df=1, max_features=None, use_idf=True,
                 smooth_idf=True, sublinear_tf=False, n_workers=0, mp_method='joblib', chunksize=None,
                 use_dill=False, n_chunks=None, sparse_framework='torch', device='cpu', norm='l2',
                 metric='bm25', bm25_k1=1.5, bm25_b=0.75, bm25_epsilon=0.25, **kwargs):

        super().__init__(*args, min_df=min_df, max_df=max_df, max_features=max_features, use_idf=use_idf,
                         sparse_framework=sparse_framework, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf,
                         chunksize=chunksize, n_chunks=n_chunks, device=device, metric=metric,
                         bm25_k1=bm25_k1, bm25_b=bm25_b, bm25_epsilon=bm25_epsilon, norm=norm,
                         n_workers=n_workers, mp_method=mp_method,
                         **kwargs)
        self.df = None
        self.cf = None  # corpus frequency
        self.n_docs = None
        self.tf = None
        self.index = None
        self._is_trained = None
        self.reset()

        self.preprocessor = preprocessor or TFIDF.default_preprocessor

        self.min_df = self.get_hparam('min_df', min_df)
        self.max_df = self.get_hparam('max_df', max_df)
        self.max_features = self.get_hparam('max_features', max_features)
        self.use_idf = self.get_hparam('use_idf', use_idf)
        self.smooth_idf = self.get_hparam('smooth_idf', smooth_idf)
        self.sublinear_tf = self.get_hparam('sublinear_tf', sublinear_tf)
        self.sparse_framework = self.get_hparam('sparse_framework', sparse_framework)
        self.norm = self.get_hparam('norm', norm)
        self.bm25_k1 = self.get_hparam('bm25_k1', bm25_k1)
        self.bm25_b = self.get_hparam('bm25_b', bm25_b)
        self.bm25_epsilon = self.get_hparam('bm25_epsilon', bm25_epsilon)

        # we choose the csr layout for the sparse matrix
        # according to chatgpt it has some advantages over coo:
        # see https://chat.openai.com/share/9028c9f3-9695-4914-a15c-89902efa8837

        self.device = self.get_hparam('device', device)

        self.n_workers = self.get_hparam('n_workers', n_workers)
        self.n_chunks = self.get_hparam('n_chunks', n_chunks)
        self.chunksize = self.get_hparam('chunksize', chunksize)
        self.mp_method = self.get_hparam('mp_method', mp_method)
        self.use_dill = self.get_hparam('use_dill', use_dill)

    @cached_property
    def preprocessor_transformer(self):
        return Transformer(func=self.preprocessor, n_workers=self.n_workers, n_chunks=self.n_chunks,
                            chunksize=self.chunksize, mp_method=self.mp_method)

    @cached_property
    def chunk_tf(self):
        return ChunkTF(n_workers=self.n_workers, n_chunks=self.n_chunks, chunksize=self.chunksize,
                       mp_method=self.mp_method, use_dill=self.use_dill,
                       squeeze=False, reduce=False, sparse_framework=self.sparse_framework, device=self.device,
                       preprocessor=self.preprocessor)

    @cached_property
    def chunk_df(self):
        return ChunkDF(n_workers=self.n_workers, n_chunks=self.n_chunks, chunksize=self.chunksize,
                       mp_method=self.mp_method, use_dill=self.use_dill,
                       squeeze=False, reduce=False, preprocessor=self.preprocessor)

    def preprocess(self, x):
        if self.chunksize is not None and len(x) <= self.chunksize:
            return self.preprocessor(x)
        if self.n_chunks is not None and self.n_chunks < 2:
            return self.preprocessor(x)
        return self.preprocessor_transformer.transform(x)

    @staticmethod
    def default_preprocessor(x):

        x_type = check_type(x)

        if x_type.minor == 'torch':
            x = as_numpy(x).tolist()
        elif x_type.minor == Types.numpy:
            x = x.tolist()
        else:
            x = list(x)

        return x

    def vstack_csr_tensors(self, x):
        crow_indices = []
        col_indices = []
        values = []
        n = 0
        for xi in x:
            indptr = xi.crow_indices()
            if len(crow_indices) > 0:
                crow_indices.append(indptr[1:] + crow_indices[-1][-1])
            else:
                crow_indices.append(indptr)
            col_indices.append(xi.col_indices())
            values.append(xi.values())
            n += len(xi)
            
        return torch.sparse_csr_tensor(torch.cat(crow_indices), torch.cat(col_indices), torch.cat(values),
                                       size=(n, xi.shape[-1]), device=self.device)

    def tf_and_tfidf(self, x, tokens=None, scheme=None, idf=None, norm=None, k=.5, log_normalization=None, **kwargs):

        idf = self.idf if idf is None else idf
        norm = self.norm if norm is None else norm
        log_normalization = self.sublinear_tf if log_normalization is None else log_normalization

        if tokens is None:
            tokens = self.tokens
            max_token = self.max_token
        else:
            max_token = max(tokens)

        chunks = self.chunk_tf.transform(x, tokens=tokens, max_token=max_token, scheme=scheme, idf=idf, k=k,
                                         norm=norm, log_normalization=log_normalization, **kwargs)

        if self.sparse_framework == 'torch':
            tf = self.vstack_csr_tensors([c[0] for c in chunks])
            tfidf = self.vstack_csr_tensors([c[1] for c in chunks])
        else:
            tf = sp.vstack([c[0] for c in chunks])
            tfidf = sp.vstack([c[1] for c in chunks])

        return tf, tfidf

    def bm25(self, q, k1=1.5, b=0.75, epsilon=.25, **kwargs):

        idf = self.idf_bm25(epsilon=epsilon)
        _, q_tfidf = self.tf_and_tfidf(q, scheme='counts_times_idf', idf=idf, norm='none', log_normalization=False,
                                       **kwargs)

        if self.sparse_framework == 'torch':
            len_norm_values = (1 - b) + (b / self.avg_doc_len) * self.doc_len_sparse.values()
            bm25_tf_values = self.tf.values() * (k1 + 1) / (self.tf.values() + k1 * len_norm_values)
            bm25_tf = torch.sparse_csr_tensor(self.tf.crow_indices(), self.tf.col_indices(), bm25_tf_values,
                                              size=self.tf.shape, device=self.device)
        else:
            len_norm_values = (1 - b) + (b / self.avg_doc_len) * self.doc_len_sparse.data
            bm25_tf_values = self.tf.data * (k1 + 1) / (self.tf.data + k1 * len_norm_values)
            bm25_tf = sp.csr_matrix((bm25_tf_values, self.tf.indices, self.tf.indptr), shape=self.tf.shape)

        if self.sparse_framework == 'torch':
            scores = torch.matmul(bm25_tf, q_tfidf.to_dense().T).T
        else:
            q_idf = q_tfidf.multiply(idf)
            scores = q_idf @ bm25_tf.T

        return scores

    def as_container(self, x):
        x_type = check_type(x)
        if x_type.major == Types.scalar:
            x = [x]
        else:
            x = list(x)
        return x

    def transform(self, x: Union[List, List[List], BeamData], index: Union[None, Any] = None,
                  add_to_index: bool = False, **kwargs):

        x, index = self.extract_data_and_index(x, index, convert_to=None)
        if add_to_index or self.index is None:
            self.add_index(x, index)

        x = self.as_container(x)

        tf, tfidf = self.tf_and_tfidf(x, scheme='raw_counts', **kwargs)
        if self.tf is None:
            self.tf = tf
        elif add_to_index:
            if self.sparse_framework == 'torch':
                self.tf = self.vstack_csr_tensors([self.tf, tf])
            else:
                self.tf = sp.vstack([self.tf, tf])

        return tfidf

    def reset(self):
        self.df = Counter()
        self.cf = Counter()
        self.n_docs = 0
        self.tf = None
        self.index = np.array([])
        self._is_trained = False
        self.clear_cache('idf', 'tokens', 'n_tokens', 'avg_doc_len', 'idf_bm25', 'doc_len', 'doc_len_sparse',
                         'max_token')

    @cached_property
    def tokens(self):
        """Build a mapping from tokens to indices based on filtered tokens."""
        return set(self.df.keys())

    @cached_property
    def avg_doc_len(self):
        return sum(self.cf.values()) / self.n_docs

    @cached_property
    def n_tokens(self):
        l = list(self.tokens)
        if len(l) == 0:
            return 0
        return max(l) + 1

    @cached_property
    def idf(self):

        if self.use_idf:
            if self.smooth_idf:
                idf_version = 'smooth'
            else:
                idf_version = 'standard'
        else:
            idf_version = 'unary'

        return self.calculate_idf(scheme=idf_version)

    def idf_bm25(self, epsilon=.25):
        return self.calculate_idf(scheme='bm25', epsilon=epsilon)

    @cached_property
    def max_token(self):
        return max(list(self.tokens))

    @cached_property
    def doc_len(self):
        if self.sparse_framework == 'torch':
            doc_lengths = self.tf.sum(dim=1, keepdim=True).to_dense().squeeze(-1)
        else:
            doc_lengths = np.array(self.tf.sum(axis=1))

        return doc_lengths

    @cached_property
    def doc_len_sparse(self):
        if self.sparse_framework == 'torch':
            repeats = self.tf.crow_indices().diff()
            values = torch.repeat_interleave(self.doc_len, repeats, dim=0)
            return torch.sparse_csr_tensor(self.tf.crow_indices(), self.tf.col_indices(), values,
                                           size=self.tf.shape, device=self.device)
        else:
            repeats = np.diff(self.tf.indptr)
            values = np.repeat(self.doc_len, repeats)
            return sp.csr_matrix((values, self.tf.indices, self.tf.indptr), shape=self.tf.shape)

    def calculate_idf(self, scheme='standard', epsilon=.25, okapi=True):
        """Calculate the inverse document frequency (IDF) vector.
        version: str, default='standard' [standard, smooth, unary, bm25]
        """
        n_docs = self.n_docs
        keys = list(self.df.keys())

        if self.sparse_framework == 'torch':
            col_indices = torch.tensor(keys, dtype=torch.int64, device=self.device)
            log = torch.log
            zeros = torch.zeros
            framework_kwargs = {'device': self.device, 'dtype': torch.float32}
            array = torch.tensor
        else:
            col_indices = np.array(keys)
            log = np.log
            zeros = np.zeros
            framework_kwargs = {}
            array = np.array

        vals = array(list(self.df.values()), **framework_kwargs)
        nq = zeros(col_indices.max()+1, **framework_kwargs)

        if scheme == 'unary':
            nq[col_indices] = 1
        else:
            nq[col_indices] = vals

        if scheme == 'standard':
            idf = log(n_docs / nq)
        elif scheme == 'smooth':
            idf = log(n_docs / (nq + 1)) + 1
        elif scheme == 'bm25':
            idf = log((n_docs - nq + .5) / (nq + .5) + (1 - int(okapi)))
            idf[idf < 0] = epsilon * idf.mean()
        else:
            raise ValueError(f"Unknown version: {scheme}")

        return idf

    def fit(self, x=None, **kwargs):
        if x is not None:
            self.reset()
            self.add(x, **kwargs)
        self.filter_tokens()
        self._is_trained = True

    def fit_transform(self, x, index=None, **kwargs):

        self.fit(x, **kwargs)
        return self.transform(x, index=index, add_to_index=True)

    def add(self, x, **kwargs):
        self.n_docs += len(x)
        chunks = self.chunk_df.transform(x, **kwargs)
        for df, cf in chunks:
            self.df.update(df)
            self.cf.update(cf)

    def filter_tokens(self):

        n = self.n_docs
        if self.min_df is not None:
            min_df = self.min_df
            if self.min_df < 1:
                min_df = int(self.min_df * n)
            self.df = {k: v for k, v in self.df.items() if v >= min_df}

        if self.max_df is not None:
            max_df = self.max_df
            if self.max_df <= 1:
                max_df = int(self.max_df * n)
            self.df = {k: v for k, v in self.df.items() if v <= max_df}

        if self.max_features is not None:
            self.df = Counter(dict(sorted(self.df.items(), key=lambda x: x[1], reverse=True)[:self.max_features]))

        self.cf = {k: v for k, v in self.cf.items() if k in self.df}

    def train(self, x):
        self.fit(x)

    def search(self, q, k=1, **kwargs) -> Similarities:

        q = self.as_container(q)

        if self.metric == 'bm25':
            scores = self.bm25(q, **kwargs)
            if self.sparse_framework == 'torch':
                topk = torch.topk(scores, k=k, dim=1)
                I = topk.indices
                D = topk.values
            else:
                scores = scores.toarray()
                I = np.argsort(-scores, axis=1)[:, :k]
                D = np.take_along_axis(scores, I, axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return Similarities(index=self.get_index(I), distance=D, metric=self.metric, model='tfidf')

    @classmethod
    @property
    def special_state_attributes(cls):
        return super(TFIDF, cls).special_state_attributes.union(['df', 'cf', 'n_docs', 'tf', 'index', 'idf'])

    @classmethod
    @property
    def excluded_attributes(cls):
        return super(TFIDF, cls).excluded_attributes.union(['preprocessor_transformer', 'chunk_tf', 'chunk_df',
                                                            'preprocessor'])

