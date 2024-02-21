from typing import List, Union

import numpy as np
import pandas as pd
import torch

from .. import as_numpy, BeamData
from ..core import Processor
from ..transformer import Transformer
from ..utils import check_type, as_tensor, beam_device, tqdm_beam as tqdm, lazy_property
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

    def transform_callback(self, x, tokens=None, binary=False, max_token=None, **kwargs):

        if max_token is None:
            max_token = np.max(list(tokens))

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

                values.append(torch.tensor(list(c.values()), dtype=torch.float32, device=self.device))
                cols.append(torch.tensor(list(c.keys()), dtype=torch.int64, device=self.device))

            else:
                cols.append(np.array(list(c.keys())))
                values.append(np.array(list(c.values())))

        if self.sparse_framework == 'torch':
            ind_ptrs = torch.cumsum(torch.tensor(ind_ptrs, dtype=torch.int64, device=self.device), dim=0)
            y = torch.sparse_csr_tensor(ind_ptrs, torch.cat(cols), torch.cat(values), device=self.device,
                                        size=(len(x), max_token + 1))
        else:
            ind_ptrs = np.cumsum(ind_ptrs)
            y = sp.csr_matrix((np.concatenate(values), np.concatenate(cols), ind_ptrs),
                              shape=(len(x), max_token + 1))

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
        self.index = None
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
                                squeeze=False, reduce=False, sparse_framework=self.sparse_framework, device=self.device,
                                preprocessor=self.preprocessor)
        self.chunk_df = ChunkDF(n_workers=self.n_workers, n_chunks=n_chunks, chunksize=chunksize, mp_method=mp_method,
                                squeeze=False, reduce=False, preprocessor=self.preprocessor)

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

    def term_frequencies(self, x, tokens=None, binary=False, sublinear_tf=None, **kwargs):
        if tokens is None:
            tokens = self.tokens
        if sublinear_tf is None:
            sublinear_tf = self.sublinear_tf
        if binary:
            sublinear_tf = False
        chunks = self.chunk_tf.transform(x, tokens=tokens, max_token=self.max_token, binary=binary, **kwargs)
        if self.sparse_framework == 'torch':
            vectors = self.vstack_csr_tensors(chunks)
            vectors = torch.log1p(vectors) if sublinear_tf else vectors
        else:
            vectors = sp.vstack(chunks)
            vectors = sp.csr_matrix((np.log1p(vectors.data),
                                     vectors.indices, vectors.indptr), shape=vectors.shape) if sublinear_tf else vectors
        return vectors

    def bm25(self, q, k1=1.5, b=0.75, epsilon=.25):
        q_tf = self.term_frequencies(q, binary=False)
        if self.sparse_framework == 'torch':
            len_norm_values = (1 - b) + (b / self.avg_doc_len) * self.doc_len_sparse.values()
            bm25_tf_values = self.tf.values() * (k1 + 1) / (self.tf.values() + k1 * len_norm_values)
            bm25_tf = torch.sparse_csr_tensor(self.tf.crow_indices(), self.tf.col_indices(), bm25_tf_values,
                                              size=self.tf.shape, device=self.device)
        else:
            len_norm_values = (1 - b) + (b / self.avg_doc_len) * self.doc_len_sparse.data
            bm25_tf_values = self.tf.data * (k1 + 1) / (self.tf.data + k1 * len_norm_values)
            bm25_tf = sp.csr_matrix((bm25_tf_values, self.tf.indices, self.tf.indptr), shape=self.tf.shape)

        idf = self.idf_bm25(epsilon=epsilon)

        if self.sparse_framework == 'torch':
            idf = self.vstack_csr_tensors([idf] * len(q_tf))
        else:
            idf = sp.vstack([idf] * q_tf.shape[0])

        if self.sparse_framework == 'torch':
            q_idf = q_tf * idf
            scores = torch.matmul(bm25_tf, q_idf.to_dense().T).T
        else:
            q_idf = q_tf.multiply(idf)
            scores = q_idf @ bm25_tf.T

        return scores

    def transform(self, x: Union[List, List[List], BeamData], index=Union[None, List[int]]):

        if self.index is None:
            if index is None:
                index = pd.Series(range(len(x)))
            else:
                index = pd.Series(index)
            self.index = index
        else:
            if index is None:
                index = pd.Series(range(len(x))) + self.index.max() + 1
            else:
                index = pd.Series(index)
            self.index = pd.concat([self.index, index])

        x_type = check_type(x)
        if not x_type.major == 'container':
            x = [x]

        tfs = self.term_frequencies(x)
        if self.tf is None:
            self.tf = tfs
        else:
            if self.sparse_framework == 'torch':
                self.tf = self.vstack_csr_tensors([self.tf, tfs])
            else:
                self.tf = sp.vstack([self.tf, tfs])

        if self.sparse_framework == 'torch':
            idf = self.vstack_csr_tensors([self.idf] * len(tfs))
            tfidf = tfs * idf
        else:
            idf = sp.vstack([self.idf] * tfs.shape[0])
            tfidf = tfs.multiply(idf)

        return tfidf

    def reset(self):
        self.df = Counter()
        self.cf = Counter()
        self.n_docs = 0
        self.tf = None
        self.clear_cache('idf', 'tokens', 'n_tokens', 'avg_doc_len', 'idf_bm25', 'doc_len', 'doc_len_sparse',
                         'max_token')

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

    def idf_bm25(self, epsilon=.25):
        return self.calculate_idf(version='bm25', epsilon=epsilon)

    @lazy_property
    def max_token(self):
        return max(list(self.df.keys()))

    @lazy_property
    def doc_len(self):
        if self.sparse_framework == 'torch':
            doc_lengths = self.tf.sum(dim=1, keepdim=True).to_dense().squeeze(-1)
        else:
            doc_lengths = np.array(self.tf.sum(axis=1))

        return doc_lengths

    @lazy_property
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

    def calculate_idf(self, version='standard', epsilon=.25, okapi=True):
        """Calculate the inverse document frequency (IDF) vector.
        version: str, default='standard' [standard, smooth, unary, bm25]
        """
        n_docs = self.n_docs
        indptr = [0, len(self.df)]

        if self.sparse_framework == 'torch':
            col_indices = torch.tensor(list(self.df.keys()), dtype=torch.int64, device=self.device)
            log = torch.log
            ones_like = torch.ones_like
            framework_kwargs = {'device': self.device, 'dtype': torch.float32}
            array = torch.tensor
            clamp = torch.clamp
        else:
            col_indices = np.array(list(self.df.keys()))
            log = np.log
            ones_like = np.ones_like
            framework_kwargs = {}
            array = np.array
            clamp = np.clip

        nq = array(list(self.df.values()), **framework_kwargs)
        if version == 'standard':
            values = log(n_docs / nq)
        elif version == 'smooth':
            values = log(n_docs / (nq + 1)) + 1
        elif version == 'unary':
            values = ones_like(col_indices, **framework_kwargs)
        elif version == 'bm25':
            idf = log((n_docs - nq + .5) / (nq + .5) + (1 - int(okapi)))
            values = clamp(idf, epsilon * idf.mean())
        else:
            raise ValueError(f"Unknown version: {version}")

        if self.sparse_framework == 'torch':
            idf = torch.sparse_csr_tensor(indptr, col_indices, values, size=(1, self.max_token+1), device=self.device)
        else:
            idf = sp.csr_matrix((values, col_indices, indptr), shape=(1, self.max_token+1))

        return idf

    def fit(self, x, **kwargs):
        self.reset()
        self.n_docs = len(x)
        chunks = self.chunk_df.transform(x, **kwargs)
        for df, cf in chunks:
            self.df.update(df)
            self.cf.update(cf)
        self.filter_tokens()

    def fit_transform(self, x, index=Union[None, List[int]]):

        self.fit(x)
        return self.transform(x, index=index)

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
