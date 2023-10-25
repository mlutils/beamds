import torch

from .processor import  Processor
from .utils import check_type, as_tensor, beam_device
import scipy
from collections import Counter


class TFIDF(Processor):

    def __init__(self, *args, separator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.separator = separator
        self.state = {'idf_counter': Counter()}

    def add_single(self, x, x_type=None):

        if x_type is None:
            x_type = check_type(x)

        if x_type.major == 'scalar' and x_type.element == 'str':
            x = x.split(self.separator)
        elif x_type.minor == 'dict':
            x = x.keys()

        self.state['idf_counter'].update(set(x))

    def add(self, x):

        x_type = check_type(x)
        if not (x_type.major == 'container' and x_type.minor != 'dict'):
            self.add_single(x, x_type)

        else:
            for xi in x:
                self.add_single(xi)

    def train(self):
        pass

    def transform(self, x):

        return x * self.idf


class SparseSimilarity(Processor):

    def __init__(self, *args, similarity='cosine', layout='coo', vec_size=None, device=None, k=1, **kwars):

        super().__init__(*args, **kwars)
        # possible similarity metrics: cosine, prod, l2, max
        self.similarity = similarity
        self.layout = layout
        self.device = beam_device(device)
        self.vec_size = vec_size
        self.state = {'index': None, 'chunks': []}
        self.k = k

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

        if self.similarity in ['cosine', 'l2', 'prod']:

            if self.layout == 'csr':
                x = x.to_dense()

            ab = self.index @ x.T

            if self.similarity in ['l2', 'cosine']:

                a2 = (self.index * self.index).sum(dim=1, keepdim=True)
                b2 = (x * x).sum(dim=1, keepdim=True)

                if self.similarity == 'cosine':

                    s = 1 / torch.sqrt(a2 @ b2.T).to_dense()
                    dist = - ab * s
                else:
                    dist = a2 + b2 - 2 * ab

            elif self.similarity == 'prod':
                dist = -ab

        topk = torch.topk(dist.to_dense(), k, dim=0, largest=False, sorted=True)

        return topk.values.T, topk.indices.T
