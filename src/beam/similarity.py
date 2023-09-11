import torch

from .processor import  Processor
from .utils import check_type
import scipy


class SparseSimilarity(Processor):

    def __init__(self, *args, similarity='cosine', format='coo', vec_size=None, device=None, **kwars):

        super().__init__(*args, **kwars)
        # possible similarity metrics: cosine, prod, l2, max
        self.similarity = similarity
        self.format = format
        self.device = device
        self.vec_size = vec_size
        self.state = {'index': None, 'chunks': []}

    def sparse_tensor(self, r, c, v,):
        device = self.device
        size = (r.max(), self.vec_size)

        if self.format == 'coo':
            return torch.sparse_coo_tensor(torch.stack([r, c]), v, size=size, device=device)

        if self.format == 'csr':
            return torch.sparse_csr_tensor(torch.cat(r), torch.cat(c), torch.cat(v), size=size, device=device)

        raise ValueError(f"Unknown format: {self.format}")

    @property
    def index(self):

        if len(self.state['chunks']):
            if self.state['index'] is None:
                self.state['index'] = torch.cat(self.state['chunks'])
            else:
                self.state['index'] = torch.cat([self.state['index']] + self.state['chunks'])
            self.state['chunks'] = []

        return self.state['index']

    @staticmethod
    def scipy_csr_to_row_col_val(x):
        row_indices = []
        for i in range(x.shape[0]):
            row_indices.extend([i] * (x.indptr[i + 1] - x.indptr[i]))
        col_indices = x.indices

        return row_indices, col_indices, x.data

    @staticmethod
    def scipy_coo_to_row_col_val(x):
        return x.row, x.col, x.data

    def add(self, x):

        x_type = check_type(x)

        if x_type.minor == 'scipy_sparse':
            if type(x) is scipy.sparse.csr_matrix:
                r, c, v = self.scipy_csr_to_row_col_val(x)
            else:
                r, c, v = self.scipy_coo_to_row_col_val(x)
            t = self.sparse_tensor(r, c, v)

        elif x_type.minor == 'torch':
            if self.format == 'coo':
                t = x.to_sparse_coo()
            elif self.format == 'csr':
                t = x.to_sparse_csr()
            else:
                raise ValueError(f"Unknown format: {self.format}")

        elif x_type.minor == 'dict':
            t = self.sparse_tensor(x['row'], x['col'], x['val'])

        elif x_type.minor == 'tuple':
            t = self.sparse_tensor(x[0], x[1], x[2])

        else:
            raise ValueError(f"Unsupported type: {x_type}")

        self.state['chunks'].append(t)

    def search(self, x, k=1, **kwargs):

        if self.similarity in ['cosine', 'l2', 'prod']:

            if self.format == 'csr':
                x = x.to_dense()

            ab = torch.matmul(self.index, x.T)

            if self.similarity in ['l2', 'cosine']:

                a = torch.norm(self.index, dim=1).unsqueeze(1)
                b = torch.norm(x, dim=1).unsqueeze(0)

                if self.similarity == 'cosine':
                    dist = - ab / (a * b)
                else:
                    dist = a ** 2 + b ** 2 - 2 * ab

            elif self.similarity == 'prod':
                dist = -ab

        topk = torch.topk(dist.to_dense(), 5, dim=0, largest=False, sorted=True)

        return topk.values, topk.indices