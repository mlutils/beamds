import torch

from .core import BeamSimilarity
from ..core import Processor
from ..utils import check_type, as_tensor, beam_device
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
