
import torch

from .core import BeamSimilarity, Similarities
from ..utils import check_type, as_tensor, beam_device
from ..type import Types


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
    def __init__(self, *args, metric='cosine', layout='coo', vec_size=None, device=None, quantile=.9, **kwargs):

        super().__init__(*args, metric=metric, layout=layout, vec_size=vec_size, device=device, quantile=quantile,
                         **kwargs)
        # possible similarity metrics: cosine, prod, l2, max
        self.metric = self.get_hparam('metric', metric)
        self.layout = self.get_hparam('layout', layout)
        self.device = beam_device(self.get_hparam('device', device))
        self.vec_size = vec_size
        self.index = None
        self.vectors = None
        self.quantile = self.get_hparam('quantile', quantile)

    def reset(self):
        self.index = None
        self.vectors = None

    def sparse_tensor(self, r, c, v,):
        device = self.device
        size = (r.max() + 1, self.vec_size)

        r, c, v = as_tensor([r, c, v], device=device)

        if self.layout == 'coo':
            return torch.sparse_coo_tensor(torch.stack([r, c]), v, size=size, device=device)

        if self.layout == 'csr':
            return torch.sparse_csr_tensor(r, c, v, size=size, device=device)

        raise ValueError(f"Unknown format: {self.layout}")

    @staticmethod
    def scipy_to_row_col_val(x):

        r, c = x.nonzero()
        return r, c, x.data

    def to_sparse(self, x):

        x_type = check_type(x)

        if x_type.minor == Types.scipy_sparse:
            r, c, v = self.scipy_to_row_col_val(x)
            x = self.sparse_tensor(r, c, v)

        elif x_type.minor in [Types.tensor, Types.numpy]:

            if x_type.minor == Types.numpy:
                x = as_tensor(x)

            if self.layout == 'coo':
                x = x.to_sparse_coo()
            elif self.layout == 'csr':
                x = x.to_sparse_csr()
            else:
                raise ValueError(f"Unknown format: {self.layout}")

        elif x_type.minor == Types.dict:
            x = self.sparse_tensor(x['row'], x['col'], x['val'])

        elif x_type.minor == Types.tuple:
            x = self.sparse_tensor(x[0], x[1], x[2])

        else:
            raise ValueError(f"Unsupported type: {x_type}")

        return x

    def add(self, x, index=None, **kwargs):

        x = self.to_sparse(x)
        if self.vectors is None:
            self.vectors = x
        else:
            self.vectors = torch.cat([self.vectors, x])

        if index is not None:
            if self.index is None:
                self.index = index
            else:
                self.index = torch.cat([self.index, index])
        else:
            if self.index is None:
                self.index = torch.arange(len(x), device=self.device)
            else:
                index = torch.arange(len(x), device=self.device) + self.index.max() + 1
                self.index = torch.cat([self.index, index])

    def train(self, x=None):
        raise NotImplementedError

    def search(self, x, k=1, **kwargs) -> Similarities:

        x = self.to_sparse(x)

        if self.metric in ['cosine', 'l2', 'prod']:

            if self.layout == 'csr':
                x = x.to_dense()

            ab = self.vectors @ x.T

            if self.metric in ['l2', 'cosine']:

                a2 = (self.vectors * self.vectors).sum(dim=1, keepdim=True)
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
                    return x.quantile(self.quantile)
                else:
                    raise ValueError(f"Unknown metric: {self.metric}")

            dist = []
            for xi in x:
                d = self.vectors * xi.unsqueeze(0)
                i = d._indices()
                v = d._values()

                dist.append(as_tensor([metric(v[i[0] == j]) for j in range(len(self.index))]))

            dist = -torch.stack(dist, dim=1)

        topk = torch.topk(dist, k, dim=0, largest=False, sorted=True)
        I = topk.indices.T
        D = topk.values.T

        return Similarities(index=self.index[I], distance=D, metric=self.metric, model='sparse')
