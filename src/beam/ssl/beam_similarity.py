import math
from collections import namedtuple
import numpy as np

from ..logger import beam_logger as logger
from ..utils import pretty_format_number

# working with faiss and torch
Similarities = namedtuple("Similarities", "index distance")


class BeamSimilarity(object):

    def __init__(self, index=None, d=None, expected_population=int(1e6),
                 metric='l2', training_device='cpu', inference_device='cpu', ram_footprint=2**8*int(1e9),
                 gpu_footprint=24*int(1e9), exact=False, nlists=None, M=None,
                 reducer='umap'):

        '''
        To Choose an index, follow https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        @param d:
        @param expected_population:
        @param metric:
        @param ram_size:
        @param gpu_size:
        @param exact_results:
        @param reducer:
        '''

        metrics = {'l2': faiss.METRIC_L2, 'l1': faiss.METRIC_L1, 'linf': faiss.METRIC_Linf,
                   'cosine': faiss.METRIC_INNER_PRODUCT, 'ip': faiss.METRIC_INNER_PRODUCT,
                   'js': faiss.METRIC_JensenShannon}
        metric = metrics[metric]
        self.normalize = False
        if metric == 'cosine':
            self.normalize = True

        # choosing nlists: https://github.com/facebookresearch/faiss/issues/112,
        #  https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        if nlists is None:
            if expected_population <= int(1e6):
                # You will need between 30*K and 256*K vectors for training (the more the better)
                nlists = int(8 * math.sqrt(expected_population))
            elif expected_population > int(1e6) and expected_population <= int(1e7):
                nlists = 2 ** 16
            elif expected_population > int(1e7) and expected_population <= int(1e8):
                nlists = 2 ** 18
            else:
                nlists = 2 ** 20

        if index is None:
            if inference_device == 'cpu':

                if exact:
                    logger.info(f"Using Flat Index. Expected RAM footprint is "
                                f"{pretty_format_number(4 * d * expected_population / int(1e6))} MB")
                    index = faiss.IndexFlat(d, metric)
                else:
                    if M is None:
                        M = 2 ** np.arange(2, 7)[::-1]
                        footprints = (d * 4 + M * 8) * expected_population
                        M_ind = np.where(footprints < ram_footprint)[0]
                        if len(M_ind):
                            M = int(M[M_ind[0]])
                    if M is not None:
                        logger.info(f"Using HNSW{M}. Expected RAM footprint is "
                                    f"{pretty_format_number(footprints[M_ind[0]] / int(1e6))} MB")
                        index = faiss.IndexHNSWFlat(d, M, metric)
                    else:
                        logger.info(f"Using OPQ16_64,IVF{nlists},PQ8 Index")
                        index = faiss.index_factory(d, f'OPQ16_64,IVF{nlists},PQ8')

            else:

                res = faiss.StandardGpuResources()
                if exact:
                    config = faiss.GpuIndexFlatConfig()
                    config.device = inference_device
                    logger.info(f"Using GPUFlat Index. Expected GPU-RAM footprint is "
                                f"{pretty_format_number(4 * d * expected_population / int(1e6))} MB")

                    index = faiss.GpuIndexFlat(res, d, metric, config)
                else:

                    if (4 * d + 8) * expected_population <= gpu_footprint:
                        logger.info(f"Using GPUIndexIVFFlat Index. Expected GPU-RAM footprint is "
                                    f"{pretty_format_number((4 * d + 8) * expected_population / int(1e6))} MB")
                        config = faiss.GpuIndexIVFFlatConfig()
                        config.device = inference_device
                        index = faiss.GpuIndexIVFFlat(res, d,  nlists, faiss.METRIC_L2, config)
                    else:

                        if M is None:
                            M = 2 ** np.arange(2, 7)[::-1]
                            footprints = (M + 8) * expected_population
                            M_ind = np.where(footprints < gpu_footprint)[0]
                            if len(M_ind):
                                M = M[M_ind[0]]
                        if M is not None:
                            logger.info(f"Using GPUIndexIVFFlat Index. Expected GPU-RAM footprint is "
                                        f"{pretty_format_number((M + 8) * expected_population / int(1e6))} MB")

                            config = faiss.GpuIndexIVFPQConfig()
                            config.device = inference_device
                            index = faiss.GpuIndexIVFPQ(res, d,  nlists, M, 8, faiss.METRIC_L2, config)
                        else:
                            logger.info(f"Using OPQ16_64,IVF{nlists},PQ8 Index")
                            index = faiss.index_factory(d, f'OPQ16_64,IVF{nlists},PQ8')
                            index = faiss.index_cpu_to_gpu(res, inference_device, index)

        if index is None:
            logger.error("Cannot find suitable index type")
            raise Exception

        self.index = index
        self.inference_device = inference_device

        self.training_index = None
        res = faiss.StandardGpuResources()
        if training_device != 'cpu' and inference_device == 'cpu':
            self.training_index = faiss.index_cpu_to_gpu(res, training_device, index)

        self.training_device = training_device

        if reducer == 'umap':
            import umap
            self.reducer = umap.UMAP()
        elif reducer == 'tsne':
            from sklearn.manifold import TSNE
            self.reducer = TSNE()
        else:
            raise NotImplementedError

    def train(self, x):

        x = x.to(self.training_device)
        self.index.train(x)

    def add(self, x, train=False):

        x = x.to(self.inference_device)
        self.index.add(x)

        if (train is None and not self.index.is_trained) or train:
            self.train(x)

    def most_similar(self, x, n=1):

        x = x.to(self.inference_device)
        D, I = self.index.search(x, n)
        return Similarities(index=I, distance=D)

    def __len__(self):
        return self.index.ntotal

    def reduce(self, z):
        return self.reducer.fit_transform(z)
