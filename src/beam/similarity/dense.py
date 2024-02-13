import math
from collections import namedtuple

import faiss
import numpy as np

from .. import beam_path
from ..logger import beam_logger as logger
from ..path import local_copy
from ..utils import pretty_format_number
from ..core import Processor

# working with faiss and torch
Similarities = namedtuple("Similarities", "index distance")


class BeamSimilarity(Processor):

    def __init__(self, *args, index=None, faiss_d=None, expected_population=int(1e6),
                 metric='l2', training_device='cpu', inference_device='cpu', ram_footprint=2**8*int(1e9),
                 gpu_footprint=24*int(1e9), exact=False, nlists=None, faiss_M=None,
                 reducer='umap', **kwargs):

        '''
        To Choose an index, follow https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        @param faiss_d:
        @param expected_population:
        @param metric:
        @param ram_size:
        @param gpu_size:
        @param exact_results:
        @param reducer:
        '''

        super().__init__(*args, faiss_d=faiss_d, expected_population=expected_population,
                         metric=metric, training_device=training_device,
                         inference_device=inference_device, ram_footprint=ram_footprint,
                         gpu_footprint=gpu_footprint, exact=exact, nlists=nlists, M=faiss_M,
                         reducer=reducer, **kwargs)

        faiss_d = self.get_hparam('faiss_d', faiss_d)
        expected_population = self.get_hparam('expected_population', expected_population)
        metric = self.get_hparam('metric', metric)
        training_device = self.get_hparam('training_device', training_device)
        inference_device = self.get_hparam('inference_device', inference_device)
        ram_footprint = self.get_hparam('ram_footprint', ram_footprint)
        gpu_footprint = self.get_hparam('gpu_footprint', gpu_footprint)
        exact = self.get_hparam('exact', exact)
        nlists = self.get_hparam('nlists', nlists)
        faiss_M = self.get_hparam('faiss_M', faiss_M)
        reducer = self.get_hparam('reducer', reducer)

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
                                f"{pretty_format_number(4 * faiss_d * expected_population / int(1e6))} MB")
                    index = faiss.IndexFlat(faiss_d, metric)
                else:
                    if faiss_M is None:
                        faiss_M = 2 ** np.arange(2, 7)[::-1]
                        footprints = (faiss_d * 4 + faiss_M * 8) * expected_population
                        M_ind = np.where(footprints < ram_footprint)[0]
                        if len(M_ind):
                            faiss_M = int(faiss_M[M_ind[0]])
                    if faiss_M is not None:
                        logger.info(f"Using HNSW{faiss_M}. Expected RAM footprint is "
                                    f"{pretty_format_number(footprints[M_ind[0]] / int(1e6))} MB")
                        index = faiss.IndexHNSWFlat(faiss_d, faiss_M, metric)
                    else:
                        logger.info(f"Using OPQ16_64,IVF{nlists},PQ8 Index")
                        index = faiss.index_factory(faiss_d, f'OPQ16_64,IVF{nlists},PQ8')

            else:

                res = faiss.StandardGpuResources()
                if exact:
                    config = faiss.GpuIndexFlatConfig()
                    config.device = inference_device
                    logger.info(f"Using GPUFlat Index. Expected GPU-RAM footprint is "
                                f"{pretty_format_number(4 * faiss_d * expected_population / int(1e6))} MB")

                    index = faiss.GpuIndexFlat(res, faiss_d, metric, config)
                else:

                    if (4 * faiss_d + 8) * expected_population <= gpu_footprint:
                        logger.info(f"Using GPUIndexIVFFlat Index. Expected GPU-RAM footprint is "
                                    f"{pretty_format_number((4 * faiss_d + 8) * expected_population / int(1e6))} MB")
                        config = faiss.GpuIndexIVFFlatConfig()
                        config.device = inference_device
                        index = faiss.GpuIndexIVFFlat(res, faiss_d, nlists, faiss.METRIC_L2, config)
                    else:

                        if faiss_M is None:
                            faiss_M = 2 ** np.arange(2, 7)[::-1]
                            footprints = (faiss_M + 8) * expected_population
                            M_ind = np.where(footprints < gpu_footprint)[0]
                            if len(M_ind):
                                faiss_M = faiss_M[M_ind[0]]
                        if faiss_M is not None:
                            logger.info(f"Using GPUIndexIVFFlat Index. Expected GPU-RAM footprint is "
                                        f"{pretty_format_number((faiss_M + 8) * expected_population / int(1e6))} MB")

                            config = faiss.GpuIndexIVFPQConfig()
                            config.device = inference_device
                            index = faiss.GpuIndexIVFPQ(res, faiss_d, nlists, faiss_M, 8, faiss.METRIC_L2, config)
                        else:
                            logger.info(f"Using OPQ16_64,IVF{nlists},PQ8 Index")
                            index = faiss.index_factory(faiss_d, f'OPQ16_64,IVF{nlists},PQ8')
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

    def state_attributes(self):
        return ['index', 'training_index']

    def save_state(self, path, ext=None):

        path = beam_path(path)
        path.mkdir()
        with local_copy(path.joinpath('index.bin'), as_beam_path=False) as p:
            faiss.write_index(self.index, p)
        if self.training_index:
            with local_copy(path.joinpath('training_index.bin'), as_beam_path=False) as p:
                faiss.write_index(self.training_index, p)

    def load_state(self, path):

        path = beam_path(path)

        with local_copy(path.joinpath('index.bin'), as_beam_path=False) as p:
            self.index = faiss.read_index(p)
        if path.joinpath('training_index.bin').is_file():
            with local_copy(path.joinpath('training_index.bin'), as_beam_path=False) as p:
                self.training_index = faiss.read_index(p)
