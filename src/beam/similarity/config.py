
from ..config import BeamConfig, BeamParam


class SimilarityConfig(BeamConfig):

    parameters = [
        BeamParam('faiss_d', int, None, 'dimension of the vectors'),
        BeamParam('expected_population', int, int(1e6), 'expected population of the index'),
        BeamParam('metric', str, 'l2', 'distance metric'),
        BeamParam('training_device', str, 'cpu', 'device for training'),
        BeamParam('inference_device', str, 'cpu', 'device for inference'),
        BeamParam('ram_footprint', int, 2 ** 8 * int(1e9), 'RAM footprint'),
        BeamParam('gpu_footprint', int, 24 * int(1e9), 'GPU footprint'),
        BeamParam('exact', bool, False, 'exact search'),
        BeamParam('nlists', int, None, 'number of lists for IVF'),
        BeamParam('faiss_M', int, None, 'M for IVFPQ'),
        BeamParam('reducer', str, 'umap', 'dimensionality reduction method')

    ]

