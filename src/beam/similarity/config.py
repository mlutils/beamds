
from ..config import BeamConfig, TransformerConfig, BeamParam


class SimilarityConfig(BeamConfig):

    parameters = [
        BeamParam('vector_dimension', int, None, 'dimension of the vectors'),
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


class TFIDFConfig(TransformerConfig):

    parameters = [
        BeamParam('max_features', int, 1000, 'maximum number of features'),
        BeamParam('max_df', float, 0.95, 'maximum document frequency'),
        BeamParam('min_df', float, 2, 'minimum document frequency'),
        BeamParam('use_idf', bool, True, 'use inverse document frequency'),
        BeamParam('smooth_idf', bool, True, 'smooth inverse document frequency'),
        BeamParam('sublinear_tf', bool, False, 'apply sublinear term frequency scaling'),
        BeamParam('sparse_framework', str, 'torch', 'sparse framework, can be "torch" or "scipy"'),
        BeamParam('sparse_layout', str, 'coo', 'sparse layout, can be "coo" or "csr"')
    ]

