
from ..config import BeamConfig, TransformerConfig, BeamParam


class SimilarityConfig(BeamConfig):

    parameters = [
        BeamParam('vector_dimension', int, None, 'dimension of the vectors'),
        BeamParam('expected_population', int, int(1e6), 'expected population of the index'),
        BeamParam('metric', str, 'l2', 'distance metric [l2, cosine, max, prod, quantile]'),
        BeamParam('training_device', str, 'cpu', 'device for training'),
        BeamParam('inference_device', str, 'cpu', 'device for inference'),
        BeamParam('dense_model_device', str, 'cuda', 'device for dense model'),
        BeamParam('ram_footprint', int, 2 ** 8 * int(1e9), 'RAM footprint'),
        BeamParam('gpu_footprint', int, 24 * int(1e9), 'GPU footprint'),
        BeamParam('exact', bool, False, 'exact search'),
        BeamParam('nlists', int, None, 'number of lists for IVF'),
        BeamParam('faiss_M', int, None, 'M for IVFPQ'),
        BeamParam('reducer', str, 'umap', 'dimensionality reduction method'),
        BeamParam('quantile', float, 0.9, 'quantile for the quantile metric'),
    ]


class TFIDFConfig(TransformerConfig):

    defaults = dict(metric='bm25')

    parameters = [
        BeamParam('max_features', int, None, 'maximum number of features'),
        BeamParam('max_df', float, 0.95, 'maximum document frequency'),
        BeamParam('min_df', float, 2, 'minimum document frequency'),
        BeamParam('use_idf', bool, True, 'use inverse document frequency'),
        BeamParam('smooth_idf', bool, True, 'smooth inverse document frequency'),
        BeamParam('sublinear_tf', bool, False, 'apply sublinear term frequency scaling'),
        BeamParam('sparse_framework', str, 'torch', 'sparse framework, can be "torch" or "scipy"'),
        BeamParam('sparse_layout', str, 'coo', 'sparse layout, can be "coo" or "csr"'),
        BeamParam('norm', str, 'l2', 'Each output row will have unit norm, either [l1, l2, none]'),
        BeamParam('bm25_k1', float, 1.5, 'bm25 k1 parameter'),
        BeamParam('bm25_b', float, 0.75, 'bm25 b parameter'),
        BeamParam('bm25_epsilon', float, 0.25, 'bm25 epsilon parameter'),
    ]

