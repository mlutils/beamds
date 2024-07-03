from ..config import BeamParam
from ..similarity import SimilarityConfig, TFIDFConfig


class TextGroupExpansionConfig(SimilarityConfig, TFIDFConfig):

    # "en_core_web_trf"

    defaults = {
        'chunksize': 1000,
        'n_workers': 40,
        'mp_method': 'apply_async',
        'store_chunk': True,
        'store_path': None,
        'store_suffix': '.parquet',
        'override': False,
        'sparse_framework': 'scipy',
    }
    parameters = [
        BeamParam('tokenizer', type=str, default="BAAI/bge-base-en-v1.5", help='Tokenizer model'),
        BeamParam('dense-model', type=str, default="BAAI/bge-base-en-v1.5", help='Dense model for text similarity'),
        BeamParam('dense_model_device', type=str, default='cuda', help='Device for dense model'),
        BeamParam('tokenizer-chunksize', type=int, default=10000, help='Chunksize for tokenizer'),
        BeamParam('batch_size', int, 32, 'Batch size for dense model'),
        BeamParam('k-sparse', int, 50, 'Number of sparse similarities to include in the dataset'),
        BeamParam('k-dense', int, 50, 'Number of dense similarities to include in the dataset'),
        BeamParam('threshold', float, 0.5, 'Threshold for prediction model'),
        BeamParam('svd-components', int, 64, 'Number of PCA components to use to compress the tfidf vectors'),
        BeamParam('pca-components', int, 64, 'Number of PCA components to use to compress the dense vectors'),
        BeamParam('pu-n-estimators', int, 20, 'Number of estimators for the PU classifier'),
        BeamParam('pu-verbose', int, 10, 'Verbosity level for the PU classifier'),
        BeamParam('classifier-type', str, None, 'can be one of [None, catboost, rf]'),
    ]
