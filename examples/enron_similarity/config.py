from src.beam.utils import get_public_ip

from src.beam.config import BeamParam
from src.beam.similarity import SimilarityConfig, TFIDFConfig


class TicketSimilarityConfig(SimilarityConfig, TFIDFConfig):

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
        BeamParam('nlp-model', type=str, default="en_core_web_sm", help='Spacy NLP model'),
        BeamParam('nlp-max-length', type=int, default=2000000, help='Spacy NLP max length'),
        BeamParam('path-to-data', type=str, default=None, help='Path to emails.parquet data'),
        BeamParam('root-path', type=str, default=None, help='Root path to store results'),
        BeamParam('train-ratio', type=float, default=0.4, help='Train ratio for split_dataset'),
        BeamParam('val-ratio', type=float, default=0.3, help='Validation ratio for split_dataset'),
        BeamParam('gap-days', type=int, default=3, help='Gap of days between subsets for split_dataset'),
        BeamParam('preprocess-body', type=bool, default=False, help='Preprocess body text'),
        BeamParam('preprocess-title', type=bool, default=False, help='Preprocess title text (subject)'),
        BeamParam('split-dataset', type=bool, default=False, help='Split the dataset'),
        BeamParam('build-dataset', type=bool, default=False, help='Build the dataset'),
        BeamParam('calc-tfidf', type=bool, default=False, help='Calculate TFIDF training vectors'),
        BeamParam('calc-dense', type=bool, default=False, help='Calculate Dense training vectors'),
        BeamParam('build-features', type=bool, default=False, help='Build the classifier features'),
        BeamParam('tokenizer', type=str, default="BAAI/bge-base-en-v1.5", help='Tokenizer model'),
        BeamParam('dense-model-path', type=str,
                  default="BAAI/bge-base-en-v1.5", help='Dense model for text similarity'),
        BeamParam('dense_model_device', type=str, default='cuda', help='Device for dense model'),
        BeamParam('tokenizer-chunksize', type=int, default=10000, help='Chunksize for tokenizer'),
        BeamParam('reload-state', bool, True, 'Load saved model'),
        BeamParam('save-state', bool, False, 'Save model state'),
        BeamParam('model-state-path', str, None, 'Path to saved model state'),
        BeamParam('batch_size', int, 32, 'Batch size for dense model'),
        BeamParam('k-sparse', int, 50, 'Number of sparse similarities to include in the dataset'),
        BeamParam('k-dense', int, 50, 'Number of dense similarities to include in the dataset'),
        BeamParam('svd-components', int, 64, 'Number of PCA components to use to compress the tfidf vectors'),
        BeamParam('pca-components', int, 64, 'Number of PCA components to use to compress the dense vectors'),
        BeamParam('pu-n-estimators', int, 20, 'Number of estimators for the PU classifier'),
        BeamParam('pu-verbose', int, 10, 'Verbosity level for the PU classifier'),
        BeamParam('classifier-type', str, None, 'can be one of [None, catboost, rf]'),
    ]
