from src.beam.utils import get_public_ip

from src.beam.config import BeamParam
from src.beam.similarity import SimilarityConfig, TFIDFConfig


def get_paths():

    ip = get_public_ip()

    if ip.startswith('199'):
        root_path = '/home/shared/data/results/enron/'
        path_to_data = '/home/hackathon_2023/data/enron/emails_dd.parquet'
        model_state_path = '/home/shared/data/results/enron/models/model_state_dd'
    else:
        root_path = '/home/mlspeech/elads/data/enron/data/'
        path_to_data = '/home/mlspeech/elads/data/enron/data/emails.parquet'
        model_state_path = '/home/mlspeech/elads/data/enron/models/model_state'

    return root_path, path_to_data, model_state_path


root_path, path_to_data, model_state_path = get_paths()


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
        # 'sparse_framework': 'scipy',
    }
    parameters = [
        BeamParam('nlp-model', type=str, default="en_core_web_sm", help='Spacy NLP model'),
        BeamParam('nlp-max-length', type=int, default=2000000, help='Spacy NLP max length'),
        BeamParam('path-to-data', type=str, default=path_to_data, help='Path to emails.parquet data'),
        BeamParam('root-path', type=str, default=root_path, help='Root path to store results'),
        BeamParam('train-ratio', type=float, default=0.4, help='Train ratio for split_dataset'),
        BeamParam('val-ratio', type=float, default=0.3, help='Validation ratio for split_dataset'),
        BeamParam('gap-days', type=int, default=3, help='Gap of days between subsets for split_dataset'),
        BeamParam('preprocess-body', type=bool, default=False, help='Preprocess body text'),
        BeamParam('preprocess-title', type=bool, default=False, help='Preprocess title text (subject)'),
        BeamParam('split-dataset', type=bool, default=False, help='Split the dataset'),
        BeamParam('build-dataset', type=bool, default=False, help='Build the dataset'),
        BeamParam('calc-tfidf', type=bool, default=True, help='Calculate TFIDF training vectors'),
        BeamParam('calc-dense', type=bool, default=True, help='Calculate Dense training vectors'),
        BeamParam('tokenizer', type=str, default="BAAI/bge-base-en-v1.5", help='Tokenizer model'),
        BeamParam('dense-model-path', type=str,
                  default="BAAI/bge-base-en-v1.5", help='Dense model for text similarity'),
        BeamParam('dense_model_device', type=str, default='cuda', help='Device for dense model'),
        BeamParam('tokenizer-chunksize', type=int, default=10000, help='Chunksize for tokenizer'),
        BeamParam('reload-state', bool, True, 'Load saved model'),
        BeamParam('save-state', bool, False, 'Save model state'),
        BeamParam('model-state-path', str, model_state_path, 'Path to saved model state'),
        BeamParam('batch_size', int, 32, 'Batch size for dense model'),
        BeamParam('k-sparse', int, 50, 'Number of sparse similarities to include in the dataset'),
        BeamParam('k-dense', int, 50, 'Number of dense similarities to include in the dataset'),
        BeamParam('svd-components', int, 128, 'Number of PCA components to use to compress the tfidf vectors'),
    ]
