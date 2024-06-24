from beam.utils import get_public_ip

from beam.config import BeamParam
from beam.algorithm.config import TextGroupExpansionConfig


class TicketSimilarityConfig(TextGroupExpansionConfig):

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
        BeamParam('tokenizer-chunksize', type=int, default=10000, help='Chunksize for tokenizer'),
        BeamParam('reload-state', bool, True, 'Load saved model'),
        BeamParam('save-state', bool, False, 'Save model state'),
        BeamParam('model-state-path', str, None, 'Path to saved model state'),
        BeamParam('labels-subset', list, None, 'build features for a subset of labels'),
    ]
