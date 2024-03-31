from typing import List

from sentence_transformers import SentenceTransformer

from ..utils import beam_device
from ..core import Processor
from ..llm import default_tokenizer
from .dense import DenseSimilarity
from ..path import local_copy


class TextSimilarity(DenseSimilarity):
    def __init__(self, *args, dense_model_path="BAAI/bge-base-en-v1.5", tokenizer_path=None,
                 use_dense_model_tokenizer=True, dense_model=None, tokenizer=None, cache_folder=None,
                 device="cuda", batch_size=32, show_progress_bar=True, **kwargs):
        """
        Initialize the RAG (Retrieval-Augmented Generation) retriever.

        Parameters:
        data_train (pd.DataFrame): A dataframe containing the training data with a 'text' column.
        alfa (float): Weighting factor for combining dense and sparse retrieval scores.
        embedding_model (str): The name of the sentence transformer model used for embedding.
        model (str): The name of the transformer model used for causal language modeling.
        device (str): The device to run the models on (e.g., 'cuda:1' for GPU).
        """

        Processor.__init__(self, *args, device=device, tokenizer_path=tokenizer_path,
                           dense_model_path=dense_model_path, use_dense_model_tokenizer=use_dense_model_tokenizer,
                           cache_folder=cache_folder, batch_size=batch_size, show_progress_bar=show_progress_bar,
                           **kwargs)

        # Device to run the model
        self.device = beam_device(self.get_hparam('device'))
        # Load the sentence transformer model for embeddings
        self.cache_folder = self.get_hparam('cache_folder', cache_folder)
        self.batch_size = self.get_hparam('batch_size', batch_size)
        self.show_progress_bar = self.get_hparam('show_progress_bar', show_progress_bar)

        if dense_model is None:
            self.dense_model = SentenceTransformer(self.get_hparam('dense_model_path'), device=str(self.device))
        else:
            self.dense_model = dense_model
            self.dense_model.to(self.device)

        d = self.dense_model.get_sentence_embedding_dimension()

        super().__init__(*args, device=device, tokenizer_path=tokenizer_path,
                         dense_model_path=dense_model_path, use_dense_model_tokenizer=use_dense_model_tokenizer,
                         d=d, **kwargs)

        self._tokenizer = None
        if tokenizer is None:
            if self.get_hparam('tokenizer_path') is not None:
                from transformers import PreTrainedTokenizerFast
                self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.get_hparam('tokenizer_path'))
        else:
            self._tokenizer = tokenizer

    @property
    def tokenizer(self):

        tokenizer = self._tokenizer
        if self._tokenizer is None:
            tokenizer = default_tokenizer
            if self.get_hparam('use_dense_model_tokenizer'):
                tokenizer = self.dense_model.tokenize

        return tokenizer

    def add(self, x, index=None, **kwargs):

        x, index = self.extract_data_and_index(x, index)
        dense_vectors = self.encode(x)
        super().add(dense_vectors, index)

    def encode(self, x: List[str]):
        x = list(x)
        return self.dense_model.encode(x, batch_size=self.batch_size, show_progress_bar=True, convert_to_tensor=True)

    def search(self, x: List[str], k=1):

        x, _ = self.extract_data_and_index(x)
        dense_vectors = self.encode(x)
        similarities = super().search(dense_vectors, k)
        return similarities

    @property
    def exclude_pickle_attributes(self):
        return super().exclude_pickle_attributes + ['dense_model', '_tokenizer']

    def save_state(self, path, ext=None):

        super().save_state(path, ext)

        if self._tokenizer is not None:
            if hasattr(self._tokenizer, 'save_pretrained'):
                tokenizer_path = path.joinpath('tokenizer.hf')
                with local_copy(tokenizer_path) as p:
                    self._tokenizer.save_pretrained(p)
            else:
                tokenizer_path = path.joinpath('tokenizer.pkl')
                tokenizer_path.write(self._tokenizer)

    def load_state(self, path):

        super().load_state(path)
        self.dense_model = SentenceTransformer(self.get_hparam('dense_model_path'), device=str(self.device))
        if path.joinpath('tokenizer.hf').exists():
            from transformers import PreTrainedTokenizerFast
            with local_copy(path.joinpath('tokenizer.hf')) as p:
                self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=p)
        elif path.joinpath('tokenizer.pkl').exists():
            self._tokenizer = path.joinpath('tokenizer.pkl').read()

