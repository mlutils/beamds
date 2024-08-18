from typing import List

from ..logging import beam_logger as logger
from ..type.utils import is_beam_resource
from ..utils import beam_device, tqdm_beam as tqdm
from ..processor import Processor
from ..llm import default_tokenizer
from .dense import DenseSimilarity
from ..path import local_copy, beam_path
from ..resources import resource
from ..embedding import RobustDenseEncoder


class TextSimilarity(DenseSimilarity):
    def __init__(self, *args, dense_model=None, tokenizer_path=None,
                 use_dense_model_tokenizer=True, tokenizer=None, cache_folder=None,
                 dense_model_device='cuda', vector_store_device="cpu", vector_store_training_device='cpu', batch_size=32, show_progress_bar=True,
                 st_kwargs=None, **kwargs):
        """
        Initialize the RAG (Retrieval-Augmented Generation) retriever.

        Parameters:
        data_train (pd.DataFrame): A dataframe containing the training data with a 'text' column.
        alfa (float): Weighting factor for combining dense and sparse retrieval scores.
        embedding_model (str): The name of the sentence transformer model used for embedding.
        model (str): The name of the transformer model used for causal language modeling.
        device (str): The device to run the models on (e.g., 'cuda:1' for GPU).
        """

        Processor.__init__(self, *args, tokenizer_path=tokenizer_path, dense_model_device=dense_model_device,
                           use_dense_model_tokenizer=use_dense_model_tokenizer, cache_folder=cache_folder,
                           batch_size=batch_size, show_progress_bar=show_progress_bar, **kwargs)

        # Device to run the model
        self.dense_model_device = beam_device(self.get_hparam('dense_model_device'))
        # Load the sentence transformer model for embeddings
        self.cache_folder = self.get_hparam('cache_folder', cache_folder)
        self.batch_size = self.get_hparam('batch_size', batch_size)
        self.show_progress_bar = self.get_hparam('show_progress_bar', show_progress_bar)
        st_kwargs = st_kwargs or {}

        self.dense_model = self.load_dense_model(dense_model=dense_model,
                                                 dense_model_device=self.dense_model_device,
                                                 batch_size=self.batch_size,
                                                 **st_kwargs)

        d = self.dense_model.get_sentence_embedding_dimension()

        super().__init__(*args, inference_device=vector_store_device,
                         training_device=vector_store_training_device, tokenizer_path=tokenizer_path,
                         dense_model=dense_model, use_dense_model_tokenizer=use_dense_model_tokenizer,
                         d=d, **kwargs)

        self._tokenizer = None
        if tokenizer is None:
            if self.get_hparam('tokenizer_path') is not None:
                from transformers import PreTrainedTokenizerFast
                self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.get_hparam('tokenizer_path'))
        else:
            self._tokenizer = tokenizer

    @staticmethod
    def load_dense_model(dense_model=None, dense_model_device=None, batch_size=None, **st_kwargs):

        chunksize = None
        if type(dense_model) is str:
            dense_model_resource = resource(dense_model)
            if dense_model_resource.is_beam_client:
                dense_model = dense_model_resource
                chunksize = 100 * batch_size
            else:
                from sentence_transformers import SentenceTransformer
                dense_model = SentenceTransformer(dense_model, device=str(dense_model_device), **st_kwargs)
        elif not is_beam_resource(dense_model):
            dense_model.to(dense_model_device)
        else:
            raise ValueError(f"Invalid dense model: {dense_model}")

        dense_model = RobustDenseEncoder(dense_model, batch_size=batch_size,
                                         device=dense_model_device, chunksize=chunksize)

        return dense_model

    @property
    def tokenizer(self):

        tokenizer = self._tokenizer
        if self._tokenizer is None:
            tokenizer = default_tokenizer
            if self.get_hparam('use_dense_model_tokenizer'):
                tokenizer = self.dense_model.tokenize

        return tokenizer

    def add(self, x, index=None, cache_dir=None, **kwargs):

        x, index = self.extract_data_and_index(x, index, convert_to=None)
        dense_vectors = self.encode(x, cache_dir=cache_dir, aggregate=False)
        if not type(dense_vectors) is list:
            dense_vectors = [dense_vectors]

        logger.info(f"Adding {len(dense_vectors)} dense vectors to the index.")
        n = 0
        for dv in tqdm(dense_vectors):
            ind = None
            if index is not None:
                ind = index[n:n + len(dv)]
                n += len(dv)
            super().add(dv, ind)

    def encode(self, x: List[str], cache_dir=None, aggregate=True):
        x = list(x)
        kwargs = {}
        if cache_dir is not None:
            kwargs['cache_dir'] = cache_dir
        return self.dense_model.encode(x, batch_size=self.batch_size, show_progress_bar=True, **kwargs)

    def search(self, x: List[str], k=1):

        x, _ = self.extract_data_and_index(x, convert_to=None)
        dense_vectors = self.encode(x)
        similarities = super().search(dense_vectors, k)
        return similarities

    @classmethod
    @property
    def excluded_attributes(cls):
        return super(TextSimilarity, cls).excluded_attributes.union(['_tokenizer', 'dense_model'])

    def save_state_dict(self, state, path, ext=None, exclude: List = None, **kwargs):

        exclude = set(exclude) if exclude is not None else set()

        path = beam_path(path)
        super().save_state_dict(state, path, ext, exclude, **kwargs)

        if self.hasattr('_tokenizer') and self._tokenizer is not None:
            if hasattr(self._tokenizer, 'save_pretrained'):
                tokenizer_path = path.joinpath('tokenizer.hf')
                with local_copy(tokenizer_path) as p:
                    self._tokenizer.save_pretrained(p)
            else:
                tokenizer_path = path.joinpath('tokenizer.pkl')
                tokenizer_path.write(self._tokenizer)

    def load_state_dict(self, path, ext=None, exclude: List = None, **kwargs):

        exclude = set(exclude) if exclude is not None else set()
        exclude = exclude.update(['_tokenizer'])

        state = super().load_state_dict(path, ext, exclude, **kwargs)

        path = beam_path(path)
        self.dense_model = None
        logger.warning("Dense model is not loaded. Use set_dense_model to manually set the dense model.")

        if path.joinpath('tokenizer.hf').exists():
            from transformers import PreTrainedTokenizerFast
            with local_copy(path.joinpath('tokenizer.hf')) as p:
                self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=p)
        elif path.joinpath('tokenizer.pkl').exists():
            self._tokenizer = path.joinpath('tokenizer.pkl').read()

        return state

    def set_dense_model(self, dense_model):
        self.dense_model = dense_model
        logger.info(f"Set dense model to {dense_model}")
        return self



