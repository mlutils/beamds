from functools import cached_property

import numpy as np
from chromadb import EmbeddingFunction, HttpClient

from ..type import Types
from ..utils import beam_service_port, check_type, as_numpy
from ..logging import beam_logger as logger

from .core import BeamSimilarity, Similarities


class ChromaEmbeddingFunction(EmbeddingFunction):

    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return self.model.encode(x).tolist()


class ChromaSimilarity(BeamSimilarity):

    def __init__(self, *args, hostname=None, port=None, database=None, tenant=None, collection=None, model=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.host = hostname or 'localhost'
        self.port = port or beam_service_port('chroma_port')
        self.database = database
        self.tenant = tenant
        self.collection_name = collection
        self.embedding_function = ChromaEmbeddingFunction(model) if model else None
        self.model_name = str(model)

    @cached_property
    def client(self):

        kwargs = {}
        if self.database:
            kwargs['database'] = self.database
        if self.tenant:
            kwargs['tenant'] = self.tenant

        chroma_client = HttpClient(host=self.host, port=self.port, **kwargs)
        return chroma_client

    @cached_property
    def collection(self):
        collection = self.client.get_or_create_collection(self.collection_name,
                                                          embedding_function=self.embedding_function)
        index = as_numpy(collection.get(include=[])['ids'])

        if index is not None and len(index):
            logger.info(f"Loaded {len(index)} items from collection {self.collection_name}")
            if index.dtype.kind in 'iuf':
                index = index.astype(int)
                self._is_numeric_index = True
                if np.abs(index - np.arange(len(index))).sum() == 0:
                    self._is_range_index = True
            else:
                self._is_numeric_index = False
                self._is_range_index = False

            self.index = index

        return collection

    @staticmethod
    def chroma_api_converter(x):
        x_type = check_type(x)
        embs = None
        docs = None
        if x_type.element == Types.str:
            docs = x
        else:
            embs = as_numpy(x)
        return embs, docs

    def add(self, x, index=None, **kwargs):

        index = self.add_index(x, index)
        embs, docs = self.chroma_api_converter(x)

        self.collection.add(ids=index.astype(str).tolist(), embeddings=embs, documents=docs, **kwargs)

    def search(self, x, k=1) -> Similarities:

        embs, docs = self.chroma_api_converter(x)

        sim = self.collection.query(query_texts=docs, query_embeddings=embs, n_results=k)
        ind = as_numpy(sim['ids'])
        if self.is_range_index:
            ind = ind.astype(int)
        return Similarities(index=ind, distance=as_numpy(sim['distances']), sparse_scores=None, metric=self.metric_type,
                            model=str(self.model_name))

