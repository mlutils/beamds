from functools import cached_property

from chromadb import EmbeddingFunction, HttpClient

from ..utils import beam_service_port, check_type, as_numpy

from .core import BeamSimilarity, Similarities


class ChromaEmbeddingFunction(EmbeddingFunction):

    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return self.model.encde(x)


class ChromaSimilarity(BeamSimilarity):

    def __init__(self, *args, hostname=None, port=None, database=None, tenant=None, collection=None, model=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.host = hostname or 'localhost'
        self.port = port or beam_service_port('chroma')
        self.database = database
        self.tenant = tenant
        self.collection_name = collection
        self.embedding_function = ChromaEmbeddingFunction(model) if model else None

    @cached_property
    def client(self):
        chroma_client = HttpClient(host=self.host, port=self.port, database=self.database,
                                   tenant=self.tenant)
        return chroma_client

    @cached_property
    def collection(self):
        return self.client.get_or_create_collection(self.collection_name)

    def add(self, x, index=None, **kwargs):

        x_type = check_type(x)
        embs = None
        docs = None
        if x_type.is_array:
            embs = as_numpy(x)
        else:
            docs = x

        self.collection.add(ids=index, embeddings=embs, documents=docs, **kwargs)

    def search(self, x, k=1) -> Similarities:
        x_type = check_type(x)
        embs = None
        docs = None
        if x_type.is_array:
            embs = as_numpy(x)
        else:
            docs = [x]

        sim = self.collection.query(query_texts=docs, query_embeddings=embs, n_results=k)
        return Similarities(index=sim['ids'], distance=sim['distances'], sparse_scores=None, metric=self.metric_type,
                            model=str(self.model))

