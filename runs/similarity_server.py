from beam.similarity import SimilarityConfig, TFIDFConfig, TFIDF, DenseSimilarity, SparseSimilarity, TextSimilarity
from beam.config import BeamParam
from beam.base import BeamBase


class SimilarityServerConfig(SimilarityConfig, TFIDFConfig):
    parameters = [
        BeamParam('similarity_type', str, 'tfidf', 'similarity type [tfidf, dense, sparse, text]'),
    ]


class SimilarityDatabase(BeamBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.similarity_type = self.get_hparam('similarity_type')
        if self.similarity_type == 'tfidf':
            self.similarity_model = TFIDF
        elif self.similarity_type == 'dense':
            self.similarity_model = DenseSimilarity
        elif self.similarity_type == 'sparse':
            self.similarity_model = SparseSimilarity
        elif self.similarity_type == 'text':
            self.similarity_model = TextSimilarity

        self.models = {}

    def get_model(self, table, **kwargs):
        if table not in self.models:
            hparams = self.hparams.copy()
            hparams.update(kwargs)
            self.models[table] = self.similarity_model(hparams=hparams)
        return self.models[table]

    def add(self, x, table, **kwargs):
        model = self.get_model(table, **kwargs)
        model.add(x)

    def fit(self, x, table, **kwargs):
        model = self.get_model(table, **kwargs)
        model.fit(x)

    def transform(self, x, table, index=None, **kwargs):
        model = self.get_model(table, **kwargs)
        return model.transform(x, index=index)

    def fit_transform(self, x, table, **kwargs):
        model = self.get_model(table, **kwargs)
        return model.fit_transform(x)

    def search(self, x, table, k=5, **kwargs):
        model = self.models[table]
        return model.search(x, k=k, **kwargs)

    def reset(self, table, **kwargs):
        model = self.get_model(table, **kwargs)
        model.reset()
