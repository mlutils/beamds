from catboost import CatBoostClassifier


class BeamCatboostClassifier(CatBoostClassifier):
    def __init__(self, *args, cat_features=None, embedding_features=None, text_features=None, **kwargs):
        super().__init__(*args,  cat_features=cat_features, embedding_features=None, text_features=None, **kwargs)
        self.cat_features = cat_features
        self.embedding_features = embedding_features
        self.text_features = text_features

    def preprocess(self, x):
        return x

    def fit(self, X, *args, **kwargs):
        X = self.preprocess(X)
        return super().fit(X, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        X = self.preprocess(X)
        return super().predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        X = self.preprocess(X)
        return super().predict_proba(X, *args, **kwargs)
