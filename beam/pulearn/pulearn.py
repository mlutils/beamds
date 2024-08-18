import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from pulearn import BaggingPuClassifier


class BeamCatboostClassifier(CatBoostClassifier):
    def __init__(self, *args, **kwargs):
        self._cat_features = None
        self._embedding_features = None
        self._text_features = None
        self._columns = None
        self._columns_mapping = None
        super().__init__(*args, **kwargs)

    def update_special_features(self, columns: list[str], cat_features: list[str] = None, embedding_features: list[str] = None,
                                text_features: list[str] = None):
        self._columns = columns
        self._columns_mapping = {col: i for i, col in enumerate(columns)}

        self._cat_features = [self._columns_mapping[col] for col in cat_features] if cat_features is not None else []
        self._text_features = [self._columns_mapping[col] for col in text_features] if text_features is not None else []
        self._embedding_features = [self._columns_mapping[col] for col in embedding_features] \
            if embedding_features is not None else []

        print(self._cat_features, self._embedding_features, self._text_features)

    def rest_columns(self, l):
        return [i for i in range(l) if i not in self._cat_features + self._text_features + self._embedding_features]

    def preprocess(self, x):

        cat = x[:, self._cat_features] if self._cat_features is not None else pd.DataFrame()
        cat = cat.astype('int')
        cat = pd.DataFrame(cat, columns=[self._columns[i] for i in self._cat_features])

        text = x[:, self._text_features] if self._text_features is not None else pd.DataFrame()
        text = text.astype('str')
        text = pd.DataFrame(text, columns=[self._columns[i] for i in self._text_features])

        embedding = x[:, self._embedding_features] if self._embedding_features is not None else pd.DataFrame()
        embedding = pd.DataFrame(embedding, columns=[self._columns[i] for i in self._embedding_features])

        rest_columns = self.rest_columns(x.shape[1])
        rest = x[:, rest_columns]
        rest = pd.DataFrame(rest, columns=[self._columns[i] for i in rest_columns])

        x = pd.concat([cat, text, embedding, rest], axis=1)
        cat_features = list(range(cat.shape[1]))
        text_features = list(range(cat.shape[1], cat.shape[1] + text.shape[1]))
        embedding_features = list(range(cat.shape[1] + text.shape[1],
                                        cat.shape[1] + text.shape[1] + embedding.shape[1]))

        cat_features = cat_features if len(cat_features) > 0 else None
        text_features = text_features if len(text_features) > 0 else None
        embedding_features = embedding_features if len(embedding_features) > 0 else None

        return x, cat_features, text_features, embedding_features

    def fit(self, X, *args, **kwargs):
        X, cat_features, text_features, embedding_features = self.preprocess(X)
        return super().fit(X, *args, cat_features=cat_features, text_features=text_features,
                           embedding_features=embedding_features, **kwargs)

    def predict(self, X, *args, **kwargs):
        X, _, _, _ = self.preprocess(X)
        return super().predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        X, _, _, _ = self.preprocess(X)
        return super().predict_proba(X, *args, **kwargs)

    def __sklearn_clone__(self):
        c = self.copy()
        c._cat_features = self._cat_features
        c._embedding_features = self._embedding_features
        c._text_features = self._text_features
        c._columns = self._columns
        return c


class BeamPUClassifier(BaggingPuClassifier):
    def __init__(self, *args, cat_features: list[str] = None, embedding_features: list[str] = None,
                                text_features: list[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._cat_features = cat_features
        self._embedding_features = embedding_features
        self._text_features = text_features

    def fit(self, X, y, *args, **kwargs):
        if isinstance(self.estimator, BeamCatboostClassifier):
            self.estimator.update_special_features(X.columns, cat_features=self._cat_features,
                                                   embedding_features=self._embedding_features,
                                                   text_features=self._text_features)
        return super().fit(X, y, *args, **kwargs)

