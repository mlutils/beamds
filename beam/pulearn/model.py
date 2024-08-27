import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from pulearn import BaggingPuClassifier
from sklearn.utils import check_array


class BeamCatboostClassifier(CatBoostClassifier):
    def __init__(self, *args, **kwargs):
        self._cat_features_mapping = None
        self._embedding_features_mapping = None
        self._text_features_mapping = None
        self._numerical_features_mapping = None
        self._fit_kwargs = {}
        super().__init__(*args, **kwargs)

    def __getstate__(self):
        params = super(BeamCatboostClassifier, self).__getstate__()

        aux = dict(_cat_features_mapping=self._cat_features_mapping,
                   _embedding_features_mapping=self._embedding_features_mapping,
                   _text_features_mapping=self._text_features_mapping,
                   _numerical_features_mapping=self._numerical_features_mapping,
                   _fit_kwargs=self._fit_kwargs)
        state = {'params': params,
                 'aux': aux}

        return state

    def __setstate__(self, state):
        params = state['params']
        aux = state['aux']
        for k, v in aux.items():
            setattr(self, k, v)
        super(BeamCatboostClassifier, self).__setstate__(params)

    def update_fit_kwargs(self, fit_kwargs: dict = None):
        self._fit_kwargs = fit_kwargs if fit_kwargs is not None else {}

    def update_special_features(self, numerical_features_mapping: dict[str, int] = None,
                                      cat_features_mapping: dict[str, int] = None,
                                      embedding_features_mapping: dict[str, list] = None,
                                      text_features_mapping: dict[str, int] = None):

        self._cat_features_mapping = cat_features_mapping if cat_features_mapping is not None else {}
        self._embedding_features_mapping = embedding_features_mapping if embedding_features_mapping is not None else {}
        self._text_features_mapping = text_features_mapping if text_features_mapping is not None else {}
        self._numerical_features_mapping = numerical_features_mapping if numerical_features_mapping is not None else {}

    def preprocess(self, X):

        cat_columns, cat_indices = list(zip(*self._cat_features_mapping.items())) \
            if len(self._cat_features_mapping) > 0 else ([], [])
        text_columns, text_indices = list(zip(*self._text_features_mapping.items())) \
            if len(self._text_features_mapping) > 0 else ([], [])
        embedding_columns, embedding_indices = list(zip(*self._embedding_features_mapping.items())) \
            if len(self._embedding_features_mapping) > 0 else ([], [])
        numerical_columns, numerical_indices = list(zip(*self._numerical_features_mapping.items())) \
            if len(self._numerical_features_mapping) > 0 else ([], [])

        cat = pd.DataFrame(X[:, cat_indices].astype('int'), columns=cat_columns)
        text = pd.DataFrame(X[:, text_indices].astype('str'), columns=text_columns)
        numerical = pd.DataFrame(X[:, numerical_indices].astype('float'), columns=numerical_columns)

        embedding = []
        for col, indices in zip(embedding_columns, embedding_indices):
            embedding.append(pd.Series(X[:, indices].astype('float').tolist(), name=col))

        if len(embedding) > 0:
            embedding = pd.concat(embedding, axis=1)
        else:
            embedding = pd.DataFrame()

        X = pd.concat([numerical, cat, text, embedding], axis=1)

        cat_features = cat_columns if len(cat_columns) > 0 else None
        text_features = text_columns if len(text_columns) > 0 else None
        embedding_features = embedding_columns if len(embedding_columns) > 0 else None

        return X, cat_features, text_features, embedding_features

    def fit(self, X, *args, **kwargs):
        X, cat_features, text_features, embedding_features = self.preprocess(X)
        kwargs.update(self._fit_kwargs)
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
        c._cat_features_mapping = self._cat_features_mapping
        c._embedding_features_mapping = self._embedding_features_mapping
        c._text_features_mapping = self._text_features_mapping
        c._numerical_features_mapping = self._numerical_features_mapping
        c._fit_kwargs = self._fit_kwargs
        return c


class BeamPUClassifier(BaggingPuClassifier):
    def __init__(self, *args, cat_features: list[str] = None, embedding_features: list[str] = None,
                                text_features: list[str] = None, **kwargs):
        super().__init__(*args, **kwargs)

        self._cat_features = cat_features if cat_features is not None else []
        self._embedding_features = embedding_features if embedding_features is not None else []
        self._text_features = text_features if text_features is not None else []

    def set_features(self, cat_features: list[str] = None, embedding_features: list[str] = None,
                                text_features: list[str] = None):
        self._cat_features = cat_features if cat_features is not None else []
        self._embedding_features = embedding_features if embedding_features is not None else []
        self._text_features = text_features if text_features is not None else []

    def preprocess(self, X, y=None):

        if type(X) is Pool:
            X = X.get_features()
            y = X.get_label()

        try:
            X = check_array(X)
            return X, y
        except ValueError:
            pass

        numerical_features = [col for col in X.columns if col not in
                              self._cat_features + self._embedding_features + self._text_features]

        xs = [X[numerical_features].values, X[self._cat_features].values, X[self._text_features].values]

        numerical_features_mapping = {col: i for i, col in enumerate(numerical_features)}
        ns = len(numerical_features)
        cat_features_mapping = {col: i+ns for i, col in enumerate(self._cat_features)}
        ns += len(self._cat_features)
        text_features_mapping = {col: i+ns for i, col in enumerate(self._text_features)}
        ns += len(self._text_features)

        embedding_features_mapping = {}
        for f in self._embedding_features:
            l = len(X[f].values[0])
            xf = np.array(X[f].to_list())
            xs.append(xf)
            embedding_features_mapping[f] = [i+ns for i in range(l)]
            ns += l

        X = np.concatenate(xs, axis=1)

        if isinstance(self.estimator, BeamCatboostClassifier):
            self.estimator.update_special_features(numerical_features_mapping=numerical_features_mapping,
                                                   cat_features_mapping=cat_features_mapping,
                                                   text_features_mapping=text_features_mapping,
                                                   embedding_features_mapping=embedding_features_mapping)

        return X, y

    def fit(self, X, y=None, *args, estimator_fit_kwargs=None, **kwargs):

        X, y = self.preprocess(X, y)
        if isinstance(self.estimator, BeamCatboostClassifier):
            self.estimator.update_fit_kwargs(estimator_fit_kwargs)

        return super().fit(X, y, *args, **kwargs)

    def predict(self, X):
        X, _ = self.preprocess(X)
        return super().predict(X)

    def predict_proba(self, X):
        X, _ = self.preprocess(X)
        return super().predict_proba(X)

    def predict_log_proba(self, X):
        X, _ = self.preprocess(X)
        return super().predict_log_proba(X)

