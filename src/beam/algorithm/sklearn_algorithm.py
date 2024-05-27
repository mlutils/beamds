from .core_algorithm import Algorithm
from sklearn.base import BaseEstimator


class SKLearnAlgorithm(Algorithm, BaseEstimator):
    def __init__(self, hparams, name=None, **kwargs):
        super().__init__(hparams=hparams, name=name, **kwargs)

    def _fit(self, *args, **kwargs):
        raise NotImplementedError("please implement _fit method in your sub-class")

    def fit(self, *args, **kwargs):
        return self._fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        raise NotImplementedError('predict method not implemented')