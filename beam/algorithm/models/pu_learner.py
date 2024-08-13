


# class BasePUEstimator:
#
#     def __init__(self, *args, embedding_features=None, cat_features=None, **kwargs):
#         self.embedding_features = embedding_features
#         self.cat_features = cat_features
#         super().__init__(*args, **kwargs)
#
#     def fit(self, X, y):
#         self.preprocess(X, y)
#         super().fit(X, y)
#
#     def preprocess(self, X, y):
#         raise NotImplementedError


from catboost import CatBoostClassifier



# class PUCatBoostClassifier(CatBoostClassifier):
#
#     def __sk