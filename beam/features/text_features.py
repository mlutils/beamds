import pandas as pd
from .feature import FeaturesCategories, BeamFeature, ParameterSchema, ParameterType
from ..resources import resource
from functools import cached_property, partial, wraps


class DenseEmbeddingFeature(BeamFeature):

    @wraps(BeamFeature.__init__)
    def __init__(self, embedder, *args, d=32, embedder_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedder = resource(embedder)
        self.embedder_kwargs = embedder_kwargs or {}
        self.d = d
        self.model = None

    @cached_property
    def parameters_schema(self):
        return {
            'd': ParameterSchema(name='d', kind=ParameterType.linspace, min_value=1, max_value=100,
                                 default_value=32, description='Size of embeddings'),
        }

    def _fit(self, x=None, v=None):
        if v is None:
            v = self.embedder.encode(x, **self.embedder_kwargs)

        from sklearn.decomposition import PCA
        self.model = PCA(n_components=self.d)
        self.model.fit(v)
        return v

    def _transform(self, x, v=None):
        if v is None:
            v = self.embedder.encode(x, **self.embedder_kwargs)
        v = self.model.transform(v)
        return pd.DataFrame(v, index=x.index)

    def fit_transform(self, x, **kwargs):
        v = self.fit(x)
        return self.transform(x, v)


class SparseEmbeddingFeature(BeamFeature):

    @wraps(BeamFeature.__init__)
    def __init__(self, tokenizer, *args, d=None, min_df=None, max_df=None, max_features=None, use_idf=None,
                 smooth_idf=None, sublinear_tf=None, tokenizer_kwargs=None, n_workers=0, mp_method='joblib',
                 **kwargs):
        super().__init__(*args, **kwargs)
        if tokenizer_kwargs:
            tokenizer = partial(tokenizer, **tokenizer_kwargs)

        d = d or self.parameters_schema['d'].default_value
        min_df = min_df or self.parameters_schema['min_df'].default_value
        max_df = max_df or self.parameters_schema['max_df'].default_value
        max_features = max_features or self.parameters_schema['max_features'].default_value
        use_idf = use_idf or self.parameters_schema['use_idf'].default_value
        smooth_idf = smooth_idf or self.parameters_schema['smooth_idf'].default_value
        sublinear_tf = sublinear_tf or self.parameters_schema['sublinear_tf'].default_value

        from ..similarity import TFIDF
        self.embedder = TFIDF(preprocessor=tokenizer, d=d, min_df=min_df, max_df=max_df, max_features=max_features,
                           use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf, n_workers=n_workers,
                           mp_method=mp_method)

        self.model = None

    @cached_property
    def parameters_schema(self):
        return {
            'd': ParameterSchema(name='d', kind=ParameterType.linspace, min_value=1, max_value=100,
                                 default_value=32, description='Size of embeddings'),
            'min_df': ParameterSchema(name='min_df', kind=ParameterType.linspace, min_value=1, max_value=100,
                                        default_value=2, description='Minimum document frequency'),
            'max_df': ParameterSchema(name='max_df', kind=ParameterType.linspace, min_value=0, max_value=1,
                                        default_value=1.0, description='Maximum document frequency'),
            'max_features': ParameterSchema(name='max_features', kind=ParameterType.linspace, min_value=1, max_value=100,
                                            default_value=None, description='Maximum number of features'),
            'use_idf': ParameterSchema(name='use_idf', kind=ParameterType.categorical, possible_values=[True, False],
                                        default_value=True, description='Use inverse document frequency'),
            'smooth_idf': ParameterSchema(name='smooth_idf', kind=ParameterType.categorical,
                                          possible_values=[True, False], default_value=True, description='Smooth idf'),
            'sublinear_tf': ParameterSchema(name='sublinear_tf', kind=ParameterType.categorical,
                                            possible_values=[True, False], default_value=False, description='Sublinear tf'),

        }

    def _fit(self, x, **kwargs):

        self.embedder.fit(x)
        v = self.embedder.transform(x)

        from sklearn.decomposition import TruncatedSVD
        self.model = TruncatedSVD(n_components=self.d)
        self.model.fit(v)
        return v

    def _transform(self, x, v=None):
        if v is None:
            v = self.embedder.transform(x)
        v = self.model.transform(v)
        return pd.DataFrame(v)

    def fit_transform(self, x, **kwargs):
        v = self.fit(x)
        return self.transform(x, v)
