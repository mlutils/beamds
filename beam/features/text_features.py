import pandas as pd
from .feature import FeaturesCategories, BeamFeature, ParameterSchema, ParameterType
from ..resources import resource
from ..misc import svd_preprocess
from functools import cached_property, partial, wraps


class DenseEmbeddingFeature(BeamFeature):

    @wraps(BeamFeature.__init__)
    def __init__(self, encoder, *args, d=32, encoder_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = resource(encoder)
        self.encoder_kwargs = encoder_kwargs or {}
        self.d = d
        self.model = None

    @cached_property
    def parameters_schema(self):
        return {
            'd': ParameterSchema(name='d', kind=ParameterType.linspace, start=1, end=100,
                                 default=32, description='Size of embeddings'),
        }

    def fit_callback(self, x=None, v=None):
        if v is None:
            v = self.encoder.encode(x, **self.encoder_kwargs)

        from sklearn.decomposition import PCA
        self.model = PCA(n_components=self.d)
        self.model.fit(v)
        return v

    def transform_callback(self, x, v=None):
        if v is None:
            v = self.encoder.encode(x, **self.encoder_kwargs)
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

        self.d = d or self.parameters_schema['d'].default
        min_df = min_df or self.parameters_schema['min_df'].default
        max_df = max_df or self.parameters_schema['max_df'].default
        max_features = max_features or self.parameters_schema['max_features'].default
        use_idf = use_idf or self.parameters_schema['use_idf'].default
        smooth_idf = smooth_idf or self.parameters_schema['smooth_idf'].default
        sublinear_tf = sublinear_tf or self.parameters_schema['sublinear_tf'].default

        from ..similarity import TFIDF
        self.encoder = TFIDF(preprocessor=tokenizer, min_df=min_df, max_df=max_df, max_features=max_features,
                           use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf, n_workers=n_workers,
                           mp_method=mp_method)

        self.model = None

    @cached_property
    def parameters_schema(self):
        return {
            'd': ParameterSchema(name='d', kind=ParameterType.linspace, start=1, end=100,
                                 default=32, description='Size of embeddings'),
            'min_df': ParameterSchema(name='min_df', kind=ParameterType.linspace, start=1, end=100,
                                      default=2, description='Minimum document frequency'),
            'max_df': ParameterSchema(name='max_df', kind=ParameterType.linspace, start=0, end=1,
                                      default=1.0, description='Maximum document frequency'),
            'max_features': ParameterSchema(name='max_features', kind=ParameterType.linspace, start=1, end=100,
                                            default=None, description='Maximum number of features'),
            'use_idf': ParameterSchema(name='use_idf', kind=ParameterType.categorical, choices=[True, False],
                                       default=True, description='Use inverse document frequency'),
            'smooth_idf': ParameterSchema(name='smooth_idf', kind=ParameterType.categorical,
                                          choices=[True, False], default=True, description='Smooth idf'),
            'sublinear_tf': ParameterSchema(name='sublinear_tf', kind=ParameterType.categorical,
                                            choices=[True, False], default=False, description='Sublinear tf'),

        }

    def fit_callback(self, x, **kwargs):

        x = list(x.squeeze().values)
        v = self.encoder.fit_transform(x)

        from sklearn.decomposition import TruncatedSVD
        self.model = TruncatedSVD(n_components=self.d)
        v = svd_preprocess(v)
        self.model.fit(v)
        return v

    def transform_callback(self, x, v=None):
        if v is None:
            x = list(x.squeeze().values)
            v = self.encoder.transform(x)
        v = svd_preprocess(v)
        v = self.model.transform(v)
        return pd.Series(list(v)).apply(list).to_frame(name=self.name)

