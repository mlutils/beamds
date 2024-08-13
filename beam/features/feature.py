from functools import wraps

import numpy as np
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from ..utils import as_numpy, check_type, as_list
from ..base import BeamBase
from ..type.utils import is_pandas_dataframe, is_pandas_series


class FeaturesCategories(Enum):
    numerical = 'numerical'
    categorical = 'categorical'
    embedding = 'embedding'
    text = 'text'


class ParameterType(Enum):
    categorical = 'categorical'
    linspace = 'linspace'
    logspace = 'logspace'
    uniform = 'uniform'
    loguniform = 'loguniform'


@dataclass
class ParameterSchema:
    name: str
    kind: ParameterType
    possible_values: list[str] | None = None
    min_value: float | None = None
    max_value: float | None = None
    default_value: float | None = None
    description: str | None = None


class BeamFeature(BeamBase):

    def __init__(self, *args, func=None, columns: str | list[str] = None, name=None, kind=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.func = func
        self.columns = [columns] if isinstance(columns, str) else columns
        self.kind = kind or FeaturesCategories.numerical

    @property
    def parameters_schema(self):
        return {}

    def preprocess(self, x):
        x_type = check_type(x)
        if x_type.is_dataframe:
            if self.columns is None:
                self.columns = x.columns
            x = x[self.columns]
        else:
            # TODO: build this logic
            x = pd.DataFrame(x, columns=self.columns)
        return x

    def transform(self, x, _preprocessed=False, **kwargs) -> pd.DataFrame:
        if not _preprocessed:
            x = self.preprocess(x)
        return self._transform(x, **kwargs)

    def _transform(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self.func(x)

    def _fit(self, x: pd.DataFrame, **kwargs):
        pass

    def fit(self, x: pd.DataFrame, _preprocessed=False, **kwargs):
        if not _preprocessed:
            x = self.preprocess(x)
        return self._fit(x, **kwargs)

    def fit_transform(self, x, **kwargs) -> pd.DataFrame:
        x = self.preprocess(x)
        self.fit(x, _preprocessed=True, **kwargs)
        return self.transform(x, _preprocessed=True, **kwargs)


class BinarizedFeature(BeamFeature):

    @wraps(BeamFeature.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, kind=FeaturesCategories.categorical, **kwargs)
        from sklearn.preprocessing import MultiLabelBinarizer
        self.encoder = MultiLabelBinarizer()

    def _fit(self, x, **kwargs):
        self.encoder.fit(x.values)

    def _transform(self, x, **kwargs):

        v = self.encoder.transform(x.values)
        # Create a DataFrame with the binary indicator columns
        df = pd.DataFrame(v, columns=self.encoder.classes_, index=x.index)
        return df


class DiscretizedFeature(BeamFeature):

    @wraps(BeamFeature.__init__)
    def __init__(self, *args, n_bins: int = None, strategy='quantile', subsample=None, **kwargs):
        super().__init__(*args, kind=FeaturesCategories.categorical, **kwargs)
        self.n_bins = n_bins
        self.strategy = strategy
        from sklearn.preprocessing import KBinsDiscretizer
        self.encoder = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', subsample=subsample,
                                        strategy=self.strategy)

    def _fit(self, x, **kwargs):
        self.encoder.fit(as_numpy(x))

    def _transform(self, x, **kwargs):

        v = self.encoder.transform(x.values)
        # Create a DataFrame with the binary indicator columns
        df = pd.DataFrame(v, columns=x.columns, index=x.index)
        # df = (df * self.quantiles).astype(int) + 1
        df = df.astype(int) + 1
        return df


class ScalingFeature(BeamFeature):

    @wraps(BeamFeature.__init__)
    def __init__(self, *args, method='standard', **kwargs):
        super().__init__(*args, kind=FeaturesCategories.numerical, **kwargs)
        self.method = method
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        if method == 'standard':
            self.encoder = StandardScaler()
        elif method == 'minmax':
            self.encoder = MinMaxScaler()
        elif method == 'robust':
            self.encoder = RobustScaler()
        else:
                raise ValueError(f"Invalid scaling method: {method}")

    @property
    def parameters_schema(self):
        return {'method': ParameterSchema(name='method', kind=ParameterType.categorical)}

    def _fit(self, x, **kwargs):
        self.encoder.fit(as_numpy(x))

    def _transform(self, x, index=None, columns=None):

        v = self.encoder.transform(as_numpy(x))
        # Create a DataFrame with the binary indicator columns
        df = pd.DataFrame(v, columns=x.columns, index=x.index)
        return df


class CetegorizedFeature(BeamFeature):

    @wraps(BeamFeature.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, kind=FeaturesCategories.categorical, **kwargs)
        from sklearn.preprocessing import OrdinalEncoder
        self.encoder = OrdinalEncoder()

    def _fit(self, x, **kwargs):
        self.encoder.fit(as_numpy(x))

    def _transform(self, x, **kwargs):

        v = self.encoder.transform(as_numpy(x))
        # Create a DataFrame with the binary indicator columns
        df = pd.DataFrame(v, columns=x.columns, index=x.index)
        return df


class InverseOneHotFeature(BeamFeature):

    @wraps(BeamFeature.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, kind=FeaturesCategories.categorical, **kwargs)

    def _transform(self, x, **kwargs):

        if len(self.columns) == 1:
            column = self.columns[0]
        else:
            column = self.name

        v = np.argmax(as_numpy(x), axis=1, keepdims=True)
        return pd.DataFrame(v, index=x.index, columns=[column])


# class FeaturesAggregator(BeamBase):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, name=name, **kwargs)
#         self.features = features or []
