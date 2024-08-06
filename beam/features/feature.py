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

    def __init__(self, *args, func=None, name=None, kind=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.func = func
        self.kind = kind or FeaturesCategories.numerical

    @property
    def parameters_schema(self):
        return {}

    def transform(self, x, index=None) -> pd.DataFrame:
        v = self.func(x)
        return pd.DataFrame(v, index=index)

    def fit(self, x):
        pass

    def fit_transform(self, x, index=None, **kwargs) -> pd.DataFrame:
        self.fit(x)
        return self.transform(x, index)


class BinarizedFeature(BeamFeature):

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, kind=FeaturesCategories.categorical, **kwargs)
        from sklearn.preprocessing import MultiLabelBinarizer
        self.encoder = MultiLabelBinarizer()

    def fit(self, x):
        if is_pandas_dataframe(x):
            x = x.values
        self.encoder.fit(x)

    def transform(self, x, index=None):
        if is_pandas_dataframe(x):
            x = x.values
            index = index or x.index
        elif is_pandas_series(x):
            x = x.values
            index = index or x.index

        v = self.encoder.transform(x)
        # Create a DataFrame with the binary indicator columns
        df = pd.DataFrame(v, columns=self.encoder.classes_, index=index)
        return df


class DiscretizedFeature(BeamFeature):

    def __init__(self, *args, columns: list[str] | str = None, n_bins: int = None, strategy='quantile',
                 name=None, subsample=None, **kwargs):
        super().__init__(*args, name=name, kind=FeaturesCategories.categorical, **kwargs)
        self.n_bins = n_bins
        self.strategy = strategy
        self.columns = [columns] if isinstance(columns, str) else columns
        from sklearn.preprocessing import KBinsDiscretizer
        self.encoder = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', subsample=subsample,
                                        strategy=self.strategy)

    def fit(self, x):
        self.encoder.fit(as_numpy(x))

    def transform(self, x, index=None):
        columns = self.columns
        if is_pandas_dataframe(x):
            x = x.values
            index = index or x.index
            if columns is None:
                columns = as_list(x.columns)
        elif is_pandas_series(x):
            x = x.values
            index = index or x.index

        v = self.encoder.transform(x)
        # Create a DataFrame with the binary indicator columns
        df = pd.DataFrame(v, columns=columns, index=index)
        # df = (df * self.quantiles).astype(int) + 1
        df = df.astype(int) + 1
        return df


class ScalingFeature(BeamFeature):

    def __init__(self, *args, columns: list[str] | str = None, method='standard', name=None, **kwargs):
        super().__init__(*args, name=name, kind=FeaturesCategories.numerical, **kwargs)
        self.method = method
        self.columns = [columns] if isinstance(columns, str) else columns
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

    def fit(self, x):
        self.encoder.fit(as_numpy(x))

    def transform(self, x, index=None):
        x_type = check_type(x)
        columns = self.columns
        if x_type.is_dataframe:
            if columns is None:
                columns = as_list(x.columns)
            index = index or x.index

        v = self.encoder.transform(as_numpy(x))
        # Create a DataFrame with the binary indicator columns
        df = pd.DataFrame(v, columns=columns, index=index)
        return df


class CetegorizedFeature(BeamFeature):

    def __init__(self, *args, columns: list[str] | str = None, name=None, **kwargs):
        super().__init__(*args, name=name, kind=FeaturesCategories.categorical, **kwargs)
        self.columns = [columns] if isinstance(columns, str) else columns
        from sklearn.preprocessing import OrdinalEncoder
        self.encoder = OrdinalEncoder()

    def fit(self, x):
        self.encoder.fit(as_numpy(x))

    def transform(self, x, index=None):
        x_type = check_type(x)
        columns = self.columns
        if x_type.is_dataframe:
            if columns is None:
                columns = as_list(x.columns)
            index = index or x.index

        v = self.encoder.transform(as_numpy(x))
        # Create a DataFrame with the binary indicator columns
        df = pd.DataFrame(v, columns=columns, index=index)
        return df


class InverseOneHotFeature(BeamFeature):

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, kind=FeaturesCategories.categorical, **kwargs)

    def transform(self, x, index=None):

        column = self.name
        x_type = check_type(x)
        if x_type.is_dataframe:
            columns = x.columns
            column = columns[0]

        x = as_numpy(x)
        x = np.argmax(x, axis=1, keepdims=True)
        x = pd.DataFrame(x, index=index, columns=[column])

        return x


# class FeaturesAggregator(BeamBase):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, name=name, **kwargs)
#         self.features = features or []
