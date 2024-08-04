import pandas as pd

from ..base import BeamBase
from ..type.utils import is_pandas_dataframe, is_pandas_series


class FeaturesCategories(Enum):
    numerical = 'numerical'
    categorical = 'categorical'
    embedding = 'embedding'
    text = 'text'


class BeamFeature(BeamBase):

    def __init__(self, *args, func=None, name=None, kind=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.func = func
        self.kind = kind or FeaturesCategories.numerical

    def transform(self, x, index=None) -> pd.DataFrame:
        v = self.func(x)
        return pd.DataFrame(v, index=index)

    def fit(self, x):
        pass

    def fit_transform(self, x, index=None) -> pd.DataFrame:
        self.fit(x)
        return self.transform(x, index)


class BinarizedFeature(BeamFeature):

    def __init__(self, *args, column: str = None, name=None, **kwargs):
        super().__init__(*args, name=name, kind=FeaturesCategories.categorical, **kwargs)
        from sklearn.preprocessing import MultiLabelBinarizer
        self.encoder = MultiLabelBinarizer()
        self.column = column

    def fit(self, x):
        if is_pandas_dataframe(x):
            x = x[self.column].values
        self.encoder.fit(x)

    def transform(self, x, index=None):
        if is_pandas_dataframe(x):
            x = x[self.column].values
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
        self.n_bins = quantiles
        self.strategy = strategy
        self.columns = [columns] if isinstance(columns, str) else columns
        from sklearn.preprocessing import KBinsDiscretizer
        self.encoder = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', subsample=subsample,
                                        strategy=self.strategy)

    def fit(self, x):
        if is_pandas_dataframe(x):
            x = x[self.column].values
        self.encoder.fit(x)

    def transform(self, x, index=None):
        if is_pandas_dataframe(x):
            x = x[self.column].values
            index = index or x.index
        elif is_pandas_series(x):
            x = x.values
            index = index or x.index

        v = self.encoder.transform(x)
        # Create a DataFrame with the binary indicator columns
        df = pd.DataFrame(v, columns=self.columns, index=index)
        # df = (df * self.quantiles).astype(int) + 1
        df = df.astype(int) + 1
        return df


class ScalingFeature(BeamFeature):

    def __init__(self, *args, columns: list[str] | str = None, method='standard', name=None, **kwargs):
        super().__init__(*args, name=name, kind=FeaturesCategories.numerical, **kwargs)
        self.columns = [columns] if isinstance(columns, str) else columns
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

    def fit(self, x):
        if is_pandas_dataframe(x):
            x = x[self.columns].values
        self.encoder.fit(x)

    def transform(self, x, index=None):
        if is_pandas_dataframe(x):
            x = x[self.columns].values
            index = index or x.index
        elif is_pandas_series(x):
            x = x.values
            index = index or x.index

        v = self.encoder.transform(x)
        # Create a DataFrame with the binary indicator columns
        df = pd.DataFrame(v, columns=self.columns, index=index)
        return df


class CetegorizedFeature(BeamFeature):

    def __init__(self, *args, columns: list[str] | str = None, name=None, **kwargs):
        super().__init__(*args, name=name, kind=FeaturesCategories.categorical, **kwargs)
        self.columns = [columns] if isinstance(columns, str) else columns
        from sklearn.preprocessing import OrdinalEncoder
        self.encoder = OrdinalEncoder()

    def fit(self, x):
        if is_pandas_dataframe(x):
            x = x[self.columns].values
        self.encoder.fit(x)

    def transform(self, x, index=None):
        if is_pandas_dataframe(x):
            x = x[self.columns].values
            index = index or x.index
        elif is_pandas_series(x):
            x = x.values
            index = index or x.index

        v = self.encoder.transform(x)
        # Create a DataFrame with the binary indicator columns
        df = pd.DataFrame(v, columns=self.columns, index=index)
        return df


class FeaturesAggregator(BeamBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.features = features or []
