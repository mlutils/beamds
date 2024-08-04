import pandas as pd

from ..base import BeamBase
from ..type.utils import is_pandas_dataframe, is_pandas_series


class BeamFeature(BeamBase):

    def __init__(self, *args, func=None, name=None, kind=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.func = func
        self.kind = kind or 'numerical'

    def transform(self, x, index=None) -> pd.DataFrame:
        v = self.func(x)
        return pd.DataFrame(v, index=index)

    def fit(self, x):
        pass

    def fit_transform(self, x, index=None) -> pd.DataFrame:
        self.fit(x)
        return self.transform(x, index)


class BinaryFeature(BeamFeature):

    def __init__(self, *args, column=None, name=None, **kwargs):
        super().__init__(*args, column=column, name=name, kind='binary', **kwargs)
        from sklearn.preprocessing import MultiLabelBinarizer
        self.encoder = MultiLabelBinarizer()

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



