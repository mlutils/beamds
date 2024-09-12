from functools import wraps

import numpy as np
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from ..utils import as_numpy, check_type, as_list
from ..base import BeamBase
from ..config import BeamConfig
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
    choices: list[str | int | float | bool] | None = None
    start: float | None = None
    end: float | None = None
    dtype: type | None = None
    n_steps: int | None = None
    endpoint: bool | None = True
    default: float | None = None
    description: str | None = None


class BeamFeature(BeamBase):

    def __init__(self, name, *args, func=None, columns: str | list[str] = None, kind=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.func = func
        self.columns = [columns] if isinstance(columns, str) else columns
        self.kind = kind or FeaturesCategories.numerical
        self.my_hparams = BeamConfig({k.removeprefix(f"{name}-"): v for k, v in self.hparams.dict().items()
                                      if k.startswith(f"{name}-")})

    def get_hparam(self, hparam, default=None, specific=None):
        hparam = hparam.replace('-', '_')
        v = self.my_hparams.get(hparam, specific=specific)
        if v is None:
            v = self.hparams.get(hparam, specific=specific)
        if v is None:
            if hparam in self.parameters_schema:
                v = self.parameters_schema[hparam].default
        if v is None:
            v = default
        return v

    def add_parameters_to_study(self, study):
        for k, v in self.parameters_schema.items():

            study.add_parameter(k, func=v.kind, **{kk: vv for kk, vv in
                                                   v.__dict__.items() if vv not in [None, 'name', 'kind', 'default',
                                                                                    'description']})

    @property
    def parameters_schema(self):
        d = {'enabled': ParameterSchema(name='enabled',
                                           kind=ParameterType.categorical,
                                           choices=[True, False],
                                           default=True, description='Enable/Disable feature')}

        if self.columns is not None:
            for c in self.columns:
                d[f'{c}-column-enabled'] = ParameterSchema(name=f'{c}-enabled',
                                                    kind=ParameterType.categorical,
                                                    choices=[True, False],
                                                    default=True, description=f'Enable/Disable column {c}')

        return d

    @property
    def enabled(self):
        return self.get_hparam('enabled', default=True)

    @property
    def enabled_columns(self):
        if self.columns is None:
            return None
        return [c for c in self.columns if self.get_hparam(f'{c}-enabled', default=True)]

    def preprocess(self, x):
        x_type = check_type(x)
        # if x_type.is_dataframe:
        #     if self.columns is None:
        #         self.columns = x.columns
        #     x = x[self.columns]
        # else:
        #     # TODO: build this logic
        #     x = pd.DataFrame(x, columns=self.columns)
        if not x_type.is_dataframe:
            # TODO: build this logic
            x = pd.DataFrame(x, columns=self.columns)

        return x

    def transform(self, x, _preprocessed=False, **kwargs) -> pd.DataFrame:
        if not self.enabled:
            return pd.DataFrame(index=x.index)
        if not _preprocessed:
            x = self.preprocess(x)
        y = self._transform(x, **kwargs)
        c = self.enabled_columns
        if c is not None:
            y = y[c]
        return y

    def _transform(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self.func(x)

    def _fit(self, x: pd.DataFrame, **kwargs):
        pass

    def fit(self, x: pd.DataFrame, _preprocessed=False, **kwargs):
        if self.enabled:
            if not _preprocessed:
                x = self.preprocess(x)
            self._fit(x, **kwargs)

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
        self.encoder.fit(x.squeeze().values)

    def _transform(self, x, **kwargs):

        v = self.encoder.transform(x.squeeze().values)
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

        if self.columns is not None and len(self.columns) == 1:
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
