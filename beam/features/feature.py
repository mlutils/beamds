from collections import defaultdict
from functools import wraps, cached_property
from typing import Union, List, Set

import numpy as np
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from .. import beam_path
from ..type import check_type, Types
from ..utils import as_numpy, as_dataframe, as_list, identity_function, return_constant
from ..processor import Processor
from ..config import BeamConfig
from ..logging import beam_logger as logger
from ..data import BeamData


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
    endpoint: bool | None = None
    default: float | None = None
    description: str | None = None


class BeamFeature(Processor):

    def __init__(self, name, *args, func=None,
                 input_columns: int | str | list[str | int] | range | slice = None,
                 output_columns: str | list[str] = None,
                 kind=None, n_input_columns: int = None, add_name_prefix=False,
                 prefer_name=True,
                 input_columns_blacklist=None,
                 output_columns_blacklist=None,
                 **kwargs):
        self.my_hparams = None
        super().__init__(*args, name=name, **kwargs)
        self._is_fitted = False

        if func is None:
            func = identity_function
        self.func = func

        self._input_columns = input_columns
        self.n_input_columns = n_input_columns
        self._output_columns = output_columns
        self._schema = None
        self.add_name_prefix = add_name_prefix
        self.prefer_name = prefer_name
        self.kind = kind or FeaturesCategories.numerical
        self.my_hparams = BeamConfig({k.removeprefix(f"{name}-"): v for k, v in self.hparams.dict().items()
                                      if k.startswith(f"{name}-")}, silent=True,
                                     load_config_files=False, load_script_arguments=False)

        self.input_columns_blacklist = defaultdict(return_constant(False))
        if input_columns_blacklist is not None:
            for c in as_list(input_columns_blacklist):
                self.input_columns_blacklist[c] = True

        self.output_columns_blacklist = defaultdict(return_constant(False))
        if output_columns_blacklist is not None:
            for c in as_list(output_columns_blacklist):
                self.output_columns_blacklist[c] = True

    @property
    def is_fitted(self):
        return self._is_fitted

    @property
    def output_columns(self):
        if hasattr(self, '_output_columns') and self._output_columns is not None:
            return as_list(self._output_columns)
        return None

    def reset(self):
        self._is_fitted = False

    @property
    def input_columns(self):
        if hasattr(self, '_input_columns') and self._input_columns is not None:
            columns = self._input_columns
            columns_type = check_type(columns)
            if columns_type.major == Types.scalar:
                columns = [columns]
            else:
                columns = as_list(columns, length=self.n_input_columns)
            return columns
        return None

    def set_n_input_columns(self, x):
        n = len(x.columns)
        if self.n_input_columns is None:
            self.n_input_columns = n
        else:
            assert self.n_input_columns == n, f"Number of columns must be consistent with the feature definition," \
                                        f"expected {self.n_input_columns} but got {n}"

    def get_hparam(self, hparam, default=None, specific=None):
        hparam = hparam.replace('-', '_')
        v = None
        if self.my_hparams is not None:
            v = self.my_hparams.get(hparam, specific=specific)
        if v is None:
            v = self.hparams.get(hparam, specific=specific)
        if v is None:
            if hasattr(self, '_schema') and hparam in self.parameters_schema:
                v = self.parameters_schema[hparam].default
        if v is None:
            v = default
        return v

    def add_parameters_to_study(self, study, add_enable=True):
        for k, v in self.parameters_schema.items():
            if not add_enable and k == 'enabled':
                continue
            study.add_parameter(k, kind=v.kind.value, **{kk: vv for kk, vv in v.__dict__.items()
                                                         if vv is not None and kk not in
                                                         ['name', 'kind', 'default', 'description']})

    @property
    def basic_parameters_schema(self):
        d = {'enabled': ParameterSchema(name='enabled',
                                           kind=ParameterType.categorical,
                                           choices=[True, False],
                                           default=True, description='Enable/Disable feature')}

        if self.input_columns is not None:
            for c in self.input_columns:
                d[f'{c}-input-column-enabled'] = ParameterSchema(name=f'{c}-input-column-enabled',
                                                    kind=ParameterType.categorical,
                                                    choices=[True, False],
                                                    default=True, description=f'Enable/Disable column {c}')
        if self.output_columns is not None:
            for c in self.output_columns:
                d[f'{c}-output-column-enabled'] = ParameterSchema(name=f'{c}-output-column-enabled',
                                                    kind=ParameterType.categorical,
                                                    choices=[True, False],
                                                    default=True, description=f'Enable/Disable column {c}')

        return d

    def add_to_schema(self, name, kind, **kwargs):
        if self._schema is None:
            self._schema = {}
        self._schema[name] = ParameterSchema(name=name, kind=kind, **kwargs)

    @property
    def parameters_schema(self):
        schema = self._schema or {}
        return self.basic_parameters_schema | schema

    @property
    def enabled(self):
        return self.get_hparam('enabled', default=True)

    @property
    def enabled_input_columns(self):
        if self.input_columns is None:
            return None
        return [c for c in self.input_columns if self.get_hparam(f'{c}-input-column-enabled',
                                                                 default=not self.input_columns_blacklist[c])]

    @property
    def enabled_column_names(self):
        if self.output_columns is None:
            return None
        return [c for c in self.output_columns if self.get_hparam(f'{c}-output-column-enabled',
                                                                  default=not self.output_columns_blacklist[c])]

    def preprocess(self, x):
        x = as_dataframe(x)
        self.set_n_input_columns(x)
        if self.input_columns is not None:
            x = x[self.input_columns]

        c = self.enabled_input_columns
        if c is not None:
            x = x.drop(columns=[col for col in x.columns if col not in c])

        return x

    def transform(self, x, _preprocessed=False, **kwargs) -> pd.DataFrame:
        if not self.enabled:
            return pd.DataFrame(index=x.index)
        if not _preprocessed:
            x = self.preprocess(x)
        y = self.transform_callback(x, **kwargs)
        if self.output_columns is not None:
            y.columns = self.output_columns
        if self.enabled_column_names is not None:
            y = y.drop(columns=[col for col in y.columns if col not in self.enabled_column_names])
        if len(y.columns) == 1:
            if self.prefer_name and self.name is not None:
                y.columns = [self.name]
        else:
            if self.add_name_prefix and self.name is not None:
                y.columns = [f"{self.name}-{c}" for c in y.columns]
        return y

    def transform_callback(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self.func(x)

    def fit_callback(self, x: pd.DataFrame, **kwargs):
        pass

    def fit(self, x: pd.DataFrame, _preprocessed=False, **kwargs):
        if self.enabled:
            if not _preprocessed:
                x = self.preprocess(x)

            if self.is_fitted:
                logger.warning(f"Feature {self.name} is already fitted, skipping (use reset() to re-fit)")

            self.fit_callback(x, **kwargs)
        self._is_fitted = True

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

    def fit_callback(self, x, **kwargs):
        self.encoder.fit(x.squeeze().values)

    def transform_callback(self, x, **kwargs):

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

    def fit_callback(self, x, **kwargs):
        self.encoder.fit(as_numpy(x))

    def transform_callback(self, x, **kwargs):

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

        self.add_to_schema('method', ParameterType.categorical, choices=['standard', 'minmax', 'robust'],
                           default='standard', description='Scaling method')

    def fit_callback(self, x, **kwargs):
        self.encoder.fit(as_numpy(x))

    def transform_callback(self, x, **kwargs):

        v = self.encoder.transform(as_numpy(x))
        # Create a DataFrame with the binary indicator columns
        df = pd.DataFrame(v, columns=x.columns, index=x.index)
        return df


class CategorizedFeature(BeamFeature):

    @wraps(BeamFeature.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, kind=FeaturesCategories.categorical, **kwargs)
        from sklearn.preprocessing import OrdinalEncoder
        self.encoder = OrdinalEncoder()

    def fit_callback(self, x, **kwargs):
        self.encoder.fit(as_numpy(x))

    def transform_callback(self, x, **kwargs):

        v = self.encoder.transform(as_numpy(x))
        # Create a DataFrame with the binary indicator columns
        df = pd.DataFrame(v, columns=x.columns, index=x.index)
        return df


class InverseOneHotFeature(BeamFeature):

    @wraps(BeamFeature.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, kind=FeaturesCategories.categorical, **kwargs)

    def transform_callback(self, x, **kwargs):

        if self.input_columns is not None and len(self.input_columns) == 1:
            column = self.input_columns[0]
        else:
            column = self.name

        v = np.argmax(as_numpy(x), axis=1, keepdims=True)
        return pd.DataFrame(v, index=x.index, columns=[column])


class FeaturesAggregator(Processor):

    def __init__(self, *features, state_path=None, artifact_path=None, save_intermediate_results=True, **kwargs):
        super().__init__(**kwargs)
        self.features = features or []
        self.state_path = state_path
        self.artifact_path = artifact_path
        self.save_intermediate_results = save_intermediate_results

    @property
    def parameters_schema(self):
        d = {}
        for f in self.features:
            d.update(f.parameters_schema)
        return d

    def add_parameters_to_study(self, study):
        for f in self.features:
            f.add_parameters_to_study(study)

    def fit(self, x, **kwargs):
        for f in self.features:
            f.fit(x, **kwargs)

    @classmethod
    @property
    def excluded_attributes(cls) -> set[str]:
        return super().excluded_attributes | {'features'}

    def save_state_dict(self, state, path, ext=None, exclude: Union[List, Set] = None, override=False,
                        blacklist_priority=None, **kwargs):

        super().save_state_dict(state, path, ext, exclude, **kwargs)
        path = beam_path(path)

        for i, f in enumerate(self.features):
            name = f.name or f'feature_{i}'
            p = path.joinpath(name)
            f.save_state_dict(state, p, ext=ext, exclude=exclude, override=override,
                              blacklist_priority=blacklist_priority, **kwargs)

    def load_state_dict(self, path, ext=None, exclude: Union[List, Set] = None, hparams=True, exclude_hparams=None,
                        overwrite_hparams=None, **kwargs):

        super().load_state_dict(path, ext, exclude, hparams, exclude_hparams, overwrite_hparams, **kwargs)
        path = beam_path(path)
        for i, f in enumerate(self.features):
            name = f.name or f'feature_{i}'
            p = path.joinpath(name)
            f.load_state_dict(p, ext=ext, exclude=exclude, hparams=hparams, exclude_hparams=exclude_hparams,
                              overwrite_hparams=overwrite_hparams, **kwargs)

    def transform(self, x, **kwargs):
        y = []
        for f in self.features:
            yi = f.transform(x, **kwargs)
            y.append(yi)
            if self.artifact_path is not None:
                p = beam_path(self.artifact_path).joinpath(f.name)
                BeamData.write_object(yi, p)

        return pd.concat(y, axis=1)

    def fit_transform(self, x, **kwargs):
        y = []
        for f in self.features:

            if f.is_fitted:
                logger.warning(f"Feature {f.name} is already fitted, applying transform only")
                yi = f.transform(x, **kwargs)
            else:
                yi = f.fit_transform(x, **kwargs)

                if self.save_intermediate_results and self.state_path is not None:
                    p = beam_path(self.state_path).joinpath(f.name)
                    f.save_state(p)

            if self.artifact_path is not None:
                p = beam_path(self.artifact_path).joinpath(f.name)
                BeamData.write_object(yi, p)

            y.append(yi)

        return pd.concat(y, axis=1)
