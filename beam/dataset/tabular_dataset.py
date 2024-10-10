import pandas as pd

from .universal_dataset import UniversalDataset
from ..type import Types
from ..utils import as_numpy, check_type, as_list


class TabularDataset(UniversalDataset):

    def __init__(self, x=None, y=None, x_train=None, y_train=None, x_validation=None, y_validation=None,
                 x_test=None, y_test=None, cat_features=None, text_features=None, embedding_features=None,
                 sample_weight=None, validation=.25, test=None, seed=5782, stratify=False,
                 test_split_method='uniform', time_index=None, window=None, columns=None, **kwargs):

        self.columns = as_list(columns)
        self.sample_weight = sample_weight
        if x is not None:
            assert y is not None, "y must be provided if x is provided"
            super().__init__(x, label=y, to_torch=False, **kwargs)

            self.split(validation=validation, test=test, seed=seed, stratify=stratify,
                       test_split_method=test_split_method, time_index=time_index, window=window)

            self.unified = True
            self.eval_subset = 'validation' if validation else 'test'
            x_type = check_type(x)
            if x_type.is_dataframe:
                if self.columns is None:
                    self.columns = as_list(x.columns)

        else:
            super().__init__(x_train=x_train, y_train=y_train, x_validation=x_validation,
                             x_test=x_test, y_validation=y_validation, y_test=y_test, to_torch=False, **kwargs)

            self.unified = False
            self.eval_subset = 'validation' if x_validation is not None else 'test'
            x_type = check_type(x_train)
            if x_type.is_dataframe:
                if self.columns is None:
                    self.columns = as_list(x_train.columns)

        if self.columns is not None:
            columns = self.columns
            if cat_features is not None:
                cat_features_type = check_type(cat_features)
                if cat_features_type.element in [Types.str, Types.object]:
                    cat_features = [columns.index(c) for c in cat_features if c in columns]
                elif cat_features_type.is_array:
                    cat_features = as_numpy(cat_features)

            if text_features is not None:
                text_features_type = check_type(text_features)
                if text_features_type.element in [Types.str, Types.object]:
                    text_features = [columns.index(c) for c in text_features if c in columns]
                elif text_features_type.is_array:
                    text_features = as_numpy(text_features)

            if embedding_features is not None:
                embedding_features_type = check_type(embedding_features)
                if embedding_features_type.element in [Types.str, Types.object]:
                    embedding_features = [columns.index(c) for c in embedding_features if c in columns]
                elif embedding_features_type.is_array:
                    embedding_features = as_numpy(embedding_features)

            self.cat_features = cat_features
            self.text_features = text_features
            self.embedding_features = embedding_features

    @property
    def cat_columns(self):
        return [self.columns[i] for i in self.cat_features] if self.cat_features else None

    @property
    def text_columns(self):
        return [self.columns[i] for i in self.text_features] if self.text_features else None

    @property
    def embedding_columns(self):
        return [self.columns[i] for i in self.embedding_features] if self.embedding_features else None

    @property
    def train_pool(self):
        x, y = self.get_subset_data('train')
        return self.pool(x, y)

    @property
    def validation_pool(self):
        x, y = self.get_subset_data('validation')
        return self.pool(x, y)

    @property
    def eval_pool(self):
        x, y = self.get_subset_data(self.eval_subset)
        return self.pool(x, y)

    @property
    def test_pool(self):
        x, y = self.get_subset_data('test')
        return self.pool(x, y)

    def get_subset_data(self, subset):
        if self.unified:
            batch = self[self.indices[subset]]
            x = batch.data
            y = batch.label
        else:
            x = self.data[f'x_{subset}']
            y = self.data[f'y_{subset}']
        return x, y

    def pool(self, x, y):
        from catboost import Pool

        x_type = check_type(x)
        columns = self.columns
        if not x_type.is_dataframe:
            x = pd.DataFrame(x, columns=columns)

        return Pool(x, as_numpy(y), cat_features=self.cat_features,
                    text_features=self.text_features, embedding_features=self.embedding_features)
