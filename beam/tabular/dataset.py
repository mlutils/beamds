import torch
from ..utils import as_numpy, as_tensor
from ..data import BeamData
from ..dataset import UniversalDataset
import numpy as np
import pandas as pd
from ..logging import beam_logger as logger


class TabularDataset(UniversalDataset):

    def __init__(self, hparams):

        bd = BeamData.from_path(hparams.data_path)
        dataset = bd[hparams.dataset_name].cached()
        info = dataset['info'].values
        self.task_type = info['task_type']

        x_train = dataset['N_train'].values
        x_val = dataset['N_val'].values
        x_test = dataset['N_test'].values

        if np.isnan(x_train).any() or np.isnan(x_val).any() or np.isnan(x_test).any():
            logger.warning('NaN values in the data, replacing with 0')
            x_train = np.nan_to_num(x_train)
            x_val = np.nan_to_num(x_val)
            x_test = np.nan_to_num(x_test)

        y_train = dataset['y_train'].values

        self.numerical_features, self.cat_features = self.get_numerical_and_categorical(x_train, y_train)

        x_train_num = x_train[:, self.numerical_features]
        x_train_cat = x_train[:, self.cat_features].astype(np.int64)

        x_val_num = x_val[:, self.numerical_features]
        x_val_cat = x_val[:, self.cat_features].astype(np.int64)

        x_test_num = x_test[:, self.numerical_features]
        x_test_cat = x_test[:, self.cat_features].astype(np.int64)

        if hparams.oh_to_cat:
            self.oh_categories = self.one_hot_to_categorical(x_train_cat)

            x_val_cat = np.stack([x_val_cat.T[self.oh_categories == c].argmax(axis=0)
                                  for c in np.unique(self.oh_categories)], axis=1)
            x_train_cat = np.stack([x_train_cat.T[self.oh_categories == c].argmax(axis=0)
                                    for c in np.unique(self.oh_categories)], axis=1)
            x_test_cat = np.stack([x_test_cat.T[self.oh_categories == c].argmax(axis=0)
                                   for c in np.unique(self.oh_categories)], axis=1)

        if info['n_cat_features'] > 0:

            d = dataset['C_trainval'].values
            factors = [pd.factorize(d[:, i])[1] for i in range(d.shape[1])]

            d = dataset['C_train'].values
            x_train_cat_aux = np.stack([pd.Categorical(d[:, i], categories=f).codes
                                        for i, f in enumerate(factors)], axis=1).astype(np.int64)
            d = dataset['C_val'].values
            x_val_cat_aux = np.stack([pd.Categorical(d[:, i], categories=f).codes
                                      for i, f in enumerate(factors)], axis=1).astype(np.int64)
            d = dataset['C_test'].values
            x_test_cat_aux = np.stack([pd.Categorical(d[:, i], categories=f).codes
                                        for i, f in enumerate(factors)], axis=1).astype(np.int64)

            # plus 1 for nan values
            x_train_cat = np.concatenate([x_train_cat, x_train_cat_aux+1], axis=1)
            x_val_cat = np.concatenate([x_val_cat, x_val_cat_aux+1], axis=1)
            x_test_cat = np.concatenate([x_test_cat, x_test_cat_aux+1], axis=1)

        if hparams.scaler == 'robust':
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        elif hparams.scaler == 'quantile':
            from sklearn.preprocessing import QuantileTransformer
            self.scaler = QuantileTransformer(n_quantiles=1000, subsample=100000, random_state=hparams.seed)
        else:
            raise ValueError('Unknown scaler')

        self.scaler.fit(x_train_num)

        x_train_num_scaled = torch.FloatTensor(self.scaler.transform(x_train_num))
        x_val_num_scaled = torch.FloatTensor(self.scaler.transform(x_val_num))
        x_test_num_scaled = torch.FloatTensor(self.scaler.transform(x_test_num))

        # save these tables for catboost training
        self.x_train_num_scaled = as_numpy(x_train_num_scaled)
        self.x_val_num_scaled = as_numpy(x_val_num_scaled)
        self.x_test_num_scaled = as_numpy(x_test_num_scaled)
        self.x_train_cat = x_train_cat
        self.x_val_cat = x_val_cat
        self.x_test_cat = x_test_cat
        self.y_train = dataset['y_train'].values
        self.y_val = dataset['y_val'].values
        self.y_test = dataset['y_test'].values

        self.y_mu = None
        self.y_sigma = None
        if self.task_type == 'regression':
            y_train = torch.FloatTensor(dataset['y_train'].values)
            y_val = torch.FloatTensor(dataset['y_val'].values)
            y_test = torch.FloatTensor(dataset['y_test'].values)

            mu = y_train.mean(dim=0, keepdim=True)
            sigma = y_train.std(dim=0, keepdim=True)

            self.y_mu = float(mu)
            self.y_sigma = float(sigma)

            y_train = (y_train - mu) / (sigma + 1e-8)
            y_val = (y_val - mu) / (sigma + 1e-8)
            y_test = (y_test - mu) / (sigma + 1e-8)

        else:
            y_train = torch.LongTensor(dataset['y_train'].values)
            y_val = torch.LongTensor(dataset['y_val'].values)
            y_test = torch.LongTensor(dataset['y_test'].values)

        n_quantiles = hparams.n_quantiles
        x_train_num_quantized = (x_train_num_scaled * n_quantiles).long()
        x_val_num_quantized = (x_val_num_scaled * n_quantiles).long()
        x_test_num_quantized = (x_test_num_scaled * n_quantiles).long()

        x_train_num_fractional = x_train_num_scaled * n_quantiles - x_train_num_quantized.float()
        x_val_num_fractional = x_val_num_scaled * n_quantiles - x_val_num_quantized.float()
        x_test_num_fractional = x_test_num_scaled * n_quantiles - x_test_num_quantized.float()

        self.cat_mask = torch.cat([torch.ones(x_train_num_quantized.shape[-1]), torch.zeros(x_train_cat.shape[-1])])

        x_train_mixed = torch.cat([x_train_num_quantized, as_tensor(x_train_cat)], dim=1)
        x_val_mixed = torch.cat([x_val_num_quantized, as_tensor(x_val_cat)], dim=1)
        x_test_mixed = torch.cat([x_test_num_quantized, as_tensor(x_test_cat)], dim=1)

        self.n_tokens = torch.stack([xi.max(dim=0).values
                                     for xi in [x_train_mixed, x_val_mixed, x_test_mixed]]).max(dim=0).values + 1

        x_train_frac = torch.cat([x_train_num_fractional, torch.zeros(x_train_cat.shape)], dim=1)
        x_val_frac = torch.cat([x_val_num_fractional, torch.zeros(x_val_cat.shape)], dim=1)
        x_test_frac = torch.cat([x_test_num_fractional, torch.zeros(x_test_cat.shape)], dim=1)

        x = torch.cat([x_train_mixed, x_val_mixed, x_test_mixed], dim=0)
        x_frac = torch.cat([x_train_frac, x_val_frac, x_test_frac], dim=0)
        y = torch.cat([y_train, y_val, y_test], dim=0)

        device = None
        if hparams.store_data_on_device:
            device = hparams.device

        super().__init__(x=x, x_frac=x_frac, label=y, device=device)

        if self.task_type == 'regression':
            self.n_classes = 1
        else:
            self.n_classes = self.label.max() + 1

        # logger.warning("Check x-max consistency at dataset:")
        # logger.warning(f"n-tokens: {self.n_tokens}")
        # logger.warning(self.n_tokens < x.max(dim=0).values)
        self.split(validation=len(x_train_mixed) + np.arange(len(x_val_mixed)),
                           test=len(x_train_mixed) + len(x_val_mixed) + np.arange(len(x_test_mixed)))

    @staticmethod
    def get_numerical_and_categorical(x, y=None):
        """
        @param x: input data
        @return: numerical and categorical features
        """
        import deepchecks as dch
        dataset = dch.tabular.Dataset(x, label=y)

        return dataset.numerical_features, dataset.cat_features

    @staticmethod
    def one_hot_to_categorical(x):
        """
        @param x: one-hot encoded categorical features
        @return: mapping from one-hot to categorical
        """
        return x.cumsum(axis=1).max(axis=0)

