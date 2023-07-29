import torch
from .algorithm import Algorithm
from .experiment import Experiment
from .utils import as_numpy, as_tensor
from .data import BeamData
from .config import BeamHparams, boolean_feature
import torch.nn.functional as F
from torch import nn
from torch import distributions
from .dataset import UniversalDataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from .logger import beam_logger as logger


class TabularHparams(BeamHparams):
    def add(self, parser):

        parser.set_defaults(project_name='deep_tabular', algorithm='TabularNet', n_epochs=100, scheduler='one_cycle',
                            batch_size=512, objective='acc', lr_dense=2e-3, lr_sparse=2e-2,
                            early_stopping_patience=16)

        parser.add_argument('--emb_dim', type=int, default=128, metavar='hparam', help='latent embedding dimension')
        parser.add_argument('--n_transformer_head', type=int, default=4, metavar='hparam',
                            help='number of transformer heads')
        parser.add_argument('--n_encoder_layers', type=int, default=4, metavar='hparam', help='number of encoder layers')
        parser.add_argument('--n_decoder_layers', type=int, default=4, metavar='hparam', help='number of decoder layers')
        parser.add_argument('--transformer_hidden_dim', type=int, default=256, metavar='hparam',
                            help='transformer hidden dimension')
        parser.add_argument('--transformer_dropout', type=float, default=0., metavar='hparam', help='transformer dropout')
        parser.add_argument('--mask_rate', type=float, default=0.15, metavar='hparam',
                            help='rate of masked features during training')
        parser.add_argument('--maximal_mask_rate', type=float, default=0.2, metavar='hparam',
                            help='the maximal mask rate with dynamic masking')
        parser.add_argument('--minimal_mask_rate', type=float, default=0.1, metavar='hparam',
                            help='the minimal mask rate with dynamic masking')
        parser.add_argument('--dynamic_delta', type=float, default=0.005, metavar='hparam',
                            help='the incremental delta for dynamic masking')
        parser.add_argument('--n_rules', type=int, default=64, metavar='hparam',
                            help='number of transformers rules in the decoder')
        parser.add_argument('--activation', type=str, default='gelu', metavar='hparam', help='transformer activation')
        parser.add_argument('--n_quantiles', type=int, default=10, metavar='hparam',
                            help='number of quantiles for the quantile embeddings')
        parser.add_argument('--scaler', type=str, default='quantile', metavar='hparam',
                            help='scaler for the preprocessing [robust, quantile]')
        parser.add_argument('--dataset_name', type=str, default='covtype', metavar='hparam',
                            help='dataset name [year, california_housing, higgs_small, covtype, aloi, adult, epsilon, '
                                 'microsoft, yahoo, helena, jannis]')
        parser.add_argument('--n_ensembles', type=int, default=32, metavar='hparam',
                            help='number of ensembles of the model for prediction in inference mode')
        parser.add_argument('--label_smoothing', type=float, default=0., metavar='hparam',
                            help='label smoothing for the cross entropy loss')
        parser.add_argument('--dropout', type=float, default=.0, metavar='hparam', help='Output layer dropout of the model')

        boolean_feature(parser, "oh_to_cat", False, "Try to convert one-hot encoded categorical features to "
                                                    "categorical features")
        boolean_feature(parser, "store_data_on_device", True, "Store the data on the device (GPU/CPU) in advance")
        boolean_feature(parser, 'dynamic_masking', False, 'Use dynamic masking scheduling')

        boolean_feature(parser, 'catboost', False, 'Train a catboost model on the data')
        boolean_feature(parser, 'rulenet', True, 'Train our RuleNet model on the data')

        return parser


class TabularDataset(UniversalDataset):

    def __init__(self, hparams):

        bd = BeamData.from_path(hparams.path_to_data)
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

            self.y_mu = mu
            self.y_sigma = sigma

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


class TabularTransformer(torch.nn.Module):

    def __init__(self, hparams, n_classes, n_tokens, cat_mask):
        """

        @param hparams: hyperparameters
        @param n_classes:
        @param n_tokens:
        @param cat_mask:
        """
        super().__init__()

        self.register_buffer('n_tokens', n_tokens.unsqueeze(0))
        n_tokens = n_tokens + 1  # add masking token
        tokens_offset = n_tokens.cumsum(0) - n_tokens
        total_tokens = n_tokens.sum()

        self.register_buffer('tokens_offset', tokens_offset.unsqueeze(0))
        self.register_buffer('cat_mask', cat_mask.unsqueeze(0))
        self.emb = nn.Embedding(total_tokens, hparams.emb_dim, sparse=True)

        n_rules = hparams.n_rules

        self.rules = nn.Parameter(torch.randn(1, n_rules, hparams.emb_dim))
        self.mask = distributions.Bernoulli(1 - hparams.mask_rate)

        self.transformer = nn.Transformer(d_model=hparams.emb_dim, nhead=hparams.n_transformer_head,
                                          num_encoder_layers=hparams.n_encoder_layers,
                                          num_decoder_layers=hparams.n_decoder_layers,
                                          dim_feedforward=hparams.transformer_hidden_dim,
                                          dropout=hparams.transformer_dropout,
                                          activation=hparams.activation, layer_norm_eps=1e-05,
                                          batch_first=True, norm_first=True)

        self.lin = nn.Sequential(nn.ReLU(), nn.Dropout(hparams.dropout), nn.LayerNorm(hparams.emb_dim),
            nn.Linear(hparams.emb_dim, n_classes, bias=False))

    def forward(self, sample):

        x, x_frac = sample['x'], sample['x_frac']

        x1 = (x + 1)
        x2 = torch.minimum(x + 2, self.n_tokens)

        if self.training:
            mask = self.mask.sample(x.shape).to(x.device).long()
            x1 = x1 * mask
            x2 = x2 * mask

        x1 = x1 + self.tokens_offset
        x2 = x2 + self.tokens_offset

        x1 = self.emb(x1)
        x2 = self.emb(x2)
        x_frac = x_frac.unsqueeze(-1)
        x = (1 - x_frac) * x1 + x_frac * x2

        x = self.transformer(x, torch.repeat_interleave(self.rules, len(x), dim=0))
        x = self.lin(x.max(dim=1).values)

        x = x.squeeze(-1)
        return x


class DeepTabularAlg(Algorithm):

    def __init__(self, hparams, networks=None, **kwargs):
        # choose your network
        super().__init__(hparams, networks=networks, **kwargs)
        self.loss_function = None
        self.loss_kwargs = None
        self.task_type = None
        self.train_acc = None
        self.previous_masking = self.get_hparam('mask_rate')
        self.best_masking = self.get_hparam('mask_rate')

    def preprocess_epoch(self, results=None, epoch=None, subset=None, training=True, **kwargs):
        if epoch == 0:
            self.task_type = self.dataset.task_type

            if self.task_type == 'regression':
                self.loss_kwargs = {'reduction': 'none'}
                self.loss_function = F.mse_loss
            else:
                self.loss_kwargs = {'label_smoothing': self.get_hparam('label_smoothing'), 'reduction': 'none'}
                self.loss_function = F.cross_entropy

        if self.best_state:
            self.best_masking = self.previous_masking

        return results

    def postprocess_epoch(self, sample=None, label=None, index=None,
                          results=None, epoch=None, subset=None, training=True, **kwargs):
        if self.task_type == 'regression':
            results['scalar']['rmse'] = float(torch.sqrt(results['scalar']['mse'].mean()))
            results['scalar']['objective'] = -results['scalar']['rmse']

        if self.get_hparam('dynamic_masking'):
            if training:
                self.train_acc = float(torch.mean(results['scalar']['acc']))
            else:
                test_acc = float(torch.mean(results['scalar']['acc']))
                if test_acc > self.train_acc:
                    delta = self.get_hparam('dynamic_delta')
                else:
                    delta = -self.get_hparam('dynamic_delta')
                self.previous_masking = float(self.networks['net'].mask.probs)
                non_mask_rate = max(self.previous_masking + delta, 1. - self.get_hparam('maximal_mask_rate'))
                non_mask_rate = min(non_mask_rate, 1. - self.get_hparam('minimal_mask_rate'))
                self.networks['net'].mask = distributions.Bernoulli(non_mask_rate)

            results['scalar']['mask_rate'] = float(1 - self.networks['net'].mask.probs)

        return results

    def iteration(self, sample=None, label=None, results=None, subset=None, counter=None, index=None,
                  training=True, **kwargs):

        y = label
        net = self.networks['net']

        y_hat = net(sample)
        loss = self.loss_function(y_hat, y, **self.loss_kwargs)

        self.apply(loss, training=training, results=results)

        # add scalar measurements
        if self.task_type == 'regression':
            results['scalar']['mse'].append(as_numpy(loss.mean()) * self.dataset.y_sigma ** 2)
        else:
            results['scalar']['acc'].append(as_numpy((y_hat.argmax(1) == y).float().mean()))

        return results

    def set_best_masking(self):
        logger.info(f'Setting best masking to {self.best_masking:.3f}')
        self.networks['net'].mask = distributions.Bernoulli(self.best_masking)

    def inference(self, sample=None, label=None, results=None, subset=None, predicting=True, **kwargs):

        y = label
        net = self.networks['net']
        n_ensembles = self.get_hparam('n_ensembles')

        if n_ensembles > 1:
            net.train()
            y_hat = []
            for _ in range(n_ensembles):
                y_hat.append(net(sample))
            y_hat = torch.stack(y_hat, dim=0)
            results['predictions']['y_pred_std'].append(y_hat.std(dim=0).detach())
            y_hat = y_hat.mean(dim=0)
        else:
            y_hat = net(sample)

        # add scalar measurements
        results['predictions']['y_pred'].append(y_hat.detach())

        if not predicting:

            if self.task_type == 'regression':
                results['scalar']['mse'].append(as_numpy(F.mse_loss(y_hat, y, reduction='mean'))
                                                * self.dataset.y_sigma ** 2)
            else:
                results['scalar']['acc'].append(as_numpy((y_hat.argmax(1) == y).float().mean()))

            results['predictions']['target'].append(y)
            return {'y': y, 'y_hat': y_hat}, results

        return y_hat, results

    def postprocess_inference(self, sample=None, results=None, subset=None, predicting=True, **kwargs):

        if not predicting:

            if self.task_type == 'regression':
                results['scalar']['rmse'] = float(torch.sqrt(results['scalar']['mse'].mean()))
                results['scalar']['objective'] = -results['scalar']['rmse']

            else:

                y_pred = as_numpy(torch.argmax(results['predictions']['y_pred'], dim=1))
                y_true = as_numpy(results['predictions']['target'])
                precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)
                results['metrics']['precision'] = precision
                results['metrics']['recall'] = recall
                results['metrics']['fscore'] = fscore
                results['metrics']['support'] = support
                results['scalar']['objective'] = results['scalar']['acc'].mean()

        return results
