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

class TabularHparams(BeamHparams):
    def add(self, parser):

        parser.set_defaults(project_name='deep_tabular', algorithm='TabularNet', n_epochs=100, scheduler='one_cycle',
                            batch_size=512, objective='acc', lr_dense=2e-3, lr_sparse=2e-2)

        parser.add_argument('--emb_dim', type=int, default=128, metavar='hparam', help='latent embedding dimension')
        parser.add_argument('--n_transformer_head', type=int, default=4, metavar='hparam',
                            help='number of transformer heads')
        parser.add_argument('--n_encoder_layers', type=int, default=4, metavar='hparam', help='number of encoder layers')
        parser.add_argument('--n_decoder_layers', type=int, default=4, metavar='hparam', help='number of decoder layers')
        parser.add_argument('--transformer_hidden_dim', type=int, default=256, metavar='hparam',
                            help='transformer hidden dimension')
        parser.add_argument('--transformer_dropout', type=float, default=0., metavar='hparam', help='transformer dropout')
        parser.add_argument('--features_mask_rate', type=float, default=0.15, metavar='hparam',
                            help='rate of masked features during training')
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

        boolean_feature(parser, "oh_to_cat", False, "Try to convert one-hot encoded categorical features to "
                                                    "categorical features")

        return parser

class TabularDataset(UniversalDataset):

    def __init__(self, hparams):

        bd = BeamData.from_path(hparams.path_to_data)
        dataset = bd[hparams.dataset_name].cached()

        x_train = dataset['N_train'].values
        x_val = dataset['N_val'].values
        x_test = dataset['N_test'].values

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

        super().__init__(x=x, x_frac=x_frac, label=y)

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
        self.mask = distributions.Bernoulli(1 - hparams.features_mask_rate)

        self.transformer = nn.Transformer(d_model=hparams.emb_dim, nhead=hparams.n_transformer_head,
                                          num_encoder_layers=hparams.n_encoder_layers,
                                          num_decoder_layers=hparams.n_decoder_layers,
                                          dim_feedforward=hparams.transformer_hidden_dim,
                                          dropout=hparams.transformer_dropout,
                                          activation=hparams.activation, layer_norm_eps=1e-05,
                                          batch_first=True, norm_first=True)

        self.lin = nn.Linear(hparams.emb_dim, n_classes, bias=False)

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
        return x


class DeepTabularAlg(Algorithm):

    def __init__(self, hparams, networks=None, **kwargs):
        # choose your network
        super().__init__(hparams, networks=networks, **kwargs)

    def iteration(self, sample=None, label=None, results=None, subset=None, counter=None, training=True, **kwargs):

        y = label
        net = self.networks['net']

        y_hat = net(sample)
        loss = F.cross_entropy(y_hat, y, reduction='none')

        self.apply(loss, training=training, results=results)

        # add scalar measurements
        results['scalar']['acc'].append(as_numpy((y_hat.argmax(1) == y).float().mean()))

        return results

    def inference(self, sample=None, label=None, results=None, subset=None, predicting=True, **kwargs):

        y = label
        net = self.networks['net']
        y_hat = net(sample)

        # add scalar measurements
        results['predictions']['y_pred'].append(y_hat.detach())

        if not predicting:
            results['scalar']['acc'].append(float((y_hat.argmax(1) == y).float().mean()))
            results['predictions']['target'].append(y)
            return {'y': y, 'y_hat': y_hat}, results

        return y_hat, results

    def postprocess_inference(self, sample=None, results=None, subset=None, predicting=True, **kwargs):

        y_pred = as_numpy(torch.argmax(results['predictions']['y_pred'], dim=1))

        if not predicting:
            y_true = as_numpy(results['predictions']['target'])
            precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)
            results['metrics']['precision'] = precision
            results['metrics']['recall'] = recall
            results['metrics']['fscore'] = fscore
            results['metrics']['support'] = support

        return results
