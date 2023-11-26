import torch
from ..core.algorithm import Algorithm
from ..utils import as_numpy, as_tensor
from ..data import BeamData
from ..config import BeamHparams, BeamParam
import torch.nn.functional as F
from torch import nn
from torch import distributions
from ..dataset import UniversalDataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from ..logger import beam_logger as logger


class TabularTransformer(torch.nn.Module):

    def __init__(self, hparams, n_classes, n_tokens, cat_mask):
        """

        @param hparams: hyperparameters
        @param n_classes:
        @param n_tokens:
        @param cat_mask:
        """
        super().__init__()

        n_tokens = as_tensor(n_tokens)
        cat_mask = as_tensor(cat_mask)
        self.register_buffer('n_tokens', n_tokens.unsqueeze(0))
        n_tokens = n_tokens + 1  # add masking token
        tokens_offset = n_tokens.cumsum(0) - n_tokens
        total_tokens = n_tokens.sum()

        self.register_buffer('tokens_offset', tokens_offset.unsqueeze(0))
        self.register_buffer('cat_mask', cat_mask.unsqueeze(0))

        # self.emb = nn.Embedding(total_tokens, hparams.emb_dim, sparse=True)
        # TODO: figure out should we add another dummy token for the case of categorical feature in the last position
        self.emb = nn.Embedding(total_tokens + 1, hparams.emb_dim, sparse=True)

        self.n_rules = hparams.n_rules

        if hparams.feature_bias:
            self.feature_bias = nn.Parameter(torch.randn(1, len(n_tokens), hparams.emb_dim))
        else:
            self.register_buffer('feature_bias', torch.zeros(1, len(n_tokens), hparams.emb_dim))

        if hparams.rules_bias:
            self.rule_bias = nn.Parameter(torch.randn(1, 1, hparams.emb_dim))
        else:
            self.register_buffer('rule_bias', torch.zeros(1, 1, hparams.emb_dim))

        self.rules = nn.Parameter(torch.randn(1, self.n_rules, hparams.emb_dim))
        self.mask = distributions.Bernoulli(1 - hparams.mask_rate)
        self.rule_mask = distributions.Bernoulli(1 - hparams.rule_mask_rate)

        self.transformer = nn.Transformer(d_model=hparams.emb_dim, nhead=hparams.n_transformer_head,
                                          num_encoder_layers=hparams.n_encoder_layers,
                                          num_decoder_layers=hparams.n_decoder_layers,
                                          dim_feedforward=hparams.transformer_hidden_dim,
                                          dropout=hparams.transformer_dropout,
                                          activation=hparams.activation, layer_norm_eps=1e-05,
                                          batch_first=True, norm_first=True)

        if hparams.lin_version > 0:
            self.lin = nn.Sequential(nn.ReLU(), nn.Dropout(hparams.dropout), nn.LayerNorm(hparams.emb_dim),
                nn.Linear(hparams.emb_dim, n_classes, bias=False))
        else:
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
        x = (1 - x_frac) * x1 + x_frac * x2 + self.feature_bias

        if self.training:
            rules = self.rule_mask.sample(torch.Size((len(x), self.n_rules, 1))).to(x.device) * self.rules
        else:
            rules = torch.repeat_interleave(self.rules, len(x), dim=0)

        rules = rules + self.rule_bias
        x = self.transformer(x, rules)
        x = self.lin(x.max(dim=1).values)

        x = x.squeeze(-1)
        return x


class DeepTabularAlg(Algorithm):

    def __init__(self, hparams, networks=None, net_kwargs=None,  **kwargs):
        # choose your network

        if networks is None:
            if net_kwargs is None:
                net_kwargs = dict()
            net = TabularTransformer(hparams, **net_kwargs)
            networks = {'net': net}

        super().__init__(hparams, networks=networks, **kwargs)
        self.loss_function = None
        self.loss_kwargs = None
        self.task_type = None
        self.train_acc = None
        self.previous_masking = 1 - self.get_hparam('mask_rate')
        self.best_masking = 1 - self.get_hparam('mask_rate')

    def preprocess_epoch(self, epoch=None, subset=None, training=True, **kwargs):
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

    def postprocess_epoch(self, sample=None, label=None, index=None, epoch=None, subset=None, training=True, **kwargs):
        if self.task_type == 'regression':

            rmse = np.sqrt(self.get_scalar('mse', aggregate=True))
            self.report_scalar('rmse', rmse)
            objective = -rmse
        else:
            objective = self.get_scalar('acc', aggregate=True)

        self.report_scalar('objective', objective)
        if self.get_hparam('dynamic_masking'):
            if training:
                self.train_acc = float(objective)
            else:
                test_acc = float(objective)
                if test_acc > self.train_acc:
                    delta = self.get_hparam('dynamic_delta')
                else:
                    delta = -self.get_hparam('dynamic_delta')
                self.previous_masking = float(self.net.mask.probs)
                non_mask_rate = max(self.previous_masking + delta, 1. - self.get_hparam('maximal_mask_rate'))
                non_mask_rate = min(non_mask_rate, 1. - self.get_hparam('minimal_mask_rate'))
                self.net.mask = distributions.Bernoulli(non_mask_rate)

            self.report_scalar('mask_rate', 1 - self.net.mask.probs)

    def train_iteration(self, sample=None, label=None, subset=None, counter=None, index=None,
                        training=True, **kwargs):

        y = label
        net = self.net

        y_hat = net(sample)

        loss = self.loss_function(y_hat, y, **self.loss_kwargs)

        self.apply(loss, training=training)

        # add scalar measurements
        if self.task_type == 'regression':
            self.report_scalar('mse', loss.mean() * self.dataset.y_sigma ** 2)
        else:
            self.report_scalar('acc', (y_hat.argmax(1) == y).float().mean())

    def set_best_masking(self):
        logger.info(f'Setting best masking to {self.best_masking:.3f}')
        self.net.mask = distributions.Bernoulli(self.best_masking)

    def predict_iteration(self, sample=None, label=None, subset=None, predicting=True, **kwargs):

        y = label
        net = self.net
        n_ensembles = self.get_hparam('n_ensembles')

        if n_ensembles > 1:
            net.train()
            y_hat = []
            for _ in range(n_ensembles):
                y_hat.append(net(sample))
            y_hat = torch.stack(y_hat, dim=0)
            self.report_scalar('y_pred_std', y_hat.std(dim=0))
            y_hat = y_hat.mean(dim=0)
        else:
            y_hat = net(sample)

        # add scalar measurements
        self.report_scalar('y_pred', y_hat)

        if not predicting:

            if self.task_type == 'regression':
                self.report_scalar('mse', F.mse_loss(y_hat, y, reduction='mean') * self.dataset.y_sigma ** 2)
            else:
                self.report_scalar('acc', (y_hat.argmax(1) == y).float().mean())

            self.report_scalar('target', y)

            return {'y': y, 'y_hat': y_hat}

        return y_hat

    def postprocess_inference(self, sample=None, subset=None, predicting=True, **kwargs):

        if not predicting:

            if self.task_type == 'regression':

                rmse = np.sqrt(self.get_scalar('mse', aggregate=True))
                self.report_scalar('rmse', rmse)
                self.report_scalar('objective', -rmse)

            else:

                y_pred = as_numpy(torch.argmax(self.get_scalar('y_pred'), dim=1))
                y_true = as_numpy(self.get_scalar('target'))
                precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)

                self.report_data('metrics/precision', precision)
                self.report_data('metrics/recall', recall)
                self.report_data('metrics/fscore', fscore)
                self.report_data('metrics/support', support)

                self.report_scalar('objective', self.get_scalar('acc', aggregate=True))
