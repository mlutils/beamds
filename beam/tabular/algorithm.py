import torch

from ..algorithm import NeuralAlgorithm
from ..utils import as_numpy, as_tensor
from ..data import BeamData
from ..config import BeamConfig, BeamParam
import torch.nn.functional as F
from torch import nn
from torch import distributions
from ..dataset import UniversalDataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from ..logging import beam_logger as logger
from ..nn import BeamNN
import torch._dynamo
torch._dynamo.config.suppress_errors = True


class TabularTransformer(nn.Module):
    """
    The TabularTransformer class is a PyTorch module that implements a transformer-based model for tabular data classification. It takes as input a set of hyperparameters, the number of classes, the number of tokens, and a categorical mask. The class inherits from the torch.nn.Module class.

    Attributes:
        - n_tokens (torch.Tensor): A tensor that represents the number of tokens for each categorical feature.
        - tokens_offset (torch.Tensor): A tensor that represents the offset of each categorical feature in the token embedding.
        - cat_mask (torch.Tensor): A tensor that masks categorical features.
        - emb (torch.nn.Embedding): An embedding layer for the tokens.
        - n_rules (int): The number of rules used in the model.
        - feature_bias (torch.Parameter): A learnable parameter used as bias for the token embeddings.
        - rule_bias (torch.Parameter): A learnable parameter used as bias for the rule embeddings.
        - rules (torch.Parameter): A learnable parameter that represents the rules.
        - mask (torch.distributions.Bernoulli): A Bernoulli distribution used for masking tokens.
        - rule_mask (torch.distributions.Bernoulli): A Bernoulli distribution used for masking rules.
        - transformer (torch.nn.Transformer): A transformer layer used for feature transformation.
        - lin (torch.nn.Module): A linear layer used for classification.

    Methods:
        - __init__(self, hparams, n_classes, n_tokens, cat_mask): Initializes the TabularTransformer class.
        - forward(self, sample): Performs a forward pass through the model.

    Example usage:
        hparams = {
            'emb_dim': 256,
            'n_rules': 4,
            'feature_bias': True,
            'rules_bias': True,
            'mask_rate': 0.2,
            'rule_mask_rate': 0.1,
            'n_transformer_head': 4,
            'n_encoder_layers': 4,
            'n_decoder_layers': 4,
            'transformer_hidden_dim': 512,
            'transformer_dropout': 0.1,
            'activation': 'relu',
            'lin_version': 1,
            'dropout': 0.2
        }
        n_classes = 2
        n_tokens = [10, 5, 8]
        cat_mask = [True, False, True]

        model = TabularTransformer(hparams, n_classes, n_tokens, cat_mask)
        sample = {
            'x': torch.tensor([[1, 3, 5], [2, 4, 6]]),
            'x_frac': torch.tensor([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]])
        }
        output = model.forward(sample)
    """
    def __init__(self, hparams, n_classes, n_tokens, cat_mask):
        """

        @param hparams: hyperparameters
        @param n_classes:
        @param n_tokens:
        @param cat_mask:
        """
        super().__init__()

        # n_tokens = as_tensor(n_tokens)
        # cat_mask = as_tensor(cat_mask)

        self.register_buffer('n_tokens', n_tokens.unsqueeze(0))
        n_tokens = n_tokens + 1  # add masking token
        tokens_offset = n_tokens.cumsum(0) - n_tokens
        total_tokens = int(n_tokens.sum())

        self.register_buffer('tokens_offset', tokens_offset.unsqueeze(0))
        self.register_buffer('cat_mask', cat_mask.unsqueeze(0))

        # self.emb = nn.Embedding(total_tokens, hparams.emb_dim, sparse=True)
        # TODO: figure out should we add another dummy token for the case of categorical feature in the last position
        self.emb = nn.Embedding(total_tokens + 1, hparams.emb_dim, sparse=hparams.sparse_embedding)

        self.n_rules = hparams.n_rules

        if hparams.feature_bias:
            self.feature_bias = nn.Parameter(torch.randn(1, len(n_tokens), hparams.emb_dim))
        else:
            self.register_buffer('feature_bias', torch.zeros(1, len(n_tokens), hparams.emb_dim))

        if hparams.rules_bias:
            self.rule_bias = nn.Parameter(torch.randn(1, 1, hparams.emb_dim))
        else:
            self.register_buffer('rule_bias', torch.zeros(1, 1, hparams.emb_dim))

        self.mask = distributions.Bernoulli(1 - hparams.mask_rate)

        self.rules = None
        self.rule_mask = None
        if hparams.n_decoder_layers > 0:
            self.rules = nn.Parameter(torch.randn(1, self.n_rules, hparams.emb_dim))
            self.rule_mask = distributions.Bernoulli(1 - hparams.rule_mask_rate)

            self.transformer = nn.Transformer(d_model=hparams.emb_dim, nhead=hparams.n_transformer_head,
                                              num_encoder_layers=hparams.n_encoder_layers,
                                              num_decoder_layers=hparams.n_decoder_layers,
                                              dim_feedforward=hparams.transformer_hidden_dim,
                                              dropout=hparams.transformer_dropout,
                                              activation=hparams.activation, layer_norm_eps=1e-05,
                                              batch_first=True, norm_first=True)
        else:

            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hparams.emb_dim, nhead=hparams.n_transformer_head,
                                           dim_feedforward=hparams.transformer_hidden_dim,
                                           dropout=hparams.transformer_dropout, activation=hparams.activation,
                                           layer_norm_eps=1e-05, norm_first=True, batch_first=True),
                num_layers=hparams.n_encoder_layers)

        if hparams.lin_version > 0:
            self.lin = nn.Sequential(nn.ReLU(), nn.Dropout(hparams.dropout), nn.LayerNorm(hparams.emb_dim),
                nn.Linear(hparams.emb_dim, n_classes, bias=False))
        else:
            self.lin = nn.Linear(hparams.emb_dim, n_classes, bias=False)

    def forward(self, x, x_frac):

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

        if self.rules is not None:
            if self.training:
                rules = self.rule_mask.sample(torch.Size((len(x), self.n_rules, 1))).to(x.device) * self.rules
            else:
                rules = torch.repeat_interleave(self.rules, len(x), dim=0)

            rules = rules + self.rule_bias
            x = self.transformer(x, rules)
        else:
            x = self.transformer(x)

        x = self.lin(x.max(dim=1).values)

        x = x.squeeze(-1)
        return x


class DeepTabularAlg(NeuralAlgorithm):

    def __init__(self, hparams, networks=None, net_kwargs=None, task_type=None, y_sigma=None,  **kwargs):
        # choose your network

        if networks is None:
            if net_kwargs is None:
                net_kwargs = dict()
            net_kwargs = as_tensor(net_kwargs)
            net = TabularTransformer(hparams, **net_kwargs)
            networks = {'net': net}

        super().__init__(hparams, networks=networks, **kwargs)
        self.loss_function = None
        self.loss_kwargs = None
        self.train_acc = None
        self.task_type = task_type
        self.y_sigma = y_sigma
        self.previous_masking = 1 - self.get_hparam('mask_rate')
        self.best_masking = 1 - self.get_hparam('mask_rate')

    def preprocess_epoch(self, epoch=None, subset=None, training=True, **kwargs):
        if epoch == 0:

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

    # def inner_train(self, sample=None, label=None, index=None, counter=None, subset=None, training=True, **kwargs):
    #     y = label
    #     net = self.net
    #
    #     y_hat = net(sample)
    #     loss = self.loss_function(y_hat, y, **self.loss_kwargs)
    #     self.apply(loss, training=training)
    #     return loss, y_hat, y

    def train_iteration(self, sample=None, label=None, subset=None, counter=None, index=None,
                        training=True, **kwargs):

        # loss, y_hat, y = self.optimized_inner_train(sample=sample, label=label, index=index, counter=counter, subset=subset,
        #                                training=training, **kwargs)

        y = label
        net = self.net

        x, x_frac = sample['x'], sample['x_frac']
        y_hat = net(x, x_frac)
        loss = self.loss_function(y_hat, y, **self.loss_kwargs)
        self.apply(loss, training=training)

        # add scalar measurements
        if self.task_type == 'regression':
            self.report_scalar('mse', loss.mean() * self.y_sigma ** 2)
        else:
            self.report_scalar('acc', (y_hat.argmax(1) == y).float().mean())

    def set_best_masking(self):
        logger.info(f'Setting best masking to {self.best_masking:.3f}')
        self.net.mask = distributions.Bernoulli(self.best_masking)

    def inference_iteration(self, sample=None, label=None, subset=None, predicting=True, **kwargs):

        y = label
        net = self.net
        n_ensembles = self.get_hparam('n_ensembles')
        x, x_frac = sample['x'], sample['x_frac']

        if n_ensembles > 1:
            net.train()
            y_hat = []
            for _ in range(n_ensembles):
                y_hat.append(net(x, x_frac))
            y_hat = torch.stack(y_hat, dim=0)
            self.report_scalar('y_pred_std', y_hat.std(dim=0))
            y_hat = y_hat.mean(dim=0)
        else:
            y_hat = net(x, x_frac)

        # add scalar measurements
        self.report_scalar('y_pred', y_hat)

        if not predicting:

            if self.task_type == 'regression':
                self.report_scalar('mse', F.mse_loss(y_hat, y, reduction='mean') * self.y_sigma ** 2)
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

    # def save_checkpoint(self, path=None, networks=True, optimizers=True, schedulers=True,
    #                     processors=True, scaler=True, scalers=True, swa_schedulers=True, swa_networks=True,
    #                     hparams=True, aux=None, pickle_model=False):
    #     aux = {'kwargs': {'net_kwargs': {'n_classes': self.dataset.n_classes,
    #                           'n_tokens': self.dataset.n_tokens,
    #                           'cat_mask': self.dataset.cat_mask}}}
    #
    #     return super().save_checkpoint(path=path, networks=networks, optimizers=optimizers, schedulers=schedulers,
    #                             processors=processors, scaler=scaler, scalers=scalers,
    #                             swa_schedulers=swa_schedulers, swa_networks=swa_networks, hparams=hparams,
    #                             aux=aux, pickle_model=pickle_model)
