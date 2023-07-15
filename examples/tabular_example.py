from examples.example_utils import add_beam_to_path
add_beam_to_path()

import torch
from torch import distributions
import torchvision
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import os

from src.beam import beam_arguments, Experiment
from src.beam import UniversalDataset, UniversalBatchSampler
from src.beam import Algorithm, PackedFolds, as_numpy
from src.beam import DataTensor, BeamOptimizer
from src.beam.data import BeamData


class TabularTransformer(torch.nn.Module):

    def __init__(self, hparams, n_tokens, cat_mask, n_classes):

        super().__init__()

        self.register_buffer('n_tokens', n_tokens.unsqueeze(0))
        n_tokens = n_tokens + 1  # add dropout token
        tokens_offset = n_tokens.cumsum(0) - n_tokens
        total_tokens = n_tokens.sum()

        self.register_buffer('tokens_offset', tokens_offset.unsqueeze(0))
        self.register_buffer('cat_mask', cat_mask.unsqueeze(0))
        self.emb = nn.Embedding(total_tokens, hparams.emb_dim, sparse=True)
        self.quantization_noize = hparams.quantization_noize

        n_rules = hparams.n_rules

        self.rules = nn.Parameter(torch.randn(1, n_rules, hparams.emb_dim))

        self.mask = distributions.Bernoulli(1 - hparams.features_mask_rate)

        self.transformer = nn.Transformer(d_model=hparams.emb_dim, nhead=hparams.n_transformer_head,
                                          num_encoder_layers=hparams.n_encoder_layers,
                                          num_decoder_layers=hparams.n_decoder_layers,
                                          dim_feedforward=hparams.transformer_hidden_dim,
                                          dropout=hparams.transformer_dropout,
                                          activation='gelu', layer_norm_eps=1e-05,
                                          batch_first=True, norm_first=True)

        self.lin = nn.Linear(hparams.emb_dim, n_classes, bias=False)

    def forward(self, x):

        if self.training:
            x = x + self.quantization_noize * torch.randn(x.shape, device=x.device) * self.cat_mask
            x = torch.clamp(x, min=0)
            x = torch.minimum(x, self.n_tokens-1).long()

        x = (x + 1)
        if self.training:
            x = x * self.mask.sample(x.shape).to(x.device).long()

        x = x + self.tokens_offset
        x = self.emb(x)
        x = self.transformer(x, torch.repeat_interleave(self.rules, len(x), dim=0))
        x = self.lin(x[:, -1, :])
        return x


class DeepTabularAlg(Algorithm):

    def __init__(self, hparams, **kwargs):

        # choose your network
        super().__init__(hparams, **kwargs)

        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizers['net'].dense, gamma=0.99)
        self.stop_at = hparams.stop_at

    def early_stopping(self, results=None, epoch=None, **kwargs):

        if 'validation' in results:
            acc = torch.mean(results['validation']['scalar']['acc'])
        else:
            acc = torch.mean(results['test']['scalar']['acc'])

        return acc > self.stop_at

    def postprocess_epoch(self, sample=None, label=None, results=None, epoch=None, subset=None, training=True,
                          **kwargs):

        y = label
        x = sample['x']

        if training:
            # self.scheduler.step()
            results['scalar'][f'lr'] = self.optimizers['net'].dense.param_groups[0]['lr']

        return results

    def iteration(self, sample=None, label=None, results=None, subset=None, counter=None, training=True, **kwargs):

        y = label
        x = sample['x']

        net = self.networks['net']
        opt = self.optimizers['net']

        y_hat = net(x)
        loss = F.cross_entropy(y_hat, y, reduction='none')

        self.apply(loss, training=training, results=results)

        # add scalar measurements
        # results['scalar']['loss'].append(as_numpy(loss))
        results['scalar']['ones'].append(as_numpy(x.sum(dim=-1)))
        results['scalar']['acc'].append(as_numpy((y_hat.argmax(1) == y).float().mean()))

        return results

    def inference(self, sample=None, label=None, results=None, subset=None, predicting=True, **kwargs):

        if predicting:
            x = sample
        else:
            y = sample['y']
            x = sample['x']

        net = self.networks['net']
        y_hat = net(x)

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
