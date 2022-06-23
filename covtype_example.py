#!/usr/bin/env python
# coding: utf-8

# In[1]:
import collections

import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import os

from src.beam import beam_arguments, Experiment
from src.beam import UniversalDataset, UniversalBatchSampler
from src.beam import Algorithm, PackedFolds
from src.beam import LinearNet
from src.beam import DataTensor, BeamOptimizer
from src.beam.utils import is_notebook
from src.beam.config import get_beam_parser
from src.beam.model import GBN, MHRuleLayer, BetterEmbedding, mySequential

from torchvision import transforms
import torchvision
from ray import tune
from functools import partial
from collections import namedtuple
from sklearn.datasets import fetch_covtype
import pandas as pd
from torch import Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import math

ModuleType = Union[str, Callable[..., nn.Module]]


def _is_glu_activation(activation: ModuleType):
    return (
            isinstance(activation, str)
            and activation.endswith('GLU')
            or activation in [ReGLU, GEGLU]
    )


def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].

    References:

        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)

def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].

    References:

        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    if isinstance(module_type, str):
        if module_type == 'ReGLU':
            return ReGLU()
        elif module_type == 'GEGLU':
            return GEGLU()
        else:
            try:
                cls = getattr(nn, module_type)
            except AttributeError as err:
                raise ValueError(
                    f'Failed to construct the module {module_type} with the arguments {args}'
                ) from err
            return cls(*args)
    else:
        return module_type(*args)


class FFN(nn.Module):
    """The Feed-Forward Network module used in every `Transformer` block."""

    def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            activation: ModuleType,
    ):
        super().__init__()
        self.linear_first = nn.Linear(
            d_token,
            d_hidden * (2 if _is_glu_activation(activation) else 1),
            bias_first,
            )
        self.activation = _make_nn_module(activation)
        self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_first(x)
        x = self.activation(x)
        x = self.linear_second(x)
        return x


class Head(nn.Module):
    """The final module of the `Transformer` that performs BERT-like inference."""

    def __init__(
            self,
            *,
            d_in: int,
            bias: bool,
            activation: ModuleType,
            normalization: ModuleType,
            d_out: int,
    ):
        super().__init__()
        self.normalization = _make_nn_module(normalization, d_in)
        self.activation = _make_nn_module(activation)
        self.linear = nn.Linear(d_in, d_out, bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x[:, -1]
        x = self.normalization(x)
        x = self.activation(x)
        x = self.linear(x)
        return x


class ResRuleLayer(nn.Module):

    def __init__(self, n_rules, e_dim, bias=True, activation='gelu', dropout=0.0, n_out=1, n_features=None,
                 ffn_activation='ReGLU', head_activation='ReLU',head_normalization='LayerNorm'):
        super(ResRuleLayer, self).__init__()

        self.bn1 = GBN(e_dim, virtual_batch_size=64, momentum=0.1)
        self.rl1 = MHRuleLayer(n_rules, n_rules, e_dim, bias=bias, dropout=dropout)
        self.sl1 = MHRuleLayer(n_rules, n_rules, e_dim, bias=bias, dropout=dropout)
        self.bn2 = GBN(e_dim, virtual_batch_size=64, momentum=0.1)

        self.FFN = FFN(d_token=e_dim,
                       d_hidden=int(4/3 * e_dim),
                       bias_first=True,
                       bias_second=True,
                       activation=ffn_activation)
        self.activation = getattr(F, activation)
        self.Head = Head( d_in=e_dim,
                          d_out=n_out,
                          bias=True,
                          activation=head_activation,  # type: ignore
                          normalization=head_normalization)

    def forward(self, x, e, y):

        r = x
        r = self.bn1(r.transpose(1, 2)).transpose(1, 2)
        r = self.activation(r)

        r1, ai = self.rl1(r)
        s1, ai = self.sl1(r)

        r = torch.sigmoid(s1) * r1
        x = r + x
        r = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        r = self.activation(r)
        r = self.FFN(r)

        r = r + x

        y.append(self.Head(r))

        return r, e, y

class RuleNet(nn.Module):

    def __init__(self, numerical_indices, categorical_indices, n_categories,
                 embedding_dim=256, n_rules=128, n_layers=5, dropout=0., n_out=1, bias=True,
                 activation='gelu', n_quantiles=20, n_tables=15,
                 initial_mask=1., qnorm_momentum=.001, k_p=0.05, k_i=0.005,
                 k_d=0.005, T=20, clip=0.005, quantile_resolution=1e-4):
        super().__init__()

        self.emb = BetterEmbedding(numerical_indices, categorical_indices, n_quantiles, n_categories, embedding_dim,
                 momentum=qnorm_momentum, track_running_stats=True, n_tables=n_tables, initial_mask=initial_mask,
                 k_p=k_p, k_i=k_i, k_d=k_d, T=T, clip=clip, quantile_resolution=quantile_resolution,
                 use_stats_for_train=True, boost=True, flatten=False, quantile_embedding=True, tokenizer=True,
                 qnorm_flag=True, kaiming_init=False, init_spline_equally=True, sparse=True)

        n_features = len(numerical_indices) + len(categorical_indices)
        self.first_rule = MHRuleLayer(n_rules, n_features, embedding_dim,
                                      bias=bias, dropout=dropout)
        self.rules = mySequential(*[ResRuleLayer(n_rules, embedding_dim,
                                                 bias=bias, activation=activation,
                                                 dropout=dropout, n_out=n_out)
                                    for _ in range(n_layers)],
                                  )

    def forward(self, x):

        e = self.emb(x )

        x, _ = self.first_rule(e)

        x, _, y = self.rules(x, e, [])

        y = torch.stack(y, dim=1).sum(dim=1)

        return y



class CovtypeDataset(UniversalDataset):

    def __init__(self, path,
                 train_batch_size, eval_batch_size, device='cpu', weight_factor=.5, seed=5782):

        dataset = fetch_covtype(data_home=path)
        data = dataset['data']
        columns = dataset['feature_names']
        y = np.array(dataset['target'], dtype=np.int64)
        df = pd.DataFrame(data=data, columns=columns, index=np.arange(len(data)))

        soils_columns = [c for c in df.columns if 'Soil' in c]
        soil = np.where(df[soils_columns])[1]

        wilderness_columns = [c for c in df.columns if 'Wilderness' in c]
        wilderness = np.where(df[wilderness_columns])[1]

        df_cat = pd.DataFrame({'Soil': soil, 'Wilderness': wilderness})
        df_num = df.drop(columns=(soils_columns+wilderness_columns))

        covtype = pd.concat([df_num, df_cat], axis=1)
        super().__init__(x=covtype.values, y=y, device=device)

        self.split(validation=.2, test=.2, seed=seed, stratify=True, labels=self.data['y'].cpu())
        self.build_samplers(train_batch_size, eval_batch_size, oversample=True, weight_factor=weight_factor)

        self.categorical_indices = torch.arange(10, 12)
        self.numerical_indices = torch.arange(10)
        self.n_categories = torch.tensor(df_cat.nunique().values)
        self.n_classes = int(y.max() + 1)


class CovtypeAlgorithm(Algorithm):

    def __init__(self, *args, **argv):
        super().__init__(*args, **argv)

        networks, dataloaders, experiment = args

        self.eval_ensembles = experiment.eval_ensembles

        self.label_smoothing = experiment.label_smoothing
        self.scheduler = self.optimizers['net'].set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau,
                                                              mode='min', factor=experiment.scheduler_factor,
                                                              patience=experiment.scheduler_patience,
                                                              threshold=0, threshold_mode='rel', cooldown=0, min_lr=0,
                                                              eps=1e-06, verbose=True)

    def postprocess_epoch(self, sample=None, aux=None, results=None, epoch=None, subset=None, training=True):

        x, y = sample['x'], sample['y']


        if training:
            results['scalar'][f'lr'] = self.optimizers['net'].dense.param_groups[0]['lr']
            self.last_train_loss = float(np.mean(results['scalar']['loss']))
        else:
            val_loss = float(np.mean(results['scalar']['loss']))
            self.networks['net'].emb.step(self.last_train_loss, val_loss)
            self.scheduler.step(val_loss)

        aux = {}
        return aux, results

    def iteration(self, sample=None, aux=None, results=None, subset=None, training=True):

        x, y = sample['x'], sample['y']

        net = self.networks['net']
        opt = self.optimizers['net']

        with torch.cuda.amp.autocast(enabled=self.amp):
            y_hat = net(x)
            loss = F.cross_entropy(y_hat, y, reduction='sum', label_smoothing=self.label_smoothing)
                   # + net.emb.get_llr()

        opt.apply(loss, training=training)

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['acc'].append(float((y_hat.argmax(1) == y).float().mean()))

        return aux, results

    def report(self, results, i):

        acc = np.mean(results['validation']['scalar']['acc'])

        if self.hpo == 'tune':
            tune.report(mean_accuracy=acc)
        elif self.hpo == 'optuna':

            self.trial.report(acc, i)
            results['objective'] = acc

        else:
            raise NotImplementedError

        return results

    def inference(self, sample=None, aux=None, results=None, subset=None, with_labels=True):

        if with_labels:
            x, y = sample['x'], sample['y']

        else:
            x = sample

        net = self.networks['net']

        if subset == 'validation':
            y_hat = net(x)

        else:
            net.train()
            y_hat = []
            for i in range(self.eval_ensembles):
                y_hat.append(net(x))

            y_hat = torch.stack(y_hat).mean(dim=0)

        # add scalar measurements
        results['predictions']['y_pred'].append(y_hat.detach())

        if with_labels:
            results['scalar']['acc'].append(float((y_hat.argmax(1) == y).float().mean()))
            aux['predictions']['target'].append(y)

        return aux, results

    def postprocess_inference(self, sample=None, aux=None, results=None, subset=None, with_labels=True):
        y_pred = torch.cat(results['predictions']['y_pred'])

        y_pred = torch.argmax(y_pred, dim=1).data.cpu().numpy()

        if with_labels:
            y_true = torch.cat(aux['predictions']['target']).data.cpu().numpy()
            precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)
            results['metrics']['precision'] = precision
            results['metrics']['recall'] = recall
            results['metrics']['fscore'] = fscore
            results['metrics']['support'] = support

        return aux, results


def covtype_algorithm_generator(experiment, **kwargs):

    dataset = CovtypeDataset(experiment.path_to_data, experiment.batch_size_train, experiment.batch_size_eval,
                             device=experiment.device, weight_factor=experiment.weight_factor, seed=experiment.seed)

    pin_memory = 'cpu' not in str(experiment.device)
    dataloader = dataset.build_dataloaders(num_workers=experiment.cpu_workers, pin_memory=pin_memory)

    # choose your network
    net = RuleNet(dataset.numerical_indices, dataset.categorical_indices,
                  dataset.n_categories, n_out=dataset.n_classes, embedding_dim=experiment.channels,
                  n_rules=experiment.n_rules, n_layers=experiment.n_layers, dropout=experiment.dropout,
                 activation=experiment.activation, n_quantiles=experiment.n_quantiles, n_tables=experiment.n_tables,
                  initial_mask=experiment.initial_mask, qnorm_momentum=experiment.qnorm_momentum, k_p=experiment.k_p,
                  k_i=experiment.k_i, k_d=experiment.k_d, T=experiment.T_pid, clip=experiment.clip_pid)

    optimizer = None

    # we recommend using the algorithm argument to determine the type of algorithm to be used
    Alg = globals()[experiment.algorithm]
    alg = Alg(net, dataloader, experiment, optimizers=optimizer)

    return alg


# Add experiment hyperparameter arguments

def get_covtype_parser():

    parser = get_beam_parser()

    parser.add_argument('--weight-factor', type=float, default=0.5, help='Squashing factor for the oversampling probabilities')
    parser.add_argument('--label-smoothing', type=float, default=0.05, help='Smoothing factor in the Cross Entropy loss')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function in RuleNet')
    parser.add_argument('--channels', type=int, default=256, help='Size of embedding')
    parser.add_argument('--Dropout', type=float, default=0.0, help='Dropout value for rule layers')
    parser.add_argument('--n-rules', type=int, default=128, help='Number of rules')
    parser.add_argument('--n-layers', type=int, default=5, help='Number of Residual Rule layers')
    parser.add_argument('--n-quantiles', type=int, default=20, help='Number of quantiles for the BetterEmbedding')
    parser.add_argument('--n-tables', type=int, default=15, help='Number of tables for the BetterEmbedding')

    parser.add_argument('--scheduler-factor', type=float, default=1 / math.sqrt(10), help='Reduce factor on Plateau')
    parser.add_argument('--scheduler-patience', type=int, default=16, help='Patience for plateau identification')
    parser.add_argument('--eval-ensembles', type=int, default=64, help='Number of repetitions of eval passes for each example (to build the ensemble)')

    parser.add_argument('--initial-mask', type=float, default=1., help='Initial PID masking')
    parser.add_argument('--qnorm-momentum', type=float, default=.001, help='Momentum for the quantile normalization')
    parser.add_argument('--k_p', type=float, default=.05, help='Kp PID coefficient')
    parser.add_argument('--k_i', type=float, default=.005, help='Ki PID coefficient')
    parser.add_argument('--k_d', type=float, default=.005, help='Kd PID coefficient')
    parser.add_argument('--T-pid', type=int, default=20, help='PID integration memory')
    parser.add_argument('--clip-pid', type=int, default=.005, help='PID clipping value')


    return parser


if __name__ == '__main__':

    # here you put all actions which are performed only once before initializing the workers
    # for example, setting running arguments and experiment:

    path_to_data = '/home/shared/data/dataset/covtype'
    root_dir = '/home/shared/data/results/covtype'

    args = beam_arguments(get_covtype_parser(),
        f"--project-name=covtype --root-dir={root_dir} --algorithm=CovtypeAlgorithm --device=1 --amp --lr-d=1e-3 --batch-size=256",
        "--n-epochs=100 --clip=0 --parallel=1 --accumulate=1 --cudnn-benchmark --identifier=half_precision",
        "--weight-decay=1e-5 --beta1=0.9 --beta2=0.999", label_smoothing=.05, weight_factor=.5,
        path_to_data=path_to_data, dropout=.0, activation='gelu', channels=512)


    experiment = Experiment(args)
    alg = experiment(covtype_algorithm_generator, tensorboard_arguments=None)

    # ## Inference
    inference = alg('test')

    print('Test inference results:')
    for n, v in inference['metrics'].items():
        print(f'{n}:')
        print(v)

