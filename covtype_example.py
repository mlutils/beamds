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
from src.beam.config import parser

from torchvision import transforms
import torchvision
from ray import tune
from functools import partial
from collections import namedtuple
from sklearn.datasets import fetch_covtype
import pandas as pd


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

        self.bn1 = GBN(e_dim, virtual_batch_size=256, momentum=0.1)
        self.rl1 = MHRuleLayer(n_rules, e_dim, e_dim, bias=bias, dropout=dropout)
        self.sl1 = MHRuleLayer(n_rules, e_dim, e_dim, bias=bias, dropout=dropout)
        self.bn2 = GBN(e_dim, virtual_batch_size=256, momentum=0.1)

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

    def __init__(self, n_features, features_offset, embedding_dim=256, n_rules=128,
                 n_layers=5, dropout=0.2, n_out=1, bias=True, activation='gelu', noise=0.1,
                 quantiles=50, predefined_boundaries=None, n_tables=15):
        super(RuleNet, self).__init__()

        self.q_norm = LazyQuantileNorm(quantiles=quantiles, predefined=predefined_boundaries, noise=noise)
        self.register_buffer('features_offset', features_offset)

        self.emb = nn.Embedding(n_features * n_tables, embedding_dim, sparse=True)

        self.first_rule = MHRuleLayer(n_rules, embedding_dim, embedding_dim,
                                      bias=bias, dropout=dropout)
        self.rules = mySequential(*[ResRuleLayer(n_rules, embedding_dim,
                                                 bias=bias, activation=activation,
                                                 dropout=dropout, n_out=n_out)
                                    for _ in range(n_layers)],
                                  )
        self.n_tables = n_tables
        self.n_features = n_features

    def forward(self, x_num, x_cat):

        x_num = self.q_norm(x_num)

        x = torch.cat([x_num, x_cat], dim=1) + self.features_offset

        table = np.random.randint(self.n_tables)
        e = self.emb(x + self.n_features * table)
        x, _ = self.first_rule(e)

        x, _, y = self.rules(x, e, [])

        y = torch.stack(y, dim=1).sum(dim=1)

        return y, None, None



class CovtypeDataset(UniversalDataset):

    def __init__(self, path,
                 train_batch_size, eval_batch_size, device='cuda', weight_factor=.5, seed=5782):

        dataset = fetch_covtype(data_home=path)
        data = dataset['data']
        columns = dataset['feature_names']
        y = dataset['target']
        df = pd.DataFrame(data=data, columns=columns, index=np.arange(len(data)))

        soils_columns = [c for c in df.columns if 'Soil' in c]
        soil = np.where(df[soils_columns])[1]

        wilderness_columns = [c for c in df.columns if 'Wilderness' in c]
        wilderness = np.where(df[wilderness_columns])[1]

        df_cat = pd.DataFrame({'Soil': soil, 'Wilderness': wilderness})
        df_num = df.drop(columns=(soils_columns+wilderness_columns))

        covtype = pd.concat([df_num, df_cat], axis=1)
        super().__init__(x=covtype.values, y=y)

        self.split(validation=.2, test=.2, seed=seed, stratify=True, labels=self.data['y'])
        self.build_samplers(train_batch_size, eval_batch_size, oversample=True, weight_factor=weight_factor)


class CovtypeAlgorithm(Algorithm):

    def __init__(self, *args, **argv):
        super().__init__(*args, **argv)

        networks, dataloaders, experiment = args

        self.scheduler = self.optimizers['net'].set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlatue, last_epoch=- 1,
                                                              lr_lambda=LRPolicy(gain=experiment.gain,
                                                                                 turn_point=experiment.turn_point,
                                                                                 final_point=experiment.final_point,
                                                                                 minimal_gain=experiment.minimal_gain))

    def postprocess_epoch(self, sample=None, aux=None, results=None, epoch=None, subset=None, training=True):

        x, y = sample['x'], sample['y']

        results['images']['sample'] = x[:16].view(16, 3, 32, 32).data.cpu()

        if training:
            results['scalar'][f'lr'] = self.optimizers['net'].dense.param_groups[0]['lr']

        aux = {}
        return aux, results

    def iteration(self, sample=None, aux=None, results=None, subset=None, training=True):

        x, y = sample['x'], sample['y']

        net = self.networks['net']
        opt = self.optimizers['net']

        with torch.cuda.amp.autocast(enabled=self.amp):
            y_hat = net(x)
            loss = F.cross_entropy(y_hat, y, reduction='sum', label_smoothing=0.2)

        opt.apply(loss, training=training)

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['acc'].append(float((y_hat.argmax(1) == y).float().mean()))

        if training:
            self.scheduler.step()

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

        y_hat = net(x)

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
    net = RuleNet(experiment.channels, dropout=experiment.dropout, activation=experiment.activation)

    optimizer = None
    optimizer = partial(BeamOptimizer, dense_args={'lr': experiment.lr_dense, 'weight_decay': experiment.weight_decay,
                                                   'momentum': experiment.beta1, 'nesterov': True}, clip=experiment.clip, accumulate=experiment.accumulate,
                        amp=experiment.amp,
                        sparse_args=None, dense_optimizer='SGD', sparse_optimizer='SparseAdam')

    # we recommend using the algorithm argument to determine the type of algorithm to be used
    Alg = globals()[experiment.algorithm]
    alg = Alg(net, dataloader, experiment, optimizers=optimizer)

    return alg


# ## Training

if __name__ == '__main__':

    # here you put all actions which are performed only once before initializing the workers
    # for example, setting running arguments and experiment:

    parser.add_argument('--beta2', type=float, default=0.999, metavar='β', help='Adam\'s β2 parameter')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='β', help='Adam\'s β2 parameter')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='β', help='Adam\'s β2 parameter')

    path_to_data = '/home/shared/data/dataset/covtype'
    root_dir = '/home/shared/data/results/covtype'

    args = beam_arguments(
        f"--project-name=covtype --root-dir={root_dir} --algorithm=CovtypeAlgorithm --device=1 --half --lr-d=1e-4 --batch-size=512",
        "--n-epochs=2 --epoch-length-train=50000 --epoch-length-eval=10000 --clip=0 --parallel=1 --accumulate=1 --cudnn-benchmark",
        "--weight-decay=.00256 --beta1=0.9 --beta2=0.9",
        path_to_data=path_to_data, gamma=1., dropout=.0, activation='celu', channels=512,
        scale_down=.7, scale_up=1.4, ratio_down=.7, ratio_up=1.4)


    experiment = Experiment(args)
    alg = experiment(covtype_algorithm_generator, tensorboard_arguments=None)

    # ## Inference
    inference = alg('test')

    print('Test inference results:')
    for n, v in inference['metrics'].items():
        print(f'{n}:')
        print(v)

