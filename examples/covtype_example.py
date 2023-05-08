import rtdl

try:
    from examples.example_utils import add_beam_to_path
except ModuleNotFoundError:
    from example_utils import add_beam_to_path
add_beam_to_path()

import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import os

from src.beam import beam_arguments, Experiment, as_tensor, as_numpy
from src.beam import UniversalDataset, UniversalBatchSampler
from src.beam import Algorithm, PackedFolds, LinearNet
from src.beam.model import PID
from src.beam.config import get_beam_parser
from src.beam.model import GBN, MHRuleLayer, mySequential

from ray import tune
from sklearn.datasets import fetch_covtype
import pandas as pd
from torch import Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import math
from sklearn.preprocessing import QuantileTransformer
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import math
from src.beam import beam_logger as logger

ModuleType = Union[str, Callable[..., nn.Module]]

# def preprocess_feature(v, nq=20, feature_type=None):
#     '''
#     get vector of features and calculate
#     quantiles/categories
#
#     returns vc, categories
#
#     vc - categorical representation of v
#     categories - names of categories (if quantile it is (a, b])
#
#     currently does not handle nan.
#     '''
#
#     if type(v) is not pd.Series:
#         v = pd.Series(v)
#
#     # for now we use a simple rule to distinguish between categorical and numerical features
#     n = v.nunique()
#
#     if n > nq or (feature_type is not None and feature_type == 'numerical'):
#
#         c_type = 'numerical'
#
#         q = (np.arange(nq + 1)) / nq
#
#         vc = pd.qcut(v, q, labels=False, duplicates='drop')
#         categories = v.quantile(q).values[:-1]
#         vc = vc.fillna(-1).values
#
#     else:
#
#         c_type = 'categorical'
#
#         vc, categories = pd.factorize(v)
#
#     # allocate nan value
#     categories = np.insert(categories, 0, np.nan)
#     vc = vc + 1
#
#     return vc, categories, c_type
#
#
# def preprocess_table(df, nq=20, offset=True, feature_type=None):
#     metadata = defaultdict(OrderedDict)
#     n = 0
#     dfc = {}
#
#     for c in df.columns:
#
#         if feature_type is not None:
#             ft = feature_type[c]
#
#         vc, categories, c_type = preprocess_feature(df[c], nq=nq, feature_type=ft)
#
#         m = len(categories)
#         metadata['n_features'][c] = m
#         metadata['categories'][c] = categories
#         metadata['aggregated_n_features'][c] = n
#         metadata['c_type'][c] = c_type
#
#         if offset:
#             vc = vc + n
#
#         n = n + m
#         dfc[c] = vc
#
#     dfc = pd.DataFrame(dfc).astype(np.int64)
#
#     metadata['total_features'] = n
#
#     return dfc, metadata


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


class GEGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


def _is_glu_activation(activation: ModuleType):
    return (
            isinstance(activation, str)
            and activation.endswith('GLU')
            or activation in [ReGLU, GEGLU]
    )


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


class BetterEmbedding(torch.nn.Module):

    def __init__(self, n_num, n_categories, emb_dim, n_tables=15, initial_mask=1.,
                 k_p=0.05, k_i=0.005, k_d=0.005, T=20, clip=0.005, flatten=False, kaiming_init=False, sparse=True):

        super().__init__()

        n_categories = as_tensor(n_categories)
        n_cat = len(n_categories)
        n_categories = n_categories + 1
        cat_offset = n_categories.cumsum(0) - n_categories
        self.register_buffer("cat_offset", cat_offset.unsqueeze(0))
        self.register_buffer("null_emb_cat", torch.FloatTensor(1, 0, emb_dim))

        self.flatten = flatten
        self.n_tables = n_tables

        self.n_emb = int(n_categories.sum())

        self.pid = PID(k_p=k_p, k_i=k_i, k_d=k_d, T=T, clip=clip)
        self.br = initial_mask

        if n_cat:
            self.emb_cat = nn.Embedding(1 + self.n_emb * n_tables, emb_dim, sparse=sparse)
        else:
            self.emb_cat = lambda x: self.null_emb_cat.repeat(len(x), 1, 1)

        if n_num:
            self.weight = nn.Parameter(torch.empty((n_tables, n_num, emb_dim)))
            self.mask_num = nn.Parameter(torch.empty((1, n_num, emb_dim)))
            if kaiming_init:
                nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.mask_num, a=math.sqrt(5))
            else:
                nn.init.normal_(self.weight)
                nn.init.normal_(self.mask_num)
        else:
            self.register_buffer('weight', torch.zeros((n_num, emb_dim)))

        self.bias = nn.Parameter(torch.empty((1, n_num+n_cat, emb_dim)))
        if kaiming_init:
            nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))
        else:
            nn.init.normal_(self.bias)

        self.emb_dim = emb_dim
        self.n_num = n_num
        self.n_cat = n_cat

    def step(self, train_loss, val_loss):
        self.br = min(max(0, self.br + self.pid((train_loss - val_loss) / val_loss)), 1)
        logger.info(f"br was changed to {self.br}")

    def forward(self, x_num, x_cat, mask=True):

        if mask:
            bernoulli = torch.distributions.bernoulli.Bernoulli(probs=self.br)
            mask_num = bernoulli.sample(sample_shape=x_num.shape).unsqueeze(-1).to(x_num.device)
            mask_cat = bernoulli.sample(sample_shape=x_cat.shape).long().to(x_cat.device)

        else:
            mask_num = 1
            mask_cat = 1

        if self.training:
            rand_table = torch.randint(self.n_tables, size=(1, 1), device=x_cat.device)
        else:
            rand_table = torch.randint(self.n_tables, size=(len(x_num), 1), device=x_cat.device)

        x_cat = (x_cat + 1) * mask_cat + self.cat_offset + self.n_emb * rand_table
        e_cat = self.emb_cat(x_cat)

        e_num = (1 - mask_num) * self.mask_num + mask_num * self.weight[rand_table.squeeze(-1)] * x_num.unsqueeze(-1)

        e = torch.cat([e_cat, e_num], dim=1)
        e = e + self.bias

        if self.flatten:
            e = e.view(len(e), -1)

        return e


class ResRuleLayer(nn.Module):

    def __init__(self, n_rules, e_dim, bias=True, activation='gelu', dropout=0.0, n_out=1, n_features=None,
                 ffn_activation='ReGLU', head_activation='ReLU', head_normalization='LayerNorm'):
        super(ResRuleLayer, self).__init__()

        self.bn1 = GBN(e_dim, virtual_batch_size=64, momentum=0.1)
        self.rl1 = MHRuleLayer(n_rules, n_rules, e_dim, bias=bias, dropout=dropout)
        self.sl1 = MHRuleLayer(n_rules, n_rules, e_dim, bias=bias, dropout=dropout)
        self.bn2 = GBN(e_dim, virtual_batch_size=64, momentum=0.1)

        self.ffn = FFN(d_token=e_dim,
                       d_hidden=int(4/3 * e_dim),
                       bias_first=True,
                       bias_second=True,
                       activation=ffn_activation)
        self.activation = getattr(F, activation)
        self.head = Head(d_in=e_dim,
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
        r = self.ffn(r)

        r = r + x

        y.append(self.head(r))

        return r, e, y


class RuleNet(nn.Module):

    def __init__(self, n_num, n_cat, n_categories,
                 embedding_dim=256, n_rules=128, n_layers=5, dropout=0., n_out=1, bias=True,
                 activation='gelu', n_tables=15, initial_mask=1.,  k_p=0.05, k_i=0.005, k_d=0.005, T=20, clip=0.005):
        super().__init__()

        self.emb = BetterEmbedding(n_num, n_categories, embedding_dim,
                                   n_tables=n_tables, initial_mask=initial_mask, k_p=k_p, k_i=k_i, k_d=k_d, T=T,
                                   clip=clip, flatten=False, kaiming_init=False, sparse=True)

        n_features = n_num + n_cat
        self.first_rule = MHRuleLayer(n_rules, n_features, embedding_dim,
                                      bias=bias, dropout=dropout)
        self.rules = mySequential(*[ResRuleLayer(n_rules, embedding_dim,
                                                 bias=bias, activation=activation,
                                                 dropout=dropout, n_out=n_out)
                                    for _ in range(n_layers)],
                                  )

    def forward(self, x_num, x_cat):

        e = self.emb(x_num, x_cat)
        x, _ = self.first_rule(e)
        x, _, y = self.rules(x, e, [])

        y = torch.stack(y, dim=1).sum(dim=1)

        return y


class CovtypeDataset(UniversalDataset):

    def __init__(self, hparams):

        path = hparams.path_to_data
        device = hparams.device
        seed = hparams.seed

        dataset = fetch_covtype(data_home=path)
        data = dataset['data']
        columns = dataset['feature_names']
        y = np.array(dataset['target'], dtype=np.int64)
        df = pd.DataFrame(data=data, columns=columns, index=np.arange(len(data)))

        soils_columns = [c for c in df.columns if 'Soil' in c]
        soil = np.where(df[soils_columns])[1]

        wilderness_columns = [c for c in df.columns if 'Wilderness' in c]
        wilderness = np.where(df[wilderness_columns])[1]

        # df_cat = pd.DataFrame({'Soil': soil, 'Wilderness': wilderness})
        df_cat = df[soils_columns + wilderness_columns]
        df_num = df.drop(columns=(soils_columns + wilderness_columns))

        # covtype = pd.concat([df_num, df_cat], axis=1)
        # super().__init__(x=covtype.values.astype(np.float32), y=y, device=device)
        super().__init__(x_num=df_num.values.astype(np.float32), x_cat=df_cat.values.astype(np.int64),
                         y=y, device=device)
        # super().__init__(x_num=df_num.values.astype(np.float32), x_cat=df_cat.values.astype(np.float32),
        #                  y=y, device=device)

        self.split(validation=92962, test=116203, seed=seed, stratify=False, labels=self.data['y'].cpu())

        # self.categorical_indices = torch.arange(10, 12)
        # self.numerical_indices = torch.arange(10)

        self.qt = QuantileTransformer(n_quantiles=1000,
                                      output_distribution='uniform',
                                      ignore_implicit_zeros=False,
                                      subsample=100000,
                                      random_state=None,
                                      copy=True).fit(df_num.iloc[self.indices['train']].values)

        self.data['x_num'] = self.qt_transform(df_num, device=self.device)

        # super().__init__(x_num=df_num.values.astype(np.float32), x_cat=df.cat.values.astype(np.int65), y=y,
        #                  device=device)
        self.n_categories = torch.tensor(df_cat.nunique().values)

        self.n_classes = int(y.max() + 1)
        self.n_num = df_num.shape[-1]
        self.n_cat = df_cat.shape[-1]

    def qt_transform(self, df_num, device=None):
        return as_tensor(self.qt.transform(as_numpy(df_num)).astype(np.float32), device=device)

class CovtypeAlgorithm(Algorithm):

    def __init__(self, hparams, dataset=None, optimizers=None):

        self.eval_ensembles = hparams.eval_ensembles
        self.label_smoothing = hparams.label_smoothing

        net = RuleNet(dataset.n_num, dataset.n_cat, dataset.n_categories,
                      embedding_dim=hparams.channels, n_rules=hparams.n_rules, n_layers=hparams.n_layers,
                      dropout=hparams.dropout, n_out=dataset.n_classes, bias=True,
                      activation=hparams.activation, n_tables=hparams.n_tables, initial_mask=hparams.initial_mask,
                      k_p=hparams.k_p, k_i=hparams.k_i, k_d=hparams.k_d, T=hparams.T_pid, clip=hparams.clip_pid)

        # net = LinearNet(dataset.n_num + dataset.n_cat, 256, dataset.n_classes, 4)

        # net = rtdl.FTTransformer.make_baseline(n_num_features=dataset.n_num,
        #                                        cat_cardinalities=list(as_numpy(dataset.n_categories)),
        #                                        d_token=hparams.channels, n_blocks=3, attention_dropout=hparams.dropout,
        #                                        ffn_dropout=hparams.dropout, d_out=dataset.n_classes,
        #                                        residual_dropout=hparams.dropout, ffn_d_hidden=hparams.channels)

        # optimizers = torch.optim.AdamW(net.optimization_param_groups(), lr=hparams.lr_dense,
        #                                weight_decay=hparams.weight_decay)

        super().__init__(hparams, networks=net, dataset=dataset, optimizers=optimizers)

    def postprocess_epoch(self, sample=None, results=None, epoch=None, subset=None, training=True, **kwargs):

        # x, y = sample['x'], sample['y']
        # x_num, x_cat, y = sample['x_num'], sample['x_cat'], sample['y']

        if training:
            self.last_train_loss = float(np.mean(results['scalar']['loss']))
        else:
            val_loss = float(np.mean(results['scalar']['loss']))
            # self.networks['net'].emb.step(self.last_train_loss, val_loss)

        return results

    def iteration(self, sample=None, results=None, subset=None, training=True, **kwargs):

        # x, y = sample['x'], sample['y']
        x_num, x_cat, y = sample['x_num'], sample['x_cat'], sample['y']

        net = self.networks['net']

        y_hat = net(x_num, x_cat)
        # y_hat = net(torch.cat([x_num, x_cat], dim=1))
        loss = F.cross_entropy(y_hat, y, reduction='none', label_smoothing=self.label_smoothing)

        self.apply(loss, results=results, training=training)

        # add scalar measurements
        results['scalar']['acc'].append(float((y_hat.argmax(1) == y).float().mean()))

        return results

    def inference(self, sample=None, results=None, subset=None, with_labels=True, **kwargs):

        if with_labels:
            x_num, x_cat, y = sample['x_num'], sample['x_cat'], sample['y']

        else:
            x_num, x_cat = sample['x_num'], sample['x_cat']

        net = self.networks['net']
        # x = torch.cat([x_num, x_cat])

        if subset == 'validation':
            # y_hat = net(x)
            y_hat = net(x_num, x_cat)
        else:
            net.train()
            y_hat = []
            for i in range(self.eval_ensembles):
                # y_hat.append(net(x))
                y_hat.append(net(x_num, x_cat))

            y_hat = torch.stack(y_hat).mean(dim=0)

        # add scalar measurements
        results['predictions']['y_pred'].append(y_hat.detach())

        if with_labels:
            results['scalar']['acc'].append(float((y_hat.argmax(1) == y).float().mean()))
            results['predictions']['target'].append(y)
            return {'y': y, 'y_hat': y_hat}, results

        return y_hat, results

    def postprocess_inference(self, sample=None, results=None, subset=None, with_labels=True, **kwargs):

        y_pred = torch.cat(results['predictions']['y_pred'])
        y_pred = as_numpy(torch.argmax(y_pred, dim=1))

        if with_labels:
            y_true = as_numpy(torch.cat(results['predictions']['target']))
            precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)
            results['metrics']['precision'] = precision
            results['metrics']['recall'] = recall
            results['metrics']['fscore'] = fscore
            results['metrics']['support'] = support

        return results


# Add experiment hyperparameter arguments
def get_covtype_parser():

    parser = get_beam_parser()

    parser.add_argument('--weight-factor', type=float, default=0., help='Squashing factor for the oversampling probabilities')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Smoothing factor in the Cross Entropy loss')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function in RuleNet')
    parser.add_argument('--objective', type=str, default='acc', help='The objective is the accuracy of the '
                                                                     'downstream task')
    parser.add_argument('--channels', type=int, default=256, help='Size of embedding')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout value for rule layers')
    parser.add_argument('--n-rules', type=int, default=128, help='Number of rules')
    parser.add_argument('--n-layers', type=int, default=5, help='Number of Residual Rule layers')
    parser.add_argument('--n-quantiles', type=int, default=20, help='Number of quantiles for the BetterEmbedding')
    parser.add_argument('--n-tables', type=int, default=15, help='Number of tables for the BetterEmbedding')
    parser.add_argument('--scheduler', type=str, default='reduce_on_plateau', help='Activation function in RuleNet')
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
                          f"--project-name=covtype --root-dir={root_dir} --algorithm=CovtypeAlgorithm --device=1",
                          f" --no-half --lr-d=1e-3 --lr-s=1e-2 --batch-size=512",
                          "--n-epochs=100 --clip-gradient=0 --parallel=1 --accumulate=1",
                          "--momentum=0.9 --beta2=0.99", weight_factor=.0, scheduler_patience=16,
                          weight_decay=1e-5, label_smoothing=0.,
                          k_p=.05, k_i=0.001, k_d=0.005, initial_mask=1,
                          path_to_data=path_to_data, dropout=.0, activation='gelu', channels=256, n_rules=128,
                          n_layers=2, scheduler_factor=1 / math.sqrt(10))

    experiment = Experiment(args)
    alg = experiment.fit(Alg=CovtypeAlgorithm, Dataset=CovtypeDataset, tensorboard_arguments=None)

    # ## Inference
    inference = alg('test')

    logger.info('Test inference results:')
    for n, v in inference.statistics['metrics'].items():
        logger.info(f'{n}:')
        logger.info(v)

