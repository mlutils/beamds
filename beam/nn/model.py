import copy
import math
import random
import sympy
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from torch import nn

from .optim import BeamOptimizer
from ..logging import beam_logger as logger
from ..utils import slice_to_index, hash_tensor, PackedTensor


class PositionalHarmonicExpansion(object):

    def __init__(self, xs, xe, delta=2, rank='cuda', n=100):

        self.xs = xs
        self.xe = xe

        delta = delta / (xe - xs)
        i = torch.arange(n+1, device=rank)[None, :]
        self.sigma = 2 * ((1 / delta) ** (i / n))

    def transform(self, x):

        x = x.unsqueeze(-1)

        sin = torch.sin(2 * np.pi / self.sigma * x)
        cos = torch.cos(2 * np.pi / self.sigma * x)

        out = torch.cat([sin, cos], dim=-1)
        return out


class GaussianHarmonicExpansion(object):

    def __init__(self, xs, xe, delta, n=100):

        self.xs = xs
        self.xe = xe

        delta = delta / (xe - xs)
        self.b = (1 / delta) * np.random.randn(n, 2)

    def transform(self, x, y):

        xy = np.stack([x, y], axis=0)

        xy = (xy - self.xs) / (self.xe - self.xs)
        sin = np.sin(2 * np.pi * self.b @ xy)
        cos = np.cos(2 * np.pi * self.b @ xy)

        return np.concatenate([sin, cos], axis=0).T


class LinearNet(nn.Module):

    def __init__(self, l_in, l_h=256, l_out=1, n_l=2, bias=True,
                 activation='ReLU', batch_norm=False, input_dropout=0.0, dropout=0.0):
        super().__init__()

        if type(activation) is str:
            activation = getattr(nn, activation)()

        if batch_norm:
            norm = nn.BatchNorm1d
        else:
            norm = nn.Identity

        if dropout > 0.0:
            d_layer = nn.Dropout
        else:
            d_layer = nn.Identity

        sequence = []
        if input_dropout > 0.0:
            sequence.append(nn.Dropout(input_dropout))

        if n_l > 1:
            sequence.extend([nn.Linear(l_in, l_h, bias=bias), activation, d_layer(dropout), norm(l_h)])
        sequence.extend(sum([[nn.Linear(l_h, l_h, bias=bias), activation, d_layer(dropout), norm(l_h)] for _ in range(max(n_l - 2, 0))], []))
        sequence.append(nn.Linear(l_h if n_l > 1 else l_in, l_out, bias=bias))

        self.lin = nn.Sequential(*sequence)

    def forward(self, x):

        y = self.lin(x)
        return y.squeeze(1)


class RuleLayer(nn.Module):

    def __init__(self, n_rules, e_dim_in, e_dim_out, bias=True, pos_enc=None, dropout=0.0):
        super().__init__()

        self.query = nn.Parameter(torch.empty((n_rules, e_dim_out)))
        nn.init.kaiming_uniform_(self.query, a=math.sqrt(5))

        self.key = nn.Linear(e_dim_in, e_dim_out, bias=bias)
        self.value = nn.Linear(e_dim_in, e_dim_out, bias=bias)
        self.e_dim_out = e_dim_out
        self.sparsemax = nn.Softmax(dim=1)
        self.tau = 1.

        if pos_enc is None:
            self.pos_enc = nn.Parameter(torch.empty((e_dim_in, e_dim_out)))
            nn.init.kaiming_uniform_(self.pos_enc, a=math.sqrt(5))

    def forward(self, x):
        b, nf, ne = x.shape

        pos = self.pos_enc[:nf].unsqueeze(0).repeat(b, 1, 1)
        x = x + pos

        k = self.key(x)
        v = self.value(x)
        q = self.query

        a = k @ q.T / math.sqrt(self.e_dim_out)
        a_prob = self.sparsemax(a / self.tau).transpose(1, 2)

        r = a_prob @ v

        return r, a_prob


class MHRuleLayer(nn.Module):

    def __init__(self, n_rules, n_features, e_dim_out, bias=True, pos_enc=None, dropout=0.0, n_head=8,
                 static_attention=False):
        super().__init__()

        self.query = nn.Parameter(torch.empty((n_rules, e_dim_out)))
        nn.init.kaiming_uniform_(self.query, a=math.sqrt(5))

        self.key = nn.Linear(e_dim_out, e_dim_out, bias=bias)
        self.value = nn.Linear(e_dim_out, e_dim_out, bias=bias)
        self.out = nn.Linear(e_dim_out, e_dim_out, bias=bias)
        self.e_dim_out = e_dim_out
        #         self.sparsemax = Sparsemax(dim=2)
        self.sparsemax = nn.Softmax(dim=2)
        self.tau = 1.
        self.n_head = n_head
        self.n_rules = n_rules

        if pos_enc is None:
            self.pos_enc = nn.Parameter(torch.empty((n_features, e_dim_out)))
            nn.init.kaiming_uniform_(self.pos_enc, a=math.sqrt(5))

    def forward(self, x):
        b, nf, ne = x.shape
        pos = self.pos_enc.unsqueeze(0).repeat(b, 1, 1)

        x = x + pos
        v = self.value(x)

        k = self.key(x)
        b, nf, ne = k.shape

        k = k.view(b, nf, self.n_head, ne // self.n_head).transpose(1, 2).reshape(b, self.n_head, nf, ne // self.n_head)
        v = v.view(b, nf, self.n_head, ne // self.n_head).transpose(1, 2).reshape(b, self.n_head, nf, ne // self.n_head)

        q = self.query.view(1, self.n_rules, self.n_head, ne // self.n_head).transpose(1, 2)

        a = k @ q.transpose(2, 3) / math.sqrt(self.e_dim_out // self.n_head)
        a_prob = self.sparsemax(a / self.tau).transpose(2, 3)

        r = (a_prob @ v).transpose(1, 2).reshape(b, self.n_rules, self.e_dim_out)
        r = self.out(r)

        return r, a_prob


class LazyLinearRegression(nn.Module):

    def __init__(self, quantiles, lr=0.001):
        super().__init__()

        self.lr = lr
        self.quantiles = quantiles
        self.register_buffer('a', None)
        self.register_buffer('b', None)

    def forward(self, e):

        _, e_dim = e.shape
        e = e.view(-1, self.quantiles + 2, e_dim)[:, 1:]

        x = torch.arange(self.quantiles + 1, device=e.device) / self.quantiles - 0.5
        x = x.reshape(1, -1, 1)

        if self.b is None:
            self.b = e.detach().mean(dim=1)
            self.a = self.quantiles * torch.diff(e.detach(), dim=1).mean(dim=1)

        else:

            grad = self.a.unsqueeze(1) * x + self.b.unsqueeze(1) - e.detach()
            self.b -= 2 * self.lr * grad.mean(dim=1)
            self.a -= 2 * self.lr * (grad * x).mean(dim=1)

        l = (e - (self.a.unsqueeze(1) * x + self.b.unsqueeze(1))) ** 2
        return l.mean(dim=1).sum()


class SplineEmbedding(nn.Module):

    def __init__(self, n_features, n_quantiles, emb_dim, n_tables=1, enable=True, init_weights=None, sparse=False):
        super().__init__()

        self.n_features = n_features
        self.emb_dim = emb_dim
        self.enable = enable

        self.n_tables = n_tables
        self.n_emb = (n_quantiles + 2) * n_features
        self.register_buffer('ind_offset', (n_quantiles + 2) * torch.arange(n_features, dtype=torch.int64).unsqueeze(0))
        self.register_buffer("n_quantiles", torch.FloatTensor([n_quantiles]))

        if init_weights is None:
            self.emb = nn.Embedding(self.n_emb * n_tables, emb_dim, sparse=sparse)
        else:

            none_val, base_1, base_0 = init_weights

            q = (torch.arange(n_quantiles + 1) / n_quantiles).view(1, 1, -1, 1)
            weights = base_1.unsqueeze(2) * q + base_0.unsqueeze(2) * (1 - q)

            weights = torch.cat([none_val.unsqueeze(2), weights], dim=2).reshape(-1, emb_dim)

            self.emb = nn.Embedding.from_pretrained(weights, sparse=sparse, freeze=False)

    def forward(self, x, mask, rand_table):

        # if self.enable:
        #     if not self.n_features:
        #         return torch.Tensor(len(x), 0, self.emb_dim, device=x.device, dtype=x.dtype)
        # else:
        #     return torch.zeros(*x.shape, self.emb_dim, device=x.device, dtype=x.dtype)

        th = 1e-3 if x.dtype == torch.float16 else 1e-6
        x = torch.clamp(x, min=th, max=1-th)

        offset = self.ind_offset + self.n_emb * rand_table

        xli = (x * self.n_quantiles).floor().long()
        xl = xli / self.n_quantiles

        xhi = (x * self.n_quantiles + 1).floor().long()
        xh = xhi / self.n_quantiles

        bl = self.emb((xli + 1) * mask + offset)
        bh = self.emb((xhi + 1) * mask + offset)

        delta = 1 / self.n_quantiles
        h = bh / delta * (x - xl).unsqueeze(2) + bl / delta * (xh - x).unsqueeze(2)

        return h


class PID(object):

    def __init__(self, k_p=0.05, k_i=0.005, k_d=0.005, T=20, clip=0.005):
        super().__init__()

        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.T = T
        self.eps_list = []
        self.val_loss_list = []
        self.clip = clip

    def __call__(self, eps, val_loss=None):

        self.eps_list.append(eps)
        self.eps_list = self.eps_list[-self.T:]
        if val_loss is not None:
            self.val_loss_list.append(val_loss)
            self.val_loss_list = self.val_loss_list[-self.T:]
        r = self.k_p * eps + self.k_i * np.sum(self.eps_list)
        if val_loss is None:
            if len(self.eps_list) > 1:
                r += self.k_d * (eps - self.eps_list[-2])
        else:
            if len(self.val_loss_list) > 1:
                r += self.k_d * (val_loss - self.val_loss_list[-2])
        r = self.clip * np.tanh(r / (self.clip + 1e-8))

        return r


class LazyQuantileNorm(nn.Module):

    def __init__(self, quantiles=100, momentum=.001,
                 track_running_stats=True, use_stats_for_train=True, boost=True, predefined=None, scale=True):
        super().__init__()

        quantiles = torch.arange(quantiles) / (quantiles - 1)

        n_quantiles = len(quantiles) if predefined is None else predefined.shape[-1]
        self.register_buffer("n_quantiles", torch.FloatTensor([n_quantiles]))

        boundaries = None if predefined is None else predefined
        self.predefined = False if predefined is None else True

        self.register_buffer("boundaries", boundaries)
        self.register_buffer("quantiles", quantiles)

        self.lr = momentum
        self.boost = boost
        self.scale = scale

        assert (not use_stats_for_train) or (use_stats_for_train and track_running_stats)

        self.track_running_stats = track_running_stats
        self.use_stats_for_train = use_stats_for_train

    def forward(self, x):

        dtype = x.dtype

        if not self.track_running_stats or self.boundaries is None:

            boundaries = torch.quantile(x.float(), self.quantiles.float(), dim=0).transpose(0, 1)
            boundaries = boundaries.type(dtype)

            if self.boundaries is None:
                self.boundaries = boundaries
        else:
            if self.training and not self.predefined:

                q = self.quantiles.view(1, 1, -1)
                b = self.boundaries.unsqueeze(0)
                xv = x.unsqueeze(-1).detach()
                q_th = (q * (xv - b) > (1 - q) * (b - xv)).type(dtype)
                q_grad = (- q * q_th + (1 - q) * (1 - q_th)) * (~torch.isinf(xv)).type(dtype)
                q_grad = q_grad.sum(dim=0)

                if self.boost:
                    q = self.quantiles.unsqueeze(0)
                    factor = (torch.max(1 / (q + 1e-3), 1 / (1 - q + 1e-3))) ** 0.5
                else:
                    factor = 1

                self.boundaries = self.boundaries - self.lr * torch.std(x, dim=0).unsqueeze(-1) * factor * q_grad
                self.boundaries = self.boundaries.sort(dim=1).values


        if (self.training and self.use_stats_for_train) or (
                not self.training and self.track_running_stats) or self.predefined:
            boundaries = self.boundaries

        xq = torch.searchsorted(boundaries, x.transpose(0, 1)).transpose(0, 1)
        if self.scale:
            xq = xq / self.n_quantiles

        return xq


class BetterEmbedding(torch.nn.Module):

    def __init__(self, numerical_indices, categorical_indices, n_quantiles, n_categories, emb_dim,
                 momentum=.001, track_running_stats=True, n_tables=15, initial_mask=1.,
                 k_p=0.05, k_i=0.005, k_d=0.005, T=20, clip=0.005, quantile_resolution=1e-4,
                 use_stats_for_train=True, boost=True, flatten=False, quantile_embedding=True, tokenizer=True,
                 qnorm_flag=False, kaiming_init=False, init_spline_equally=True, sparse=True, spline=True):

        super().__init__()

        self.categorical_indices = categorical_indices
        self.numerical_indices = numerical_indices

        n_feature_num = len(numerical_indices)
        n_features = len(torch.cat([categorical_indices, numerical_indices]))

        n_categories = n_categories + 1
        cat_offset = n_categories.cumsum(0) - n_categories
        self.register_buffer("cat_offset", cat_offset.unsqueeze(0))
        self.register_buffer("null_emb_cat", torch.FloatTensor(1, 0, emb_dim))

        self.flatten = flatten
        self.n_tables = n_tables

        self.n_emb = int(n_categories.sum())

        self.pid = PID(k_p=k_p, k_i=k_i, k_d=k_d, T=T, clip=clip)
        self.br = initial_mask

        if len(categorical_indices):
            self.emb_cat = nn.Embedding(1 + self.n_emb * n_tables, emb_dim, sparse=sparse)
        else:
            self.emb_cat = lambda x: self.null_emb_cat.repeat(len(x), 1, 1)

        if init_spline_equally:

            none_val = torch.randn(n_tables, n_feature_num, emb_dim)
            base_1 = torch.randn(n_tables, n_feature_num, emb_dim)
            base_2 = torch.randn(n_tables, n_feature_num, emb_dim)
            weights = (none_val, base_1, base_2)
        else:
            weights = None

        self.emb_num = None
        if spline:
            self.emb_num = SplineEmbedding(n_feature_num, n_quantiles, emb_dim, n_tables=n_tables,
                                       enable=quantile_embedding, init_weights=weights, sparse=sparse)

        self.llr = LazyLinearRegression(n_quantiles, lr=0.001)
        self.lambda_llr = 0
        if n_quantiles > 1:
            self.pid_llr = PID(k_p=1e-1, k_i=0, k_d=0, T=T, clip=1e-3)
        else:
            self.pid_llr = PID(k_p=1e-4, k_i=0, k_d=0, T=T, clip=0)

        self.qnorm = None
        if qnorm_flag:
            self.qnorm = LazyQuantileNorm(quantiles=int(1 / quantile_resolution), momentum=momentum,
                                      track_running_stats=track_running_stats,
                                      use_stats_for_train=use_stats_for_train, boost=boost)

        self.tokenizer = tokenizer
        if tokenizer:
            self.weight = nn.Parameter(torch.empty((1, n_features, emb_dim)))
            if kaiming_init:
                nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            else:
                nn.init.normal_(self.weight)
        else:
            self.register_buffer('weight', torch.zeros((n_features, emb_dim)))

        self.bias = nn.Parameter(torch.empty((1, n_features, emb_dim)))
        if kaiming_init:
            nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))
        else:
            nn.init.normal_(self.bias)

        self.emb_dim = emb_dim

    def step(self, train_loss, val_loss):

        self.br = min(max(0, self.br + self.pid((train_loss - val_loss) / val_loss)), 1)
        self.lambda_llr = min(max(0, self.lambda_llr - self.pid_llr((train_loss - val_loss) / val_loss)), 1)
        logger.info(f"br was changed to {self.br}")
        logger.info(f"lambda_llr was changed to {self.lambda_llr}")

    def get_llr(self):
        return self.llr(self.emb_num.emb.weight) * self.lambda_llr

    def forward(self, x, ensemble=True):

        x_num = x[:, self.numerical_indices]

        if self.qnorm is not None:
            x_num = self.qnorm(x_num)
        x_cat = x[:, self.categorical_indices].long()

        if ensemble:

            bernoulli = torch.distributions.bernoulli.Bernoulli(probs=self.br)
            mask_num = bernoulli.sample(sample_shape=x_num.shape).long().to(x.device)
            mask_cat = bernoulli.sample(sample_shape=x_cat.shape).long().to(x.device)

        else:
            mask_num = 1
            mask_cat = 1

        if self.training:
            rand_table = torch.randint(self.n_tables, size=(1, 1), device=x_cat.device)
        else:
            rand_table = torch.randint(self.n_tables, size=(len(x), 1), device=x_cat.device)

        x_cat = (x_cat + 1) * mask_cat + self.cat_offset + self.n_emb * rand_table

        e_cat = self.emb_cat(x_cat)

        if self.emb_num is not None:
            e_num = self.emb_num(x_num, mask_num, rand_table)
        else:
            e_num = torch.zeros(*x_num.shape, self.emb_dim, device=x_num.device, dtype=x_num.dtype)

        e = torch.cat([e_cat, e_num], dim=1)
        e = e + self.bias

        if self.tokenizer:
            x = torch.cat([torch.zeros_like(x_cat), x_num * mask_num], dim=1)
            y = self.weight * x.unsqueeze(-1)
            e = e + y
        if self.flatten:
            e = e.view(len(e), -1)

        return e


class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input


class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super().__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super().__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=input.device, dtype=input.dtype).view(1,
                                                                                                                     -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input


class BeamEnsemble(torch.nn.Module):

    def __init__(self, net, n_ensembles, optimizer=None):
        super().__init__()

        if isinstance(net, nn.Module):
            ensembles = []
            for _ in range(n_ensembles):
                new_net = copy_network(net)
                reset_network(new_net)
                ensembles.append(new_net)
        else:
            ensembles = [net() for _ in range(n_ensembles)]

        self.ensembles = nn.ModuleList(ensembles)
        self.n_ensembles = n_ensembles

        self.optimizers = None
        if optimizer is not None:
            self.optimizers = self.set_optimizers(optimizer)

        self.optimizer = None
        self.net = None
        self.active_model = None

    def set_optimizers(self, optimizer):

        try:
            self.optimizers = [optimizer(net) for net in self.ensembles]
        except TypeError:
            self.optimizers = [optimizer(net.paramters()) for net in self.ensembles]

        return self.optimizers

    def __len__(self):
        return self.n_ensembles

    def forward(self, x, reduction='mean', index=None):

        if self.training:
            if index is None:
                    index = random.randint(0, self.n_ensembles-1)

            self.net = self.ensembles[index]
            if self.optimizers is not None:
                self.optimizer = self.optimizers[index]
            self.active_model = index

            y = self.net(x)

        else:
            self.optimizer = None
            self.net = None
            self.active_model = None

            y = [net(x) for net in self.ensembles]
            if reduction == 'mean':
                y = torch.stack(y).mean(dim=0)
            elif reduction == 'none':
                y = torch.stack(y)
            else:
                raise NotImplementedError

        return y


class FeatureHasher(object):

    def __init__(self, num_embeddings, embedding_dim, n_classes=None, distribution='uniform', seed=None, device='cpu'):

        manual_seed = seed is not None
        with torch.random.fork_rng(devices=[device], enabled=manual_seed):

            if manual_seed:
                torch.random.manual_seed(seed)

            rand_func = torch.rand if distribution == 'uniform' else torch.randn
            self.weight = rand_func(num_embeddings, embedding_dim, device=device)

            if n_classes is not None:
                self.weight = (self.weight * n_classes).long()

    def __call__(self, x):
        return self.weight[x]


def beam_weights_initializer(net, black_list=None, white_list=None, zero_bias=True,
                             method=None, gain=None, nonlinearity='relu', method_argv=None,
                             nonlinearity_argv=None, method_linear=None, method_linear_argv=None,
                             method_conv=None, method_conv_argv=None, method_embedding=None, method_embedding_argv=None,
                             temperature=1):
    """

    @param net:
    @param black_list:
    @param white_list:
    @param zero_bias:
    @param method: [kaiming_uniform, kaiming_normal, xavier_uniform, xavier_normal, orthogonal, trunc_normal, sparse]
    @param gain:
    @param nonlinearity:
    @param nonlinearity_argv:
    @return:
    """

    def bias_init(m):
        if zero_bias:
            nn.init.zeros_(m.bias)
        else:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    if gain is None:
        if nonlinearity_argv is None:
            nonlinearity_argv = {}
        gain = nn.init.calculate_gain(nonlinearity, **nonlinearity_argv)

    # bert initializes embedding of 768 with .02 std and pytorch initializes with 1. stdout
    # so by default we use something in the middle which is also more reasonable for embedding of ~256 dimensions.
    if method_embedding is None:
        method_embedding = partial(nn.init.normal_, mean=0., std=.2)
    else:
        if method_embedding_argv is None:
            method_embedding_argv = {}
        method_embedding = partial(getattr(nn.init, f"{method_embedding}_"), **method_embedding_argv)

    if method is not None:
        if method_argv is None:
            method_argv = {}
        method = partial(getattr(nn.init, f"{method}_"), **method_argv)

    if method_conv is not None:
        if method_conv_argv is None:
            method_conv_argv = {}
        method_conv = partial(getattr(nn.init, f"{method_conv}_"), **method_conv_argv)
    elif method is not None:
        method_conv = method
    else:
        method_conv = partial(nn.init.xavier_uniform_, gain=gain)

    if method_linear is not None:
        if method_linear_argv is None:
            method_linear_argv = {}
        method_linear = partial(getattr(nn.init, f"{method_conv}_"), **method_linear_argv)
    elif method is not None:
        method_linear = method
    else:
        method_linear = partial(nn.init.xavier_uniform_, gain=gain)

    if black_list is None:
        black_list = []
    if white_list is None:
        # white_list = [n for n, p in net.named_parameters()]
        white_list = ['']

    def valid_param(n, suffix=None):
        if suffix is not None:
            n = f"{n}.{suffix}"
        return any([nw in n for nw in white_list]) and not any([nb in n for nb in black_list])

    for n, m in net.named_modules():
        if len(list(m.children())) or not valid_param(n):
            continue

        if 'Norm' in str(type(m)):
            if valid_param(n, suffix='weight') and hasattr(m, 'weight'):
                nn.init.ones_(m.weight)
            if valid_param(n, suffix='bias') and hasattr(m, 'bias'):
                nn.init.ones_(m.bias)
        elif 'Embedding' in str(type(m)):
            if valid_param(n, suffix='weight') and hasattr(m, 'weight'):
                method_embedding(m.weight)

        elif 'Linear' in str(type(m)):
            if valid_param(n, suffix='weight') and hasattr(m, 'weight'):
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                if fan_in > 1 and fan_out > 1:
                    method_linear(m.weight)
                else:
                    with torch.no_grad():
                        m.weight.normal_()
                        m.weight.data = torch.exp(m.weight.data / (temperature * math.sqrt(torch.numel(m.weight))))
                        m.weight.data = m.weight.data / m.weight.data.sum()

            if valid_param(n, suffix='bias') and hasattr(m, 'bias'):
                bias_init(m)
        elif 'Conv' in str(type(m)):
            if valid_param(n, suffix='weight') and hasattr(m, 'weight'):
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                if fan_in > 1 and fan_out > 1:
                    method_conv(m.weight)
                else:
                    with torch.no_grad():
                        m.weight.normal_()
                        m.weight.data = torch.exp(m.weight.data / (temperature * math.sqrt(torch.numel(m.weight))))
                        m.weight.data = m.weight.data / m.weight.data.sum()
            if valid_param(n, suffix='bias') and hasattr(m, 'bias'):
                bias_init(m)
        else:
            if len(list(m.parameters())) > 0:
                logger.warning(f"Beam weight initializer does not support layer type: {n}")


def reset_network(net):
    prev_hash = {n: hash_tensor(p) for n, p in net.named_parameters()}
    for n, m in net.named_modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_metadata()
    for n, p in net.named_parameters():
        if prev_hash[n] == hash_tensor(p):
            logger.warning(f"Parameter {n} was not reset. Check if its nn.Module supports .reset_parameters()")


def free_network_params(*nets):
    for net in nets:
        for p in net.parameters():
            p.requires_grad = True


def freeze_network_params(*nets):
    for net in nets:
        for p in net.parameters():
            p.requires_grad = False


def copy_network(net):
    return copy.deepcopy(net)


def soft_target_update(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * tau + param.data * (1.0 - tau))


def target_copy(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def reset_networks_and_optimizers(networks=None, optimizers=None):
    if networks is not None:
        net_iter = networks.keys() if isinstance(networks, dict) else range(len(networks))
        for i in net_iter:
            for n, m in networks[i].named_modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_metadata()

    if optimizers is not None:
        opt_iter = optimizers.keys() if isinstance(optimizers, dict) else range(len(optimizers))
        for i in opt_iter:
            opt = optimizers[i]

            if type(opt) is BeamOptimizer:
                opt.reset()
            else:
                opt.state = defaultdict(dict)



# Function to find an optimal offset based on bucket size
def find_optimal_offset(num_buckets, num_heads):
    """
    Finds an optimal offset to separate different heads in multi-head hashing.

    num_buckets: Total number of buckets (B)
    num_heads: Number of heads (H)

    Returns: Optimal offset (O)
    """
    min_offset = num_buckets // 2  # Ensuring sufficient separation
    max_offset = num_buckets

    # Find a prime number that does not divide num_buckets
    for candidate in range(min_offset, max_offset):
        if sympy.isprime(candidate) and num_buckets % candidate != 0:
            return candidate

    # Fallback: If no prime found, return a large odd number
    return min_offset | 1  # Ensure it's odd


# Define the class with adaptive bucket and head selection
class MultiHeadHashedEmbeddingAdaptive(nn.Module):
    def __init__(self, embedding_dim, num_buckets=None, num_heads=None, num_categories=None, collision_rate=.1,
                 offset=None):
        """
        Initializes the Multi-Head Hashed Embedding module with adaptive bucket and head selection.

        embedding_dim: Total embedding dimension
        num_buckets: Number of unique buckets (optional)
        num_heads: Number of independent heads (optional)
        num_categories: Number of unique categorical values (optional, used to compute optimal num_buckets and num_heads)
        collision_rate: Desired maximum collision rate (optional, used with num_categories)
        offset: Optional offset for shifting hashes between heads; if None, an optimal offset is chosen.
        """
        super().__init__()

        # Determine num_buckets and num_heads dynamically if not provided
        if num_buckets is None or num_heads is None:
            if num_categories is not None and collision_rate is not None:
                # Solve for the optimal num_buckets and num_heads given collision_rate
                num_buckets, num_heads = self._determine_buckets_and_heads(num_categories, embedding_dim,
                                                                           collision_rate)
            else:
                raise ValueError("Either provide (num_buckets and num_heads) or (num_categories and collision_rate).")

        assert embedding_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads  # Split embedding into heads
        self.num_buckets = num_buckets

        # If no offset is provided, find an optimal one
        self.offset = offset if offset else find_optimal_offset(num_buckets, num_heads)

        # Create independent embedding tables for each head
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_buckets, self.head_dim, sparse=True) for _ in range(num_heads)
        ])

        # Initialize weights
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def _determine_buckets_and_heads(self, num_categories, embedding_dim, target_collision_rate):
        """
        Computes optimal num_buckets and num_heads based on the desired collision rate.

        num_categories: Number of unique categorical values
        embedding_dim: Total embedding dimension
        target_collision_rate: Desired max collision rate (threshold)

        Returns: (optimal_num_buckets, optimal_num_heads)
        """

        if not (0 < target_collision_rate < 1):
            raise ValueError("target_collision_rate must be between 0 and 1 (exclusive).")

        num_heads = 2 ** int(math.log2(1 / target_collision_rate))

        # Ensure num_heads divides embedding_dim
        num_heads = math.gcd(embedding_dim, num_heads)

        num_buckets = max(1, int(num_categories / target_collision_rate))

        return num_buckets, num_heads

    def forward(self, hashed_value):
        """
        hashed_value: Precomputed integer hash value
        Returns: Concatenated embedding from all heads
        """
        indices = [(hashed_value + i * self.offset) % self.num_buckets for i in range(self.num_heads)]
        # indices = torch.tensor(indices, dtype=torch.long)

        # Fetch embeddings for all heads
        # embeddings = [self.embeddings[i](indices[i].unsqueeze(0)) for i in range(self.num_heads)]
        embeddings = [self.embeddings[i](indices[i]) for i in range(self.num_heads)]

        return torch.cat(embeddings, dim=-1)  # Concatenate embeddings from all heads