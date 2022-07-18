import torch
from torch import nn
import itertools
from collections import defaultdict
import numpy as np
import math
from .utils import slice_to_index, logger
from functools import partial

class PackedSet(object):

    def __init__(self, data, length=None):

        if length is None:
            self.length = torch.LongTensor([0] + [len(x) for x in data])
            self.data = torch.cat(data, dim=0)
        else:
            self.length = torch.LongTensor([0] + list(length))
            self.data = data

        self._offset = self.length.cumsum(dim=0)
        self.length = self.length[1:]
        self.offset = self._offset[:-1]

    def __len__(self):
        return len(self.offset)

    def clone(self):
        return PackedSet(self.data.clone(), self.length.clone())

    def to(self, device):
        self.data = self.data.to(device)
        self._offset = self._offset.to(device)
        self.length = self.length.to(device)
        self.offset = self.offset.to(device)
        return self

    def aggregate(self, func):
        return torch.stack([func(self.data[self._offset[i]:self._offset[i + 1]]) for i in range(len(self))])

    def __getitem__(self, index):

        index = slice_to_index(index)
        if issubclass(type(index), np.ndarray):
            index = torch.LongTensor(index)
        elif type(index) is int:
            index = torch.scalar_tensor(index, dtype=torch.int64)
        if issubclass(type(index), torch.Tensor):
            shape = index.shape
            if len(shape) == 0:
                return self.data[self._offset[index]:self._offset[index + 1]]
            elif len(shape) == 1:
                return PackedSet([self.data[self._offset[i]:self._offset[i + 1]] for i in index])
            else:
                raise NotImplementedError

        elif type(index) is tuple:
            assert len(index) == 2
            a, b = index
            return self.data[b + self._offset[a]]

        else:
            raise NotImplementedError

    def __repr__(self):
        return repr(self.data)


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


class LinearNet(nn.Module):

    def __init__(self, l_in, l_h=256, l_out=1, n_l=2, bias=True, activation='ReLU'):
        super().__init__()

        activation = getattr(nn, activation)
        self.lin = nn.Sequential(*([nn.Linear(l_in, l_h, bias=bias), activation()] if n_l > 1 else []),
                                 *sum([[nn.Linear(l_h, l_h, bias=bias), activation()] for _ in range(max(n_l - 2, 0))],
                                      []),
                                 nn.Linear(l_h if n_l > 1 else l_in, l_out, bias=bias))

    def forward(self, x):

        y = self.lin(x)
        return y.squeeze(1)


class MultipleScheduler(object):

    def __init__(self, multiple_optimizer, scheduler, *argc, **argv):

        self.schedulers = {}
        self.multiple_optimizer = multiple_optimizer

        for op in multiple_optimizer.optimizers.keys():
            self.schedulers[op] = scheduler(multiple_optimizer.optimizers[op], *argc, **argv)

    def step(self, *argc, **argv):
        for op in self.multiple_optimizer.optimizers.keys():
            self.schedulers[op].step(*argc, **argv)


class BeamOptimizer(object):

    def __init__(self, net, dense_args=None, clip=0, accumulate=1, amp=False,
                 sparse_args=None, dense_optimizer='AdamW', sparse_optimizer='SparseAdam'):

        sparse_optimizer = getattr(torch.optim, sparse_optimizer)
        dense_optimizer = getattr(torch.optim, dense_optimizer)

        if dense_args is None:
            dense_args = {'lr': 1e-3, 'eps': 1e-4}
        if sparse_args is None:
            sparse_args = {'lr': 1e-2, 'eps': 1e-4}

        self.clip = clip
        self.accumulate = accumulate
        self.iteration = 0
        self.amp = amp
        self.autocast_device = 'cpu' if 'cpu' in str(next(net.parameters()).device) else 'cuda'

        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.optimizers = {}

        sparse_parameters = []
        dense_parameters = []

        for nm, m in net.named_modules(remove_duplicate=True):
            is_sparse = BeamOptimizer.check_sparse(m)
            if is_sparse:
                for n, p in m.named_parameters(recurse=False):
                    if not any([p is pi for pi in sparse_parameters]):
                        sparse_parameters.append(p)
            else:
                for n, p in m.named_parameters(recurse=False):
                    if not any([p is pi for pi in dense_parameters]):
                        dense_parameters.append(p)

        if len(dense_parameters) > 0:
            self.optimizers['dense'] = dense_optimizer(dense_parameters, **dense_args)

        if len(sparse_parameters) > 0:
            self.optimizers['sparse'] = sparse_optimizer(sparse_parameters, **sparse_args)

        for k, o in self.optimizers.items():
            setattr(self, k, o)

    @staticmethod
    def prototype(dense_args=None, clip=0, accumulate=1, amp=False,
                  sparse_args=None, dense_optimizer='AdamW', sparse_optimizer='SparseAdam'):
        return partial(BeamOptimizer, dense_args=dense_args, clip=clip, accumulate=accumulate, amp=amp,
                       sparse_args=sparse_args, dense_optimizer=dense_optimizer, sparse_optimizer=sparse_optimizer)

    @staticmethod
    def check_sparse(m):
        return (issubclass(type(m), nn.Embedding) or issubclass(type(m), nn.EmbeddingBag)) and m.sparse

    def set_scheduler(self, scheduler, *argc, **argv):
        return MultipleScheduler(self, scheduler, *argc, **argv)

    def reset(self):
        self.iteration = 0
        for op in self.optimizers.values():
            op.state = defaultdict(dict)

        self.zero_grad(set_to_none=True)
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None

    def zero_grad(self, set_to_none=True):
        for op in self.optimizers.values():
            op.zero_grad(set_to_none=set_to_none)

    def apply(self, loss, training=False, set_to_none=True):

        with torch.autocast(self.autocast_device, enabled=False):
            self.iteration += 1

            if training:

                if self.amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if self.clip > 0:
                    for op in self.optimizers.values():
                        for pg in op.param_groups:

                            if self.amp:
                                for op in self.optimizers.values():
                                    self.scaler.unscale_(op)

                            torch.nn.utils.clip_grad_norm_(iter(pg['params']), self.clip)

                if not (self.iteration % self.accumulate):
                    self.step()
                    self.zero_grad(set_to_none=set_to_none)

    def step(self):
        for op in self.optimizers.values():
            if self.amp:
                self.scaler.step(op)
            else:
                op.step()

        if self.amp:
            self.scaler.update()

    def state_dict(self):
        return {k: op.state_dict() for k, op in self.optimizers.items()}

    def load_state_dict(self, state_dict, state_only=False):

        for k, op in self.optimizers.items():

            if state_only:
                state_dict[k]['param_groups'] = op.state_dict()['param_groups']

            op.load_state_dict(state_dict[k])


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


def beam_weights_initializer(m, method='none'):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def reset_networks_and_optimizers(networks=None, optimizers=None):
    if networks is not None:
        net_iter = networks.keys() if issubclass(type(networks), dict) else range(len(networks))
        for i in net_iter:
            for n, m in networks[i].named_modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()

    if optimizers is not None:
        opt_iter = optimizers.keys() if issubclass(type(optimizers), dict) else range(len(optimizers))
        for i in opt_iter:
            opt = optimizers[i]

            if type(opt) is BeamOptimizer:
                opt.reset()
            else:
                opt.state = defaultdict(dict)