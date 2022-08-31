import copy
import torch
from torch import nn
import itertools
from collections import defaultdict
import numpy as np
import math
from functools import partial


class MultipleScheduler(object):

    def __init__(self, multiple_optimizer, scheduler, *argc, **argv):

        self.schedulers = {}
        self.multiple_optimizer = multiple_optimizer

        for op in multiple_optimizer.optimizers.keys():
            self.schedulers[op] = scheduler(multiple_optimizer.optimizers[op], *argc, **argv)

    def step(self, *argc, **argv):
        for op in self.multiple_optimizer.optimizers.keys():
            self.schedulers[op].step(*argc, **argv)


class BeamScheduler(object):

    def __init__(self, optimizer, total_epochs, warmup=5, method='one_cycle', decay=math.sqrt(.1),):

        if method == 'one_cycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, )

        if warmup is not None and warmup > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=warmup)
            scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler, warmup_scheduler])
    def step(self, metric=None):
        self.scheduler.step()


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
        self.autocast_device = next(net.parameters()).device.type
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None

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

    def apply(self, loss, set_to_none=True, gradient=None, retain_graph=None, create_graph=False, inputs=None):

        with torch.autocast(self.autocast_device, enabled=False):
            self.iteration += 1

            if self.amp:
                self.scaler.scale(loss).backward(gradient=gradient, retain_graph=retain_graph,
                                                 create_graph=create_graph, inputs=inputs)
            else:
                loss.backward(gradient=gradient, retain_graph=retain_graph,
                              create_graph=create_graph, inputs=inputs)

            if self.clip > 0:
                for op in self.optimizers.values():
                    if self.amp:
                        self.scaler.unscale_(op)
                    for pg in op.param_groups:
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
        state_dict = {k: op.state_dict() for k, op in self.optimizers.items()}
        state_dict['scaler'] = self.scaler.state_dict() if self.scaler is not None else None
        return state_dict

    def load_state_dict(self, state_dict, state_only=False):

        for k, op in self.optimizers.items():

            if state_only:
                state_dict[k]['param_groups'] = op.state_dict()['param_groups']

            op.load_state_dict(state_dict[k])

        if self.scaler is not None and 'scaler' in state_dict.keys():
            self.scaler.load_state_dict(state_dict["scaler"])
