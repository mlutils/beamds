import math
from collections import defaultdict
from functools import partial

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..logging import beam_logger as logger


class MultipleScheduler(LRScheduler):

    def __init__(self, multiple_optimizer, scheduler, *argc, **argv):

        self.schedulers = {}
        self.multiple_optimizer = multiple_optimizer

        for op in multiple_optimizer.optimizers.keys():
            self.schedulers[op] = scheduler(multiple_optimizer.optimizers[op], *argc, **argv)

    def prepare(self, accelerator):
        for k, scheduler in self.schedulers.items():
            if isinstance(scheduler, BeamScheduler):
                scheduler.prepare(accelerator)
            else:
                self.schedulers[k] = accelerator.prepare_scheduler(scheduler)

    # def set_prepared(self, prepared):
    #     for k, scheduler in self.schedulers.items():
    #         if isinstance(scheduler, BeamScheduler):
    #             scheduler.set_prepared(prepared[k])
    #         else:
    #             self.schedulers[k] = prepared[k]

    def get_lr(self):
        lr = []
        for op in self.multiple_optimizer.optimizers.keys():
            lr.extend(self.schedulers[op].get_lr())
        return lr

    def step(self, *argc, **argv):
        for op in self.multiple_optimizer.optimizers.keys():
            self.schedulers[op].step(*argc, **argv)

    def state_dict(self):
        return {k: sch.state_dict() for k, sch in self.schedulers.items()}

    def load_load_state_dict(self, state):
        for k, sch in self.schedulers.items():
            if k in state:
                sch.load_state_dict(state[k])
            else:
                logger.error(f"Missing scheduler key from state_dict: {k}")


class BeamScheduler(LRScheduler):

    def __init__(self, optimizer, total_steps=None, epochs=None, steps_per_epochs=None,
                 warmup=5, method='one_cycle', step_type='epoch',
                 pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, start_factor=0.3,
                 max_momentum=0.95, div_factor=25.0, eta_min=1e-6, factor=math.sqrt(.1), patience=None,
                 threshold=0.0001, T_0=10, T_mult=1, threshold_mode='rel', cooldown=0, min_lr=1e-6):

        self.method = method
        self.epoch = 0
        self.warmup_scheduler = None
        self.warmup = warmup
        self.optimizer = optimizer
        self.last_lr = None
        self.last_momentum = None

        if method == 'one_cycle':
            self.step_type = 'iteration'
        else:
            self.step_type = step_type

        self.total_steps = self.get_total_steps(total_steps=total_steps, epochs=epochs,
                                                steps_per_epochs=steps_per_epochs)

        if method == 'one_cycle':

            max_lr = optimizer.param_groups[0]['lr']
            if self.total_steps is None:
                scheduler = partial(torch.optim.lr_scheduler.OneCycleLR, optimizer=optimizer, max_lr=max_lr,
                                                                pct_start=pct_start, anneal_strategy=anneal_strategy,
                                                                cycle_momentum=cycle_momentum, base_momentum=base_momentum,
                                                                max_momentum=max_momentum, div_factor=div_factor)
            else:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps=self.total_steps,
                                                            pct_start=pct_start, anneal_strategy=anneal_strategy,
                                                            cycle_momentum=cycle_momentum, base_momentum=base_momentum,
                                                            max_momentum=max_momentum, div_factor=div_factor)
        else:

            if self.warmup is not None and self.warmup > 0:
                self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor,
                                                                          total_iters=warmup)
                if self.total_steps is not None:
                    self.total_steps = self.total_steps - self.warmup

            if method == 'reduce_on_plateau':

                if patience is None and self.total_steps is not None:
                    patience = self.patience_heuristics(self.total_steps)

                if patience is not None:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=factor,
                                                                           patience=patience, threshold=threshold,
                                                                           threshold_mode=threshold_mode,
                                                                           cooldown=cooldown, min_lr=min_lr)
                else:
                    scheduler = partial(torch.optim.lr_scheduler.ReduceLROnPlateau, optimizer=optimizer, mode='max',
                                        factor=factor, threshold=threshold,
                                        threshold_mode=threshold_mode, cooldown=cooldown, min_lr=min_lr)

            elif method == 'cosine_annealing':

                if self.total_steps is None:
                    scheduler = partial(torch.optim.lr_scheduler.CosineAnnealingLR, optimizer=optimizer,
                                        eta_min=eta_min)
                else:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.total_steps, eta_min=eta_min)

            elif method == 'cawr':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult,
                                                                                 eta_min=eta_min)

            elif method == 'exponential':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=factor)

            elif method == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=patience, gamma=factor)

            elif method is None:
                scheduler = None
            else:
                logger.warning(f"Unsupported scheduler method: {method}, using None instead")

        self.scheduler = scheduler

    def prepare(self, accelerator):
        if self.warmup_scheduler is not None:
            self.warmup_scheduler = accelerator.prepare_scheduler(self.warmup_scheduler)
        if self.scheduler is not None:
            self.scheduler = accelerator.prepare_scheduler(self.scheduler)

    # def set_prepared(self, prepared):
    #     if self.warmup_scheduler is not None:
    #         self.warmup_scheduler = prepared['warmup_scheduler']
    #     if self.scheduler is not None:
    #         self.scheduler = prepared['scheduler']

    def get_total_steps(self, total_steps=None, epochs=None, steps_per_epochs=None):
        if epochs is not None and self.step_type == 'epoch':
            total_steps = epochs
        elif epochs is not None and steps_per_epochs is not None:
            total_steps = epochs * steps_per_epochs
        return total_steps

    @staticmethod
    def patience_heuristics(total_steps):

        # return 2 * int(np.log2(total_steps / 12.5))

        if total_steps > 400:
            return 10
        if total_steps > 200:
            return 8
        if total_steps > 100:
            return 6
        if total_steps > 50:
            return 4
        return 2

    def update_total_steps(self, total_steps=None, epochs=None, steps_per_epochs=None, initial_state=None):

        self.total_steps = self.get_total_steps(total_steps=total_steps, epochs=epochs,
                                                steps_per_epochs=steps_per_epochs)

        if self.warmup_scheduler is not None:
            self.total_steps = self.total_steps - self.warmup
        if type(self.scheduler) is partial:
            if self.method == 'one_cycle':
                self.scheduler = self.scheduler(total_steps=self.total_steps)
            elif self.method == 'cosine_annealing':
                self.scheduler = self.scheduler(T_max=self.total_steps)
            elif self.method == 'reduce_on_plateau':
                self.scheduler = self.scheduler(patience=self.patience_heuristics(self.total_steps))
            else:
                raise NotImplementedError(f"Method: {self.method} is still unsupported")

            if initial_state is not None:
                self.scheduler.load_state_dict(initial_state)

    def get_current_state(self):

        lr = self.optimizer.param_groups[0]['lr']
        if self.method in ['one_cycle']:
            if self.scheduler.use_beta1:
                momentum = self.optimizer.param_groups[0]['betas'][0]
            else:
                momentum = self.optimizer.param_groups[0]['momentum']
        else:
            momentum = None

        return {'lr': lr, 'momentum': momentum}

    def state_dict(self):
        return {'epoch': 0,
                'warmup_scheduler': None if self.warmup_scheduler is None else self.warmup_scheduler.state_dict(),
                'scheduler': self.scheduler.state_dict()}

    def load_state_dict(self, state):
        self.epoch = state['epoch']
        self.scheduler.load_state_dict(state['scheduler'])
        if self.warmup_scheduler is not None:
            self.warmup_scheduler.load_state_dict(state['warmup_scheduler'])

    def step(self, objective=None, step_type=None):

        if step_type != self.step_type and step_type is not None:
            return
        if self.warmup_scheduler is not None and self.epoch < self.warmup:
            self.warmup_scheduler.step()
        else:
            if self.method == 'reduce_on_plateau':
                self.scheduler.step(objective)
            else:
                if self.scheduler is not None:
                    self.scheduler.step()

        self.epoch = self.epoch + 1
        self.get_current_state()

    def get_lr(self):
        if self.warmup_scheduler is not None and self.epoch < self.warmup:
            return self.warmup_scheduler.get_lr()
        else:
            return self.scheduler.get_lr()


class BeamOptimizer(Optimizer):

    def __init__(self, net, dense_args=None, clip=0, accumulate=1,
                 amp=False, model_dtype='float16', sparse_args=None, dense_optimizer='AdamW',
                 sparse_optimizer='SparseAdam'):

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
        self.model_dtype = model_dtype
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

        self.param_groups = []
        self.defaults = None
        if len(dense_parameters) > 0:
            self.optimizers['dense'] = dense_optimizer(dense_parameters, **dense_args)
            self.param_groups.extend(self.optimizers['dense'].param_groups)

        if len(sparse_parameters) > 0:
            self.optimizers['sparse'] = sparse_optimizer(sparse_parameters, **sparse_args)
            self.param_groups.extend(self.optimizers['sparse'].param_groups)

        for k, o in self.optimizers.items():
            setattr(self, k, o)

    @property
    def state(self):
        combined_dict = {k: v for opt in self.optimizers.values() for k, v in opt.state.items()}
        return combined_dict

    def prepare(self, accelerator):
        for k, o in self.optimizers.items():
            self.optimizers[k] = accelerator.prepare_optimizer(o)

    # def set_prepared(self, prepared):
    #     for k, o in self.optimizers.items():
    #         self.optimizers[k] = prepared[k]

    @staticmethod
    def prototype(dense_args=None, clip=0, accumulate=1, amp=False,
                  sparse_args=None, dense_optimizer='AdamW', sparse_optimizer='SparseAdam'):
        return partial(BeamOptimizer, dense_args=dense_args, clip=clip, accumulate=accumulate, amp=amp,
                       sparse_args=sparse_args, dense_optimizer=dense_optimizer, sparse_optimizer=sparse_optimizer)

    @staticmethod
    def check_sparse(m):
        return (isinstance(m, nn.Embedding) or isinstance(m, nn.EmbeddingBag)) and m.sparse

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

        with torch.autocast(self.autocast_device, dtype=self.model_dtype, enabled=False):
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

    def step(self, closure=None):

        for op in self.optimizers.values():
            if self.amp:
                self.scaler.step(op)
            else:
                op.step(closure=closure)

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
