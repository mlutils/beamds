import math
from collections import defaultdict
from torch import nn
import torch
import copy
from .utils import tqdm_beam as tqdm
from .logger import beam_logger as logger
import numpy as np
from .optim import BeamOptimizer, BeamScheduler, MultipleScheduler
from .utils import finite_iterations, to_device, check_type, rate_string_format, recursive_concatenate, \
    stack_batched_results, as_numpy, beam_device, retrieve_name, filter_dict, \
    recursive_collate_chunks, is_notebook, DataBatch, pretty_format_number, nested_defaultdict
from .config import beam_arguments, get_beam_parser
from .dataset import UniversalBatchSampler, UniversalDataset, TransformedDataset
from .experiment import Experiment
from timeit import default_timer as timer
from .path import beam_path, PureBeamPath
from .processor import Processor
from .logger import beam_kpi, BeamResult
from .data import BeamData


class Algorithm(object):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, processors=None, dataset=None,
                 name=None, **kwargs):

        self._experiment = None
        self._default_hparams = None
        self.hparams_warnings = []
        self.trial = None

        self.hparams = hparams

        self.device = beam_device(self.get_hparam('device'))
        self.ddp = self.get_hparam('ddp')
        self.hpo = self.get_hparam('hpo')
        self.rank = self.get_hparam('rank')
        self.world_size = self.get_hparam('world_size')

        # some experiment hyperparameters
        self.half = self.get_hparam('half')
        self.enable_tqdm = hparams.enable_tqdm if hparams.tqdm_threshold == 0 or not hparams.enable_tqdm else None
        self.n_epochs = hparams.n_epochs
        self.swa_epochs = 0

        self.batch_size_train = hparams.batch_size_train
        self.batch_size_eval = hparams.batch_size_eval

        self.cuda = (self.device.type == 'cuda')
        self.pin_memory = self.cuda
        self.autocast_device = 'cuda' if self.cuda else 'cpu'
        amp_dtype = self.get_hparam('amp_dtype', default='float16')
        if self.cuda:
            self.amp_dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}[amp_dtype]
        else:
            if amp_dtype != 'bfloat16':
                logger.warning(f'Autocast on CPU is only supported for bfloat16, defaulting to bfloat16.')
            self.amp_dtype = torch.bfloat16
        self.amp = hparams.amp if self.cuda else False
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None

        self.scalers = {}
        self.epoch = 0

        self.networks = {}
        self.processors = {}
        self.swa_networks = {}
        self.inference_networks = {}

        self.optimizers = {}
        self.schedulers = {}
        self.swa_schedulers = {}
        self.schedulers_initial_state = {}

        self.optimizers_name_by_id = {}
        self.schedulers_name_by_id = {}
        self.schedulers_flat = {}
        self.optimizers_flat = {}
        self.optimizers_steps = {}
        self.epoch_length = None

        self.dataset = None
        self.persistent_dataloaders = {}
        self.dataloaders = {}
        self.eval_subset = None
        self.objective = None
        self.best_objective = None
        self.best_epoch = None
        self.best_state = False
        self._name = name

        self.add_components(networks=networks, optimizers=optimizers, schedulers=schedulers, processors=processors)

        if self.get_hparam('reload_path') is not None:
            self.load_checkpoint(hparams.reload_path)

        if hparams.store_initial_weights:
            self.initial_weights = self.save_checkpoint()

        if dataset is not None:
            self.load_dataset(dataset)

        self.cb_model = None
        self._is_notebook = None

    @property
    def is_notebook(self):
        if self._is_notebook is None:
            self._is_notebook = is_notebook()

        return self._is_notebook

    @property
    def default_hparams(self):

        if self._default_hparams is None:
            self._default_hparams = beam_arguments(get_beam_parser())

        return self._default_hparams

    @property
    def name(self):
        if self._name is None:
            self._name = retrieve_name(self)
        return self._name

    def get_hparam(self, hparam, specific=None, default=None):

        if type(specific) is list:
            for s in specific:
                if f"{s}_{hparam}" in self.hparams:
                    return getattr(self.hparams, f"{specific}_{hparam}")
        elif specific is not None and f"{specific}_{hparam}" in self.hparams:
            return getattr(self.hparams, f"{specific}_{hparam}")

        if hparam in self.hparams:
            return getattr(self.hparams, hparam)

        if default is None and hparam in self.default_hparams:
            default = self.default_hparams[hparam]
            setattr(self.hparams, hparam, default)

        if hparam not in self.hparams_warnings:
            experment_generated_hparams = ['ddp', 'hpo', 'rank', 'world_size', 'reload_path']
            if hparam in experment_generated_hparams:
                logger.error("Please pass to the algorithm the hparams object from the experiment: Algorithm(hparams=experiment.hparams).")
                self.hparams_warnings.extend(experment_generated_hparams)
            else:
                logger.warning(f"Hyperparameter: {hparam} was not found in the experiment hparams object. Returning {default}.")
            self.hparams_warnings.append(hparam)

        return default

    def to(self, device):

        logger.warning("Current implementation transforms only the networks dictionary. Don't use for training.")
        device = torch.device(device)
        self.device = device

        for net in self.networks.values():
            net.to(device)

        return self

    @staticmethod
    def get_parser():
        return get_beam_parser()

    @classmethod
    def from_pretrained(cls, path, override_hparams=None, hparams=None, Dataset=None, alg_args=None, alg_kwargs=None,
                             dataset_args=None, dataset_kwargs=None, reload_iloc=-1, reload_loc=None, reload_name=None,
                             **kwargs):

        if hparams is not None:
            override_hparams = hparams

        experiment = Experiment.reload_from_path(path, override_hparams=override_hparams, reload_iloc=reload_iloc,
                                                 reload_loc=reload_loc, reload_name=reload_name, **kwargs)

        alg = experiment.algorithm_generator(cls, Dataset=Dataset, alg_args=alg_args, alg_kwargs=alg_kwargs,
                                       dataset_args=dataset_args, dataset_kwargs=dataset_kwargs)

        if hparams is not None:
            experiment = Experiment(hparams)

        alg.experiment = experiment
        return alg

    def add_networks(self, networks):
        self.add_components(networks=networks)

    def add_processors(self, processors):
        self.add_components(processors=processors)

    def add_optimizers(self, optimizers):
        self.add_components(optimizers=optimizers)

    def add_schedulers(self, schedulers):
        self.add_components(schedulers=schedulers)

    def add_network(self, network, name):
        self.add_components(networks={name: network})

    def add_processor(self, processor, name):
        self.add_components(processors={name: processor})

    def add_optimizer(self, optimizer, name):
        self.add_components(optimizers={name: optimizer})

    def add_scheduler(self, scheduler, name):
        self.add_components(schedulers={name: scheduler})

    def add_components(self, networks=None, optimizers=None, schedulers=None, processors=None,
                       build_optimizers=True, build_schedulers=True, name='net'):

        if networks is None:
            networks = self.networks
        else:
            if isinstance(networks, nn.Module):
                networks = {name: networks}
            elif check_type(networks).minor == 'dict':
                pass
            else:
                raise NotImplementedError("Network type is unsupported")

            for k, net in networks.items():
                if k in self.networks:
                    self.networks.pop(k)
                    self.inference_networks.pop(k)
                    logger.warning(f"Found network with identical keys: {k}. Overriding previous network.")
                    if k in self.optimizers:
                        self.optimizers.pop(k)

                networks[k] = self.register_network(networks[k], name=k)
                self.networks[k] = networks[k]
                self.inference_networks[k] = net
                if self.get_hparam('swa') is not None:
                    self.swa_networks[k] = torch.optim.swa_utils.AveragedModel(net)

        if optimizers is None:
            optimizers = {}

        elif isinstance(optimizers, dict):
            for k, o in optimizers.items():
                if callable(o):
                    try:
                        optimizers[k] = o(networks[k])
                    except TypeError:
                        optimizers[k] = o(networks[k].parameters())
                else:
                    o.load_state_dict(o.state_dict())
                    optimizers[k] = o

        elif isinstance(optimizers, torch.optim.Optimizer) or isinstance(optimizers, BeamOptimizer):
            optimizers.load_state_dict(optimizers.state_dict())
            optimizers = {name: optimizers}

        elif callable(optimizers):
            try:
                optimizers = {name: optimizers(networks[name])}
            except TypeError:
                optimizers = {name: optimizers(networks[name].parameters())}
        else:
            raise NotImplementedError

        if build_optimizers:

            momentum = self.get_hparam('momentum')
            if momentum is None:
                momentum = self.get_hparam('beta1')

            for k, v in networks.items():
                if k not in optimizers:
                    optimizers[k] = BeamOptimizer(v, dense_args={'lr': self.get_hparam('lr_dense', k),
                                                                  'weight_decay': self.get_hparam('weight_decay', k),
                                                                  'betas': (self.get_hparam('momentum', k,
                                                                                            default=momentum),
                                                                            self.get_hparam('beta2', k)),
                                                                  'eps': self.get_hparam('eps', k),
                                                                  'capturable': self.get_hparam('capturable', k)},
                                                   sparse_args={'lr': self.get_hparam('lr_sparse', k),
                                                                'betas': (self.get_hparam('momentum', k,
                                                                                          default=momentum),
                                                                          self.get_hparam('beta2', k)),
                                                                'eps': self.get_hparam('eps', k)},
                                                   clip=self.get_hparam('clip_gradient', k), amp=self.amp,
                                                   accumulate=self.get_hparam('accumulate', k),
                                                   amp_dtype=self.amp_dtype)

        if processors is None:
            processors = {}
        elif isinstance(processors, Processor):
            processors = {processors.name: processors}
        elif isinstance(processors, list):
            processors = {p.name: p for p in processors}

        for k, v in processors.items():
            if k in self.processors:
                self.processors.pop(k)
                logger.warning(f"Found processor with identical keys: {k}. Overriding previous processor.")
            self.processors[k] = v

        if schedulers is None:
            schedulers = {}

        for k, opt in optimizers.items():
            self.optimizers[k] = opt

            if self.get_hparam('swa') is not None:

                kwargs = {'anneal_epochs': self.get_hparam('swa_anneal_epochs', k), 'anneal_strategy': 'cos'}

                if type(opt) is BeamOptimizer:
                    self.swa_schedulers[k] = opt.set_scheduler(torch.optim.swa_utils.SWALR,
                                                               self.get_hparam('swa_lr', k), **kwargs)
                else:
                    self.swa_schedulers[k] = torch.optim.swa_utils.SWALR(opt, self.get_hparam('swa_lr', k), **kwargs)

            if k in schedulers:
                self.schedulers[k] = schedulers[k]

            elif build_schedulers and self.get_hparam('scheduler', k) is not None:

                kwargs = {'warmup': self.get_hparam('scheduler_warmup', k), 'method': self.get_hparam('scheduler', k),
                          'step_type': self.get_hparam('schedulers_steps', k),
                          'cycle_momentum': True, 'base_momentum': self.get_hparam('cycle_base_momentum', k),
                          'max_momentum': self.get_hparam('cycle_max_momentum', k),
                          'patience': self.get_hparam('scheduler_patience', k),
                          'factor': self.get_hparam('scheduler_factor', k)}

                if type(opt) is BeamOptimizer:
                    scheduler = opt.set_scheduler(BeamScheduler, **kwargs)
                else:
                    scheduler = BeamScheduler(opt, **kwargs)

                self.schedulers[k] = scheduler

        self.refresh_optimizers_and_schedulers_pointers()

    def reset_optimizers_and_schedulers(self):

        self.schedulers = {}
        self.optimizers = {}
        self.swa_schedulers = {}
        self.add_components()
        self.load_dataset(self.dataset)

    def refresh_optimizers_and_schedulers_pointers(self):
        self.optimizers_name_by_id = {id(opt): k for k, opt in self.optimizers.items()}
        self.schedulers_name_by_id = {id(sch): k for k, sch in self.schedulers.items()}
        self.schedulers_flat = self.get_flat_schedulers()
        self.optimizers_flat = self.get_flat_optimizers()

    @property
    def experiment(self):

        if self._experiment is None:
            raise ValueError('No experiment is currently linked with the algorithm')

        logger.debug(f"Fetching the experiment which is currently associated with the algorithm")
        return self._experiment

    # a setter function
    @experiment.setter
    def experiment(self, experiment):
        logger.debug(f"The algorithm is now linked to an experiment directory: {experiment.root}")
        self.trial = experiment.trial
        self._experiment = experiment

    def apply(self, *losses, weights=None, training=True, optimizers=None, set_to_none=True, gradient=None,
              retain_graph=None, create_graph=False, inputs=None, iteration=None, reduction=None,
              name=None, results=None):

        if name is None:
            name = 'loss'
        total_loss = 0

        if len(losses) == 1 and isinstance(losses[0], dict):
            losses = losses[0]
        elif len(losses) == 1:
            losses = {name: losses[0]}
        else:
            losses = {f'{name}_{i}': l for i, l in enumerate(losses)}

        if weights is None:
            weights = {k: 1 for k in losses.keys()}
        elif isinstance(weights, dict):
            pass
        else:
            weights_type = check_type(weights, check_minor=False, check_element=False)
            if weights_type.major == 'scalar':
                weights = {next(iter(losses.keys())): weights}
            else:
                weights = {f'{name}_{i}': l for i, l in enumerate(weights)}

        for k, loss in losses.items():
            n = torch.numel(loss)

            rd = self.get_hparam('reduction', k) if reduction is None else reduction

            if n > 1:

                if rd == 'sum':
                    r = 1
                elif rd == 'mean':
                    r = n
                elif rd == 'mean_batch':
                    r = len(loss)
                elif rd == 'sqrt':
                    r = math.sqrt(n)
                elif rd == 'sqrt_batch':
                    r = math.sqrt(len(loss))
                else:
                    raise NotImplementedError

                loss = loss.sum()
                losses[k] = loss
                weights[k] = weights[k] / r

            total_loss = total_loss + loss * weights[k]

        if results is not None:
            if len(losses) > 1:
                for k, l in losses.items():
                    results['scalar'][f'{k}_s'].append(as_numpy(l * weights[k]))

                    if weights[k] > 1:
                        results['scalar'][f'{k}_w'].append(as_numpy(weights[k]))
                    elif weights[k] == 1:
                        pass
                    elif weights[k] == 0:
                        results['scalar'][f'{k}_w'].append(0)
                    else:
                        results['scalar'][f'{k}_f'].append(as_numpy(1 / weights[k]))

            results['scalar'][name].append(as_numpy(total_loss))

        loss = total_loss
        if training:

            if self.amp:
                if name is None:
                    scaler = self.scaler
                else:
                    if name not in self.scalers:
                        self.scalers[name] = torch.cuda.amp.GradScaler()
                    scaler = self.scalers[name]

            if optimizers is None:
                optimizers = self.optimizers_flat
            else:
                if isinstance(optimizers, torch.optim.Optimizer) or isinstance(optimizers, BeamOptimizer):
                    optimizers = [optimizers]
                optimizers = self.get_flat_optimizers(optimizers)

            with torch.autocast(self.autocast_device, dtype=self.amp_dtype, enabled=False):

                it = {}

                for k, op in optimizers.items():

                    if k not in self.optimizers_steps:
                        self.optimizers_steps[k] = 0
                    it[k] = self.optimizers_steps[k] if iteration is None else iteration
                    it[k] = (it[k] % self.get_hparam('accumulate', name))

                    if not it[k]:
                        op.zero_grad(set_to_none=set_to_none)

                if self.amp:
                    scaler.scale(loss).backward(gradient=gradient, retain_graph=retain_graph,
                                                create_graph=create_graph, inputs=inputs)
                else:
                    loss.backward(gradient=gradient, retain_graph=retain_graph,
                                  create_graph=create_graph, inputs=inputs)

                for k, op in optimizers.items():

                    if it[k] == self.get_hparam('accumulate', name) - 1:

                        clip = self.get_hparam('clip_gradient', k)
                        if clip > 0:
                            if self.amp:
                                scaler.unscale_(op)
                            for pg in op.param_groups:
                                torch.nn.utils.clip_grad_norm_(iter(pg['params']), clip)

                        if self.amp:
                            scaler.step(op)
                        else:
                            op.step()

                    self.optimizers_steps[k] = self.optimizers_steps[k] + 1

        return loss

    @staticmethod
    def split_names(k):
        if '/' in k:
            k, ki = k.split('/')
        else:
            ki = None
        return k, ki

    def load_dataset(self, dataset, batch_size_train=None, batch_size_eval=None,
                     oversample=None, weight_factor=None, expansion_size=None,timeout=0, collate_fn=None,
                     worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2,
                     dynamic=False, buffer_size=None, probs_normalization='sum', sample_size=100000):

        self.dataset = dataset

        batch_size_train = self.get_hparam('batch_size_train') if batch_size_train is None else batch_size_train
        batch_size_eval = self.get_hparam('batch_size_eval') if batch_size_eval is None else batch_size_eval
        oversample = (self.get_hparam('oversampling_factor') > 0) if oversample is None else oversample
        weight_factor = self.get_hparam('oversampling_factor') if weight_factor is None else weight_factor
        expansion_size = self.get_hparam('expansion_size') if expansion_size is None else expansion_size
        dynamic = self.get_hparam('dynamic_sampler') if dynamic is None else dynamic
        buffer_size = self.get_hparam('buffer_size') if buffer_size is None else buffer_size
        probs_normalization = self.get_hparam('probs_normalization') if probs_normalization is None else probs_normalization
        sample_size = self.get_hparam('sample_size') if sample_size is None else sample_size
        timeout = self.get_hparam('data_fetch_timeout') if timeout is None else timeout

        self.persistent_dataloaders = {}
        self.dataloaders = {}

        if not isinstance(dataset, dict):
            subsets = dataset.indices.keys()
        else:
            subsets = dataset.keys()

        self.eval_subset = 'validation' if 'validation' in subsets else 'test'

        for s in subsets:

            if not isinstance(dataset, dict):
                sampler = dataset.build_sampler(batch_size_eval, subset=s, persistent=False)
                d = dataset
            else:
                sampler = dataset[s].build_sampler(batch_size_eval, subset=None, persistent=False)
                d = dataset[s]

            self.dataloaders[s] = d.build_dataloader(sampler, num_workers=self.get_hparam('cpu_workers'),
                                                           pin_memory=self.pin_memory,
                                                           timeout=timeout, collate_fn=collate_fn,
                                                           worker_init_fn=worker_init_fn,
                                                           multiprocessing_context=multiprocessing_context,
                                                           generator=generator,
                                                           prefetch_factor=prefetch_factor)
        for s in ['train', self.eval_subset]:

            if not isinstance(dataset, dict):
                sampler = dataset.build_sampler(batch_size_train, subset=s, persistent=True, oversample=oversample,
                                                weight_factor=weight_factor, expansion_size=expansion_size,
                                                dynamic=dynamic, buffer_size=buffer_size,
                                                probs_normalization=probs_normalization,
                                                sample_size=sample_size)
                d = dataset
            else:
                sampler = dataset[s].build_sampler(batch_size_train, subset=None, persistent=True, oversample=oversample,
                                                   weight_factor=weight_factor, expansion_size=expansion_size,
                                                   dynamic=dynamic, buffer_size=buffer_size,
                                                   probs_normalization=probs_normalization,
                                                   sample_size=sample_size)
                d = dataset[s]

            self.persistent_dataloaders[s] = d.build_dataloader(sampler, num_workers=self.get_hparam('cpu_workers'),
                                                                      pin_memory=self.pin_memory,
                                                                      timeout=timeout, collate_fn=collate_fn,
                                                                      worker_init_fn=worker_init_fn,
                                                                      multiprocessing_context=multiprocessing_context,
                                                                      generator=generator,
                                                                      prefetch_factor=prefetch_factor)

        self.epoch_length = {'train': None, self.eval_subset: None}

        if self.get_hparam('epoch_length') is not None:
            if not isinstance(dataset, dict):
                l_train = len(dataset.indices['train'])
                l_eval = len(dataset.indices[self.eval_subset])
            else:
                l_train = len(dataset['train'])
                l_eval = len(dataset[self.eval_subset])

            self.epoch_length['train'] = int(np.round(self.get_hparam('epoch_length') * l_train / (l_train + l_eval)))
            self.epoch_length[self.eval_subset] = self.get_hparam('epoch_length') - self.epoch_length['train']

        if self.get_hparam('epoch_length_train') is not None:
            self.epoch_length['train'] = self.get_hparam('epoch_length_train')

        if self.get_hparam('epoch_length_eval') is not None:
            self.epoch_length[self.eval_subset] = self.get_hparam('epoch_length_eval')

        if self.epoch_length['train'] is None:
            dataset = self.persistent_dataloaders['train'].dataset
            self.epoch_length['train'] = len(dataset.indices['train'])

        if self.epoch_length[self.eval_subset] is None:
            dataset = self.persistent_dataloaders[self.eval_subset].dataset
            self.epoch_length[self.eval_subset] = len(dataset.indices[self.eval_subset])

        if self.get_hparam('scale_epoch_by_batch_size'):
            self.epoch_length[self.eval_subset] = math.ceil(self.epoch_length[self.eval_subset] / self.batch_size_eval)
            self.epoch_length['train'] = math.ceil(self.epoch_length['train'] / self.batch_size_train)

        if self.n_epochs is None:
            self.n_epochs = self.get_hparam('total_steps') // self.epoch_length['train']

        if self.get_hparam('swa') is not None:
            if int(self.get_hparam('swa')) == self.get_hparam('swa'):
                self.swa_epochs = int(self.get_hparam('swa'))
            else:
                self.swa_epochs = int(np.round(self.get_hparam('swa') * self.n_epochs))

        for k, scheduler in self.schedulers_flat.items():
            if type(scheduler) is BeamScheduler:

                k, ki = self.split_names(k)

                state = self.schedulers_initial_state[k] if k in self.schedulers_initial_state else None
                if ki is not None and state is not None:
                    state = state[ki]

                scheduler.update_total_steps(epochs=self.n_epochs,
                                             steps_per_epochs=self.epoch_length['train'], initial_state=state)

    def get_optimizer_name(self, opt):
        i = id(opt)
        if i in self.optimizers_name_by_id:
            return self.optimizers_name_by_id[i]
        return str(i)

    def get_scheduler_name(self, sch):
        i = id(sch)
        if i in self.schedulers_name_by_id:
            return self.schedulers_name_by_id[i]
        return str(i)

    def get_flat_optimizers(self, optimizers=None):

        if isinstance(optimizers, list):
            optimizers = {self.get_optimizer_name(opt): opt for opt in optimizers}
        elif optimizers is None:
            optimizers = self.optimizers

        optimizers_flat = {}
        for k, op in optimizers.items():
            if isinstance(op, BeamOptimizer):
                for ki, opi in op.optimizers.items():
                    if len(op.optimizers) > 1:
                        optimizers_flat[f'{k}/{ki}'] = opi
                    else:
                        optimizers_flat[k] = opi
            else:
                optimizers_flat[k] = op

        return optimizers_flat

    def get_flat_schedulers(self, schedulers=None):

        if isinstance(schedulers, list):
            schedulers = {self.get_scheduler_name(sch): sch for sch in schedulers}
        elif schedulers is None:
            schedulers = self.schedulers

        schedulers_flat = {}
        for k, scheduler in schedulers.items():
            if isinstance(scheduler, MultipleScheduler):
                for ki, sch in scheduler.schedulers.items():
                    if len(scheduler.schedulers) > 1:
                        schedulers_flat[f'{k}/{ki}'] = sch
                    else:
                        schedulers_flat[k] = sch
            else:
                schedulers_flat[k] = scheduler

        return schedulers_flat

    def register_network(self, net, name=None):

        if self.half:
            net = net.half()

        net = net.to(self.device)

        if self.ddp:

            if self.device.type == 'cuda':
                device_ids = [self.device]
            else:
                device_ids = None

            from torch.nn.parallel import DistributedDataParallel as DDP
            net_ddp = DDP(net, device_ids=device_ids,
                      find_unused_parameters=self.get_hparam('find_unused_parameters', name),
                      broadcast_buffers=self.get_hparam('broadcast_buffers', name))

            for a in dir(net):
                if a not in dir(net_ddp) and not a.startswith('_'):
                    setattr(net_ddp, a, getattr(net, a))
            net = net_ddp

        return net

    def return_dataset(self, subset):

        if type(subset) is str or isinstance(subset, torch.utils.data.DataLoader) \
                or isinstance(subset, torch.utils.data.Dataset):
            return True
        return False

    def build_dataloader(self, subset):

        if type(subset) is str:
            dataloader = self.dataloaders[subset]
        elif isinstance(subset, torch.utils.data.DataLoader):
            dataloader = subset
        elif isinstance(subset, torch.utils.data.Dataset):

            dataset = subset
            sampler = dataset.build_sampler(self.get_hparam('batch_size_eval'), persistent=False)
            dataloader = dataset.build_dataloader(sampler, num_workers=self.get_hparam('cpu_workers'),
                                                  pin_memory=self.pin_memory)
        else:

            subset_type = check_type(subset)
            index = None

            if type(subset) is DataBatch:
                index = subset.index
                dataset = UniversalDataset(subset.data, index=index)
            elif subset_type.minor in ['list', 'tuple']:
                dataset = UniversalDataset(*subset)
            elif subset_type.minor in ['dict']:
                dataset = UniversalDataset(**subset)
            else:
                dataset = UniversalDataset(subset)

            if index is None:
                index = len(dataset)
            sampler = UniversalBatchSampler(index, self.get_hparam('batch_size_eval'), shuffle=False,
                                            tail=True, once=True)
            dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=None,
                                                     num_workers=self.get_hparam('cpu_workers'),
                                                     timeout=self.get_hparam('data_fetch_timeout'),
                                                     pin_memory=self.pin_memory)
        return dataloader

    def schedulers_step(self, objective=None, step_type=None):
        if objective is None:
            objective = self.objective
        for k, scheduler in self.schedulers_flat.items():

            k, ki = self.split_names(k)

            if isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
                if self.get_hparam('schedulers_steps', specific=[f'{k}/{ki}', k]) == step_type:
                    scheduler.step()
            elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.get_hparam('schedulers_steps', specific=[f'{k}/{ki}', k]) == step_type:
                    scheduler.step(objective)
            elif isinstance(scheduler, BeamScheduler):
                scheduler.step(objective, step_type=step_type)
            else:
                try:
                    scheduler.step()
                except:
                    raise Exception(f"Unknown scheduler type: {type(scheduler)}")

    def data_generator(self, subset, max_iterations=None, persistent=False):

        if persistent:
            dataloader = self.persistent_dataloaders[subset]
        else:
            dataloader = self.build_dataloader(subset)
        for i, (ind, label, sample) in enumerate(dataloader):
            if max_iterations is not None and i >= max_iterations:
                break
            sample = to_device(sample, self.device, half=self.half)
            label = to_device(label, self.device, half=False)
            yield i, DataBatch(index=ind, label=label, data=sample)

    def preprocess_epoch(self, results=None, epoch=None, subset=None, training=True, **kwargs):
        '''
        :param aux: auxiliary data dictionary - possibly from previous epochs
        :param epoch: epoch number
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return results

    def iteration(self, sample=None, label=None, index=None,
                  results=None, counter=None, subset=None, training=True, **kwargs):
        '''
        :param sample: the data fetched by the dataloader
        :param aux: a dictionary of auxiliary data
        :param results: a dictionary of dictionary of lists containing results of
        :param subset: name of dataset subset (usually train/validation/test)
        :param training: train/test flag
        :return:
        loss: the loss fo this iteration
        aux: an auxiliary dictionary with all the calculated data needed for downstream computation (e.g. to calculate accuracy)
        '''
        return results

    def postprocess_epoch(self, sample=None, label=None, index=None,
                          results=None, epoch=None, subset=None, training=True, **kwargs):
        '''
        :param epoch: epoch number
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return results

    def epoch_iterator(self, n_epochs, subset, training):

        for n in range(n_epochs):

            t0 = timer()
            results = nested_defaultdict(list)

            if not training and self.rank > 0:
                yield results
                continue

            if n == self.n_epochs + self.swa_epochs:
                logger.warning("This is an extra epoch to calculate BN statistics. "
                               "It is not used for training so we set training=False.")
                training = False
                bu_networks = self.networks
                self.networks = self.swa_networks
                self.set_mode(training=True)

            else:
                self.set_mode(training=training)

            results = self.preprocess_epoch(results=results, epoch=n, training=training)

            data_generator = self.data_generator(subset, persistent=True)
            for i, (ind, label, sample) in tqdm(finite_iterations(data_generator, self.epoch_length[subset]),
                                  enable=self.enable_tqdm, notebook=(not self.ddp and self.is_notebook),
                                  threshold=self.get_hparam('tqdm_threshold'), stats_period=self.get_hparam('tqdm_stats'),
                                  desc=subset, total=self.epoch_length[subset]):

                with torch.autocast(self.autocast_device, dtype=self.amp_dtype, enabled=self.amp):
                    results = self.iteration(sample=sample, results=results, counter=i, training=training,
                                             index=ind, label=label)
                    objective = results['scalar'][self.get_hparam('objective')] \
                        if self.get_hparam('objective') in results['scalar'] else None

                    if training and n < self.n_epochs:
                        self.schedulers_step(objective, step_type='iteration')

                    if self.amp and training:
                        if self.scaler._scale is not None:
                            self.scaler.update()
                        for k, scaler in self.scalers.items():
                            if scaler._scale is not None:
                                scaler.update()

            results = stack_batched_results(results, batch_size=self.batch_size_train)
            results = self.postprocess_epoch(sample=sample, index=ind, label=label,
                                             results=results, epoch=n, training=training)

            batch_size = self.batch_size_train if training else self.batch_size_eval

            delta = timer() - t0
            n_iter = i + 1

            results['stats']['seconds'] = delta
            results['stats']['batches'] = n_iter
            results['stats']['samples'] = n_iter * batch_size
            results['stats']['batch_rate'] = rate_string_format(n_iter, delta)
            results['stats']['sample_rate'] = rate_string_format(n_iter * batch_size, delta)

            if n == self.n_epochs + self.swa_epochs:
                self.set_mode(training=False)
                self.networks = bu_networks

            yield results

    def preprocess_inference(self, results=None, subset=None, predicting=False, **argv):
        '''
        :param aux: auxiliary data dictionary - possibly from previous epochs
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return results

    def inference(self, sample=None, label=None, index=None, results=None, subset=None, predicting=False, **kwargs):
        '''
        :param sample: the data fetched by the dataloader
        :param aux: a dictionary of auxiliary data
        :param results: a dictionary of dictionary of lists containing results of
        :param subset: name of dataset subset (usually train/validation/test)
        :return:
        loss: the loss fo this iteration
        aux: an auxiliary dictionary with all the calculated data needed for downstream computation (e.g. to calculate accuracy)
        '''
        results = self.iteration(sample=sample, label=label, index=index, results=results, subset=subset, counter=0, training=False, **kwargs)
        return {}, results

    def postprocess_inference(self, sample=None, label=None, index=None, results=None, subset=None, predicting=False, **kwargs):
        '''
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return results

    def calculate_objective(self, results=None, **argv):
        '''
        This function calculates the optimization non-differentiable objective. It is used for hyperparameter optimization
        and for ReduceOnPlateau scheduling. It is also responsible for tracking the best checkpoint
        '''

        objective = None
        if 'objective' in results[self.eval_subset]['scalar']:
            objective_name = 'objective'
        else:
            objective_name = self.get_hparam('objective')

        if objective_name is not None and objective_name in results[self.eval_subset]['scalar']:

            objective = results[self.eval_subset]['scalar'][objective_name]
            objective_type = check_type(objective)
            if objective_type.major != 'scalar':
                objective = float(torch.mean(objective))

            self.objective = float(objective)
            if self.best_objective is None:
                self.best_objective = self.objective
                self.best_epoch = self.epoch
            elif self.objective > self.best_objective:
                logger.info(f"New best objective result: {pretty_format_number(self.objective)}")
                self.best_objective = self.objective
                self.best_epoch = self.epoch
                self.best_state = True
            else:
                self.best_state = False
            results['objective'] = objective
            results['global']['best_objective'] = self.best_objective
            results['global']['best_epoch'] = self.best_epoch
        elif objective_name is not None and objective_name not in results[self.eval_subset]['scalar']:
            logger.warning(f"The objective {objective_name} is missing from the validation results")

        return results, objective

    def report(self, objective, epoch=None, **argv):
        '''
        Use this function to report results to hyperparameter optimization frameworks
        also you can add key 'objective' to the results dictionary to report the final scores.
        '''

        if self.hpo == 'tune':
            if self.get_hparam('objective') is not None:
                kwargs = {self.get_hparam('objective'): objective}
            else:
                kwargs = {'objective': objective}
            from ray import tune
            tune.report(**kwargs)
        elif self.hpo == 'optuna':
            import optuna
            self.trial.report(objective, epoch)
            self.trial.set_user_attr('best_value', self.best_objective)
            self.trial.set_user_attr('best_epoch', self.best_epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

    def early_stopping(self, results=None, epoch=None, **kwargs):
        '''
        Use this function to early stop your model based on the results or any other metric in the algorithm class
        '''

        stop_at = self.get_hparam('stop_at')
        if stop_at is not None and stop_at > 0:
            if self.best_objective is not None:
                res = self.best_objective > stop_at
                if res:
                    logger.info(f"Stopping training at {self.best_objective} > {stop_at}")
                return res

        early_stopping_patience = self.get_hparam('early_stopping_patience')
        if early_stopping_patience is not None and early_stopping_patience > 0:
            res = self.epoch - self.best_epoch > early_stopping_patience
            if res:
                logger.info(f"Stopping training at epoch {self.epoch} - best epoch {self.best_epoch} > {early_stopping_patience}")
            return res

        return False

    # def train_catboost(self, eval_mode=False, enable_tqdm=None, fit_kwargs=None, **kwargs):
    #     """
    #     Use this function to train a catboost model
    #     @param eval_mode: set eval_mode to True to disable dataset augmentations
    #     @param kwargs:
    #     @return:
    #     """
    #     if fit_kwargs is None:
    #         fit_kwargs = {}
    #
    #     if enable_tqdm is None:
    #         enable_tqdm = self.enable_tqdm
    #
    #     self.set_mode(training=not eval_mode)
    #
    #     dataloader = self.build_dataloader('train')
    #     data_generator = self.data_generator(dataloader)
    #
    #
    #     logger.info(f"Build train dataset with {len(dataloader)} batches")
    #     data_train = []
    #
    #     for i, (ind, label, sample) in tqdm(data_generator, enable=enable_tqdm,
    #                                  threshold=self.get_hparam('tqdm_threshold'),
    #                                  stats_period=self.get_hparam('tqdm_stats'),
    #                                  notebook=(not self.ddp and self.is_notebook), total=len(dataloader)):
    #         data_train.append((ind, label, sample))
    #
    #     ind_train, label_train, data_train = recursive_collate_chunks(data_train)
    #     logger.info(f"Train dataset built with {len(data_train)} samples")
    #
    #     dataloader = self.build_dataloader(self.eval_subset)
    #     data_generator = self.data_generator(dataloader)
    #
    #     logger.info(f"Build eval dataset with {len(dataloader)} batches")
    #     data_eval = []
    #
    #     for i, (ind, label, sample) in tqdm(data_generator, enable=enable_tqdm,
    #                                     threshold=self.get_hparam('tqdm_threshold'),
    #                                     stats_period=self.hparams('tqdm_stats'),
    #                                     notebook=(not self.ddp and self.is_notebook), total=len(dataloader)):
    #             data_eval.append((ind, label, sample))
    #
    #     ind_eval, label_eval, data_eval = recursive_collate_chunks(data_eval)
    #     logger.info(f"Eval dataset built with {len(data_eval)} samples")
    #
    #     # find the label
    #     if 'y' in data_train:
    #         y_train = data_train['y']
    #         y_eval = data_eval['y']
    #     else:
    #         y_train = label_train
    #         y_eval = label_eval
    #
    #     y_train = as_numpy(y_train)
    #     y_eval = as_numpy(y_eval)
    #
    #     x_train = as_numpy(x_train)
    #     x_eval = as_numpy(x_eval)
    #
    #
    #     fit_args = {**fit_kwargs, 'X': x_train, 'y': y_train, 'eval_set': (x_eval, y_eval),}
    #
    #     y_type = check_type(y_train)
    #     if 'int' in y_type.element:
    #         from catboost import CatBoostClassifier
    #         catboost_model = CatBoostClassifier
    #     else:
    #         if self.get_hparam('cb_ranker'):
    #             from catboost import CatBoostRanker
    #             catboost_model = CatBoostRanker
    #
    #             if label_train is not None:
    #                 fit_args['group_id'] = label_train
    #         else:
    #             from catboost import CatBoostRegressor
    #             catboost_model = CatBoostRegressor
    #
    #     # find the data
    #     cb_kwargs = {'learning_rate': self.get_hparam('lr'),
    #                  'n_estimators': self.get_hparam('cb_n_estimators'),
    #                  'random_seed': self.get_hparam('seed'),
    #                  'l2_leaf_reg': self.get_hparam('weight_decay'),
    #     }
    #
    #
    #     cb_kwargs = {**cb_kwargs, **kwargs}
    #     cb = catboost_model(**cb_kwargs)
    #     cb.fit(**fit_args)
    #
    #     return

    def __call__(self, subset, predicting=False, enable_tqdm=None, max_iterations=None, head=None, eval_mode=True,
                 return_dataset=None, **kwargs):

        with torch.no_grad():

            self.set_mode(training=not eval_mode)
            results = nested_defaultdict(list)
            transforms = []
            index = []

            desc = subset if type(subset) is str else ('predict' if predicting else 'evaluate')

            if enable_tqdm is None:
                enable_tqdm = self.enable_tqdm

            if return_dataset is None:
                if predicting:
                    return_dataset = self.return_dataset(subset)
                    if not return_dataset:
                        logger.warning("Predicting: the inferred return type will be DataBatch and results statistics "
                                       "will be omitted. To avoid this behavior please provide a dataset or specify "
                                       "return_dataset=True")
                else:
                    return_dataset = True

            dataloader = self.build_dataloader(subset)
            dataset = dataloader.dataset

            batch_size = self.batch_size_eval
            if head is not None:
                max_iterations = math.ceil(head / batch_size)

            results = self.preprocess_inference(results=results, subset=subset, predicting=predicting, dataset=dataset,
                                                **kwargs)
            data_generator = self.data_generator(dataloader, max_iterations=max_iterations)
            total_iterations = len(dataloader) if max_iterations is None else min(len(dataloader), max_iterations)
            for i, (ind, label, sample) in tqdm(data_generator, enable=enable_tqdm,
                                  threshold=self.get_hparam('tqdm_threshold'), stats_period=self.get_hparam('tqdm_stats'),
                                  notebook=(not self.ddp and self.is_notebook), desc=desc, total=total_iterations):
                transform, results = self.inference(sample=sample, results=results, subset=subset, predicting=predicting,
                                         index=ind, **kwargs)
                transforms.append(transform)
                index.append(ind)

            index = torch.cat(index)
            transforms = recursive_concatenate(transforms)
            results = stack_batched_results(results, batch_size=batch_size)

            results = self.postprocess_inference(sample=sample, index=ind, transforms=transforms,
                                                 results=results, subset=subset, dataset=dataset,
                                                 predicting=predicting, **kwargs)

            if return_dataset:
                dataset = UniversalDataset(transforms, index=index)
                dataset.set_statistics(results)
            else:
                dataset = DataBatch(index=index, data=transforms, label=None)

        return dataset

    def __iter__(self):

        self.refresh_optimizers_and_schedulers_pointers()
        eval_generator = self.epoch_iterator(self.n_epochs+self.swa_epochs+int(self.swa_epochs > 0),
                                             subset=self.eval_subset, training=False)
        for i, train_results in enumerate(self.epoch_iterator(self.n_epochs+self.swa_epochs+int(self.swa_epochs > 0),
                                                              subset='train', training=True)):
            with torch.no_grad():
                eval_results = next(eval_generator)

            # add learning rate and momentum of schedulers_steps
            for k, scheduler in self.schedulers_flat.items():
                lr = scheduler.optimizer.param_groups[0]['lr']
                train_results['scalar'][f'lr_{k}'] = lr
                if type(scheduler) is BeamScheduler and scheduler.method in ['one_cycle']:
                    train_results['scalar'][f'momentum_{k}'] = scheduler.get_current_state()['momentum']

            self.epoch += 1
            
            results = {'train': train_results, self.eval_subset: eval_results, 'global': {}}
            eval_results, objective = self.calculate_objective(results=results)
            self.report(objective, i)

            if i+1 == self.n_epochs and self.swa_epochs > 0:
                logger.warning("Switching to SWA training")

            if i+1 >= self.n_epochs and self.swa_epochs > 0:
                for k, swa_model in self.swa_networks.items():
                    swa_model.update_parameters(self.networks[k])
                for k, sch in self.swa_schedulers.items():
                    sch.step()

                    lr = sch.optimizer.param_groups[0]['lr']
                    results['train']['scalar'][f'swalr_{k}'] = lr
            else:
                self.schedulers_step(objective=objective, step_type='epoch')

            yield results

            if self.early_stopping(results, i):
                return

    def set_mode(self, training=True):

        for net in self.networks.values():

            if training:
                net.train()
            else:
                net.eval()

        for dataloader in self.dataloaders.values():
            if hasattr(dataloader.dataset, 'train'):
                if training:
                    dataloader.dataset.train()
                else:
                    dataloader.dataset.eval()

    def save_checkpoint(self, path=None, aux=None, pickle_model=False):

        path = beam_path(path)
        state = {'aux': aux, 'epoch': self.epoch, 'best_objective': self.best_objective, 'best_epoch': self.best_epoch,}
        wrapper = copy.deepcopy if path is None else (lambda x: x)

        for k, net in self.networks.items():
            state[f"{k}_parameters"] = wrapper(net.state_dict())
            if pickle_model:
                state[f"{k}_model"] = net

        for k, optimizer in self.optimizers.items():
            state[f"{k}_optimizer"] = wrapper(optimizer.state_dict())

        for k, scheduler in self.schedulers.items():
            state[f"{k}_scheduler"] = wrapper(scheduler.state_dict())

        for k, processor in self.processors.items():
            state[f"{k}_processor"] = wrapper(processor.state_dict())

        for k, swa_scheduler in self.swa_schedulers.items():
            state[f"{k}_swa_scheduler"] = wrapper(swa_scheduler.state_dict())

        for k, swa_network in self.swa_networks.items():
            state[f"{k}_swa_network"] = wrapper(swa_network.state_dict())

        state['scaler'] = self.scaler.state_dict() if self.scaler is not None else None
        state['scalers'] = {k: scaler.state_dict()
                            if scaler is not None else None for k, scaler in self.scalers.items()}

        if path is not None:
            path.write(state, ext='.pt')
        else:
            return state

    def load_checkpoint(self, path_or_state, strict=True, networks=True, optimizers=True, schedulers=True,
                        processors=True, scaler=True, scalers=True, swa_schedulers=True, swa_networks=True, load_epoch=True):

        path_or_state = beam_path(path_or_state)

        if isinstance(path_or_state, PureBeamPath):
            logger.info(f"Loading network state from: {path_or_state}")
            state = path_or_state.read(ext='.pt', map_location=self.device)
        else:
            state = path_or_state

        for k, net in filter_dict(self.networks, networks).items():

            if f"{k}_parameters" in state.keys():
                s = state[f"{k}_parameters"]

                if not self.ddp:
                    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(s, 'module.')

                net.load_state_dict(s, strict=strict)
            else:
                logger.warning(f"Network {k} is missing from the state-dict")

        for k, net in filter_dict(self.swa_networks, swa_networks).items():

            if f"{k}_swa_network" in state.keys():
                s = state[f"{k}_swa_network"]

                if not self.ddp:
                    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(s, 'module.')

                net.load_state_dict(s, strict=strict)
            else:
                logger.warning(f"SWA Network {k} is missing from the state-dict")

        for k, optimizer in filter_dict(self.optimizers, optimizers).items():
            if f"{k}_optimizer" in state.keys():
                optimizer.load_state_dict(state[f"{k}_optimizer"])
            else:
                logger.warning(f"Optimizer {k} is missing from the state-dict")

        for k, processor in filter_dict(self.processors, processors).items():
            if f"{k}_processor" in state.keys():
                processor.load_state_dict(state[f"{k}_processor"])
            else:
                logger.warning(f"Processor {k} is missing from the state-dict")

        for k, scheduler in filter_dict(self.schedulers, schedulers).items():
            if f"{k}_scheduler" in state.keys():
                self.schedulers_initial_state[k] = state[f"{k}_scheduler"]
                try:
                    scheduler.load_state_dict(state[f"{k}_scheduler"])
                except AttributeError:
                    logger.warning("Tries to load scheduler which requires dataset info: please load dataset first")
            else:
                logger.warning(f"Scheduler {k} is missing from the state-dict")

        for k, swa_scheduler in filter_dict(self.swa_schedulers, swa_schedulers).items():
            if f"{k}_swa_scheduler" in state.keys():
                swa_scheduler.load_state_dict(state[f"{k}_swa_scheduler"])
            else:
                logger.warning(f"SWA Scheduler {k} is missing from the state-dict")

        if scaler:
            if self.scaler is not None and 'scaler' in state.keys():
                self.scaler.load_state_dict(state["scaler"])

        if "scalers" in state.keys():
            for k, s in filter_dict(self.scalers, scalers).items():
                if k in state["scalers"]:
                    self.scalers[k].load_state_dict(state["scalers"][k])

        if load_epoch and 'epoch' in state.keys():
            self.epoch = state['epoch']

        return state['aux']

    def optimize_for_inference(self, networks=True, **kwargs):

        self.inference_networks = {}

        for k, v in filter_dict(self.networks, networks).items():

            logger.warning(f"Optimizing model: {k} with torch compile")
            self.inference_networks[k] = torch.compile(v, **kwargs)


    def fit(self, dataset=None, dataloaders=None, timeout=0, collate_fn=None,
                   worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2, **kwargs):
        '''
        For training purposes
        '''

        def algorithm_generator_single(experiment, *args, **kwargs):

            if dataset is not None:
                self.load_dataset(dataset=dataset, dataloaders=dataloaders, timeout=0, collate_fn=None,
                                  worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2)

            return self

        if self.get_hparam('parallel') == 1:
            algorithm_generator = algorithm_generator_single
        else:
            raise NotImplementedError("To continue training in parallel mode: please re-run experiment() with "
                                      "your own algorithm generator and a new dataset")

        assert self._experiment is not None, "No experiment is linked with the current algorithm"

        return self._experiment(algorithm_generator, **kwargs)

    def evaluate(self, *args, **kwargs):
        '''
        For validation and test purposes (when labels are known)
        '''
        return self(*args, predicting=False, **kwargs)

    def predict(self, dataset, *args, lazy=False, kpi=True, **kwargs):
        '''
        For real data purposes (when labels are unknown)
        '''
        if lazy:
            return TransformedDataset(dataset, self, *args, **kwargs)

        if not kpi:
            return self(dataset, *args, predicting=True, **kwargs)

        @beam_kpi(BeamResult)
        def predict_wrapper(sample, algorithm=self, **kwargs):
            return algorithm(sample, predicting=True, **kwargs)

        return predict_wrapper(dataset, algorithm=self, *args, **kwargs)
