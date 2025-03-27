import math
import os
import numpy as np
from argparse import Namespace
from collections import defaultdict
from torch import nn
import torch
import copy
from timeit import default_timer as timer

from ..logging import beam_logger as logger
from ..nn import BeamOptimizer, BeamScheduler, MultipleScheduler, BeamNN, BeamDDP
from ..utils import (to_device, check_type, recursive_concatenate,
                     beam_device, filter_dict, cached_property,
                     is_notebook, DataBatch, dictionary_iterator, recursive_clone, set_item_with_tuple_key,
                     check_nvlink)
from ..dataset import UniversalBatchSampler, UniversalDataset, TransformedDataset
from ..experiment import Experiment, BeamReport
from ..path import beam_path, local_copy
from ..logging import beam_kpi, BeamResult
from ..processor import Processor
from ..base import beam_cache
from ..type import Types

from .core_algorithm import Algorithm


class NeuralAlgorithm(Algorithm):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, processors=None, dataset=None,
                 name=None, **kwargs):

        super().__init__(hparams, name=name, **kwargs)

        # the following are set by the experiment
        # some experiment hyperparameters

        self.scalers = {}
        self.networks = {}
        self.processors = {}
        self.swa_networks = {}

        self.optimizers = {}
        self.schedulers = {}
        self.swa_schedulers = {}
        self.schedulers_initial_state = {}

        self.optimizers_name_by_id = {}
        self.schedulers_name_by_id = {}
        self.schedulers_flat = {}
        self.optimizers_flat = {}
        self.optimizers_steps = {}
        self.datasets = {}
        self.persistent_dataloaders = defaultdict(dict)
        self.dataloaders = defaultdict(dict)

        self.add_components(networks=networks, optimizers=optimizers, schedulers=schedulers, processors=processors)

        if self.get_hparam('reload_path') is not None:
            self.load_checkpoint(hparams.reload_path, hparams=False)

        if hparams.store_initial_weights:
            self.initial_weights = self.save_checkpoint()

        if dataset is not None:
            self.load_dataset(dataset)

        self.training = False

    @cached_property
    def distributed_training(self):
        if self.experiment is None:
            self.no_experiment_message('distributed_training')
            return False
        return self.experiment.distributed_training

    @cached_property
    def distributed_training_framework(self):
        if self.experiment is None:
            self.no_experiment_message('distributed_training_framework')
            return None
        return self.experiment.distributed_training_framework

    @cached_property
    def rank(self):
        if self.experiment is None:
            self.no_experiment_message('rank')
            return 0
        return self.experiment.rank

    @cached_property
    def world_size(self):
        if self.experiment is None:
            self.no_experiment_message('world_size')
            return 1
        return self.experiment.world_size

    @cached_property
    def batch_size_train(self):
        return self.get_hparam('batch_size_train')

    @cached_property
    def batch_size_eval(self):
        return self.get_hparam('batch_size_eval')

    @cached_property
    def pin_memory(self):
        return self.is_cuda

    @cached_property
    def half(self):

        # maybe we should add bfloat16 + accelerate
        if self.native_training and self.model_dtype in [torch.float16, torch.bfloat16, torch.complex32]:
            return True

        return False

    @cached_property
    def autocast_device(self):
        return 'cuda' if self.is_cuda else 'cpu'

    @cached_property
    def brain(self):
        if self.model_dtype in [torch.bfloat16]:
            return True
        return False

    @cached_property
    def model_dtype(self):

        if self.amp:
            return None

        model_dtype = self.get_hparam('model_dtype', default='float32')
        if not self.is_cuda and model_dtype == 'float16':
            logger.warning(f'Autocast on CPU is only supported for bfloat16, defaulting to bfloat16.')
            model_dtype = 'bfloat16'

        return self.dtype_mapping(model_dtype)

    @staticmethod
    def dtype_mapping(model_dtype):
        model_mapping = {'float16': torch.float16, 'bfloat16': torch.bfloat16,
                         'float32': torch.float32, 'complex32': torch.complex32, 'complex64': torch.complex64, }
        return model_mapping[model_dtype]

    @staticmethod
    def accelerate_dtype_mapper(model_dtype):
        model_mapping = {torch.float16: 'fp16', torch.bfloat16: 'bf16', torch.float32: 'no'}
        return model_mapping[model_dtype]

    @cached_property
    def training_framework(self):
        return self.get_hparam('training_framework', default='torch')

    @cached_property
    def native_training(self):
        return self.training_framework == 'torch'

    @cached_property
    def mixed_precision_dtype(self):

        if self.native_training:
            return None

        model_dtype = self.get_hparam('model_dtype', default='float32')

        if self.amp:
            if not self.is_cuda and model_dtype == 'float16':
                logger.warning(f'Autocast on CPU is only supported for bfloat16, defaulting to bfloat16.')
                model_dtype = 'bfloat16'

        return self.dtype_mapping(model_dtype)

    @cached_property
    def amp(self):
        return self.training_framework == 'amp' if self.is_cuda else False

    # @cached_property
    # def deepspeed(self):
    #     return self.training_framework == 'deepspeed' and self.get_hparam('n_gpus') > 1

    @cached_property
    def deepspeed(self):
        return self.training_framework == 'deepspeed'

    @cached_property
    def scaler(self):
        return torch.cuda.amp.GradScaler() if self.amp else None

    @cached_property
    def swa_epochs(self):
        swa_epochs = 0
        if self.get_hparam('swa') is not None:
            if int(self.get_hparam('swa')) == self.get_hparam('swa'):
                swa_epochs = int(self.get_hparam('swa'))
            else:
                swa_epochs = int(np.round(self.get_hparam('swa') * self.n_epochs))
        return swa_epochs

    @cached_property
    def device(self):
        if self.in_cache('accelerator') and self.accelerator is not None and self.accelerator.device_placement:
            device = self.accelerator.device
        elif self._device is not None:
            device = self._device
        elif self.experiment is not None and hasattr(self.experiment, 'device'):
            device = beam_device(self.experiment.device)
        else:
            device = beam_device(self.get_hparam('device'))

        return device

    @cached_property
    def deepspeed_plugin(self):
        deepspeed_plugin = None
        if self.get_hparam('n_gpus') > 1 and self.accelerate:
            from accelerate.utils import DeepSpeedPlugin
            deepspeed_plugin = DeepSpeedPlugin(self.deepspeed_config(target='accelerate'),
                                               gradient_accumulation_steps=self.get_hparam('accumulate'),)

        return deepspeed_plugin

    @cached_property
    def fsdp_plugin(self):
        return None

    @cached_property
    def megatron_lm_plugin(self):
        return None

    @cached_property
    def accelerate_kwargs_handlers(self):
        return None

    @cached_property
    def gradient_accumulation_plugin(self):
        return None

    @cached_property
    def accelerate(self):
        return self.training_framework == 'accelerate'

    @cached_property
    def accelerator(self):

        acc = None
        if self.accelerate:
            from accelerate import Accelerator

            deepspeed_plugin = self.deepspeed_plugin
            device_placement = None if self.deepspeed else self.get_hparam('device_placement')

            if not check_nvlink():
                os.environ['NCCL_P2P_DISABLE'] = '1'
                os.environ['NCCL_IB_DISABLE'] = '1'

            acc = Accelerator(device_placement=device_placement,
                               split_batches=self.get_hparam('split_batches'),
                               mixed_precision=self.accelerate_dtype_mapper(self.mixed_precision_dtype),
                               gradient_accumulation_steps=self.get_hparam('accumulate'),
                               deepspeed_plugin=deepspeed_plugin,
                               fsdp_plugin=self.fsdp_plugin,
                               megatron_lm_plugin=self.megatron_lm_plugin,
                               kwargs_handlers=self.accelerate_kwargs_handlers,
                               cpu='cpu' == self.get_hparam('device'),
                               project_dir=self.experiment.experiment_dir if self.experiment is not None else None,
                               dispatch_batches=self.get_hparam('dispatch_batches'),
                               even_batches=True,
                               step_scheduler_with_optimizer=self.get_hparam('schedulers_steps') == 'iteration',
                               dynamo_backend=self.get_hparam('dynamo_backend'),
                               gradient_accumulation_plugin=self.gradient_accumulation_plugin,
                               )
            self.clear_cache('device')
        return acc

    @classmethod
    @property
    def special_state_attributes(cls):
        return super(NeuralAlgorithm, cls).special_state_attributes.union(['networks', 'optimizers', 'schedulers',
                                                                           'processors','datasets', 'scaler',
                                                                           'swa_networks', 'swa_schedulers',
                                                                           'schedulers_initial_state',
                                                                           'optimizers_name_by_id',
                                                                           'schedulers_name_by_id', 'schedulers_flat',
                                                                           'optimizers_flat', 'optimizers_steps'])

    def __getattr__(self, item):
        assert item != 'networks', 'Networks are not initialized yet'
        if item in self.networks:
            return self.networks[item]
        else:
            return super().__getattr__(item)

    @property
    def dataset(self):
        if not len(self.datasets):
            return None
        if len(self.datasets) == 1:
            return list(self.datasets.values())[0]
        return self.datasets

    def to(self, device):

        logger.warning("Current implementation transforms only the networks dictionary. Don't use for training.")
        device = torch.device(device)

        if not self.accelerate or not self.accelerator.device_placement:
            self._device = beam_device(device)
            self.clear_cache('device')

        for net in self.networks.values():
            net.to(device)

        return self

    @classmethod
    def from_pretrained(cls, path, override_hparams=None, hparams=None, dataset=None, alg_args=None, alg_kwargs=None,
                        dataset_args=None, dataset_kwargs=None, reload_iloc=None, reload_loc=None, reload_name=None,
                        device=None, **kwargs):

        if hparams is not None:
            override_hparams = hparams

        if device is not None:
            override_hparams = override_hparams or {}
            override_hparams['device'] = device

        experiment = Experiment.reload_from_path(path, override_hparams=override_hparams, reload_iloc=reload_iloc,
                                                 reload_loc=reload_loc, reload_name=reload_name, **kwargs)

        if alg_args is None and alg_kwargs is None:
            alg_args, alg_kwargs = experiment.get_alg_init_args()

            alg_kwargs.pop('experiment', None)
            dataset_e = alg_kwargs.pop('dataset', None)
            dataset = dataset or dataset_e

            # remove the hparams from the args list
            alg_args = alg_args[1:]

        alg = experiment.algorithm_generator(cls, dataset=dataset, alg_args=alg_args, alg_kwargs=alg_kwargs,
                                             dataset_args=dataset_args, dataset_kwargs=dataset_kwargs,
                                             store_init_args=False)

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
            elif check_type(networks).minor == Types.dict:
                pass
            else:
                raise NotImplementedError("Network type is unsupported")

            for k, net in networks.items():
                if k in self.networks:
                    self.networks.pop(k)
                    logger.warning(f"Found network with identical keys: {k}. Overriding previous network.")
                    if k in self.optimizers:
                        self.optimizers.pop(k)

                net = self.register_network(networks[k], name=k)

                self.networks[k] = net

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
                if len(list(v.parameters())):
                    if k not in optimizers and not self.deepspeed:
                        optimizers[k] = BeamOptimizer(v, dense_args={'lr': self.get_hparam('lr_dense', specific=k),
                                                                      'weight_decay': self.get_hparam('weight_decay', specific=k),
                                                                      'betas': (self.get_hparam('momentum', specific=k, default=momentum),
                                                                                self.get_hparam('beta2', specific=k)),
                                                                      'eps': self.get_hparam('eps', specific=k),},
                                                       sparse_args={'lr': self.get_hparam('lr_sparse', specific=k),
                                                                    'betas': (self.get_hparam('momentum', specific=k, default=momentum),
                                                                              self.get_hparam('beta2', specific=k)),
                                                                    'eps': self.get_hparam('eps', specific=k)},
                                                       clip=self.get_hparam('clip_gradient', specific=k), amp=self.amp,
                                                       accumulate=self.get_hparam('accumulate', specific=k),
                                                       model_dtype=self.mixed_precision_dtype)

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

                kwargs = {'anneal_epochs': self.get_hparam('swa_anneal_epochs', specific=k), 'anneal_strategy': 'cos'}

                if type(opt) is BeamOptimizer:
                    self.swa_schedulers[k] = opt.set_scheduler(torch.optim.swa_utils.SWALR,
                                                               self.get_hparam('swa_lr', specific=k), **kwargs)
                else:
                    self.swa_schedulers[k] = torch.optim.swa_utils.SWALR(opt, self.get_hparam('swa_lr', specific=k), **kwargs)

            if k in schedulers:
                self.schedulers[k] = schedulers[k]

            elif (build_schedulers and self.get_hparam('scheduler', specific=k) is not None
                  and not self.deepspeed):

                kwargs = {'warmup': self.get_hparam('scheduler_warmup', specific=k),
                          'method': self.get_hparam('scheduler', specific=k),
                          'step_type': self.get_hparam('schedulers_steps', specific=k),
                          'cycle_momentum': True,
                          'base_momentum': self.get_hparam('cycle_base_momentum', specific=k),
                          'max_momentum': self.get_hparam('cycle_max_momentum', specific=k),
                          'patience': self.get_hparam('scheduler_patience', specific=k),
                          'factor': self.get_hparam('scheduler_factor', specific=k)}

                if type(opt) is BeamOptimizer:
                    scheduler = opt.set_scheduler(BeamScheduler, **kwargs)
                else:
                    scheduler = BeamScheduler(opt, **kwargs)

                self.schedulers[k] = scheduler

        self.refresh_optimizers_and_schedulers_pointers()

    def deepspeed_config(self, target='accelerate'):
        from ..config import deepspeed_config_generator
        config = deepspeed_config_generator(self.hparams)
        return config

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

    def apply(self, *losses, weights=None, training=None, optimizers=None, set_to_none=True, gradient=None,
              retain_graph=None, create_graph=False, inputs=None, iteration=None, reduction=None, name=None,
              report=True):

        if training is None:
            training = self.training

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
            weights_type = check_type(weights, minor=False, element=False)
            if weights_type.major == Types.scalar:
                weights = {next(iter(losses.keys())): weights}
            else:
                weights = {f'{name}_{i}': l for i, l in enumerate(weights)}

        for k, loss in losses.items():
            n = torch.numel(loss)

            rd = self.get_hparam('reduction', specific=k) if reduction is None else reduction

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

        if report:
            if len(losses) > 1:
                for k, l in losses.items():

                    self.report_scalar(f'{k}_s', l * weights[k])

                    if weights[k] > 1:
                        self.report_scalar(f'{k}_w', weights[k])
                    elif weights[k] == 1:
                        pass
                    elif weights[k] == 0:
                        self.report_scalar(f'{k}_w', 0)
                    else:
                        self.report_scalar(f'{k}_f', 1 / weights[k])

            self.report_scalar(name, total_loss)

        loss = total_loss
        if training:

            if self.deepspeed:
                engines = optimizers
                if engines is None:
                    engines = list(self.networks.values())
                for eng in engines:
                    eng.zero_grad()
                for eng in engines:
                    eng.backward(loss, retain_graph=retain_graph, scale_wrt_gas=True)
                for eng in engines:
                    eng.step()

            else:
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

                with torch.autocast(self.autocast_device, dtype=self.mixed_precision_dtype, enabled=False):

                    it = {}

                    for k, op in optimizers.items():

                        if k not in self.optimizers_steps:
                            self.optimizers_steps[k] = 0
                        it[k] = self.optimizers_steps[k] if iteration is None else iteration
                        it[k] = (it[k] % self.get_hparam('accumulate', specific=name))

                        if not it[k]:
                            op.zero_grad(set_to_none=set_to_none)

                    if self.amp:
                        scaler.scale(loss).backward(gradient=gradient, retain_graph=retain_graph,
                                                    create_graph=create_graph, inputs=inputs)
                    elif self.accelerate:
                        self.accelerator.backward(loss, retain_graph=retain_graph)
                    elif self.deepspeed:
                        # runs backpropagation
                        op.backward(loss, retain_graph=retain_graph)

                    else:
                        loss.backward(gradient=gradient, retain_graph=retain_graph,
                                      create_graph=create_graph, inputs=inputs)

                    for k, op in optimizers.items():

                        if it[k] == self.get_hparam('accumulate', specific=name) - 1:

                            clip = self.get_hparam('clip_gradient', specific=k)
                            if clip > 0:
                                if self.amp:
                                    scaler.unscale_(op)
                                if self.accelerator is not None and self.accelerator.sync_gradients:
                                    for pg in op.param_groups:
                                        self.accelerator.clip_grad_norm_(pg['params'], clip)
                                else:
                                    for pg in op.param_groups:
                                        torch.nn.utils.clip_grad_norm_(iter(pg['params']), clip)

                            if self.amp:
                                scaler.step(op)
                            elif self.deepspeed:
                                # weight update
                                op.step()
                            else:
                                # from torch.cuda.amp import grad_scaler
                                # from accelerate import optimizer
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

    def load_dataset(self, dataset, *args, name=None, **kwargs):
        if name is None:
            name = 'dataset'
        self.load_datasets({name: dataset}, *args, **kwargs)

    def load_datasets(self, datasets, set_epoch_length='first', batch_size_train=None, batch_size_eval=None,
                     oversample=None, weight_factor=None, expansion_size=None,timeout=0, collate_fn=None,
                     worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2,
                     dynamic=False, buffer_size=None, train_on_tail=True, probs_normalization='sum', sample_size=100000):

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
        train_on_tail = self.get_hparam('train_on_tail') if train_on_tail is None else train_on_tail

        for i, (k, dataset) in enumerate(datasets.items()):

            self.datasets[k] = dataset
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

                self.dataloaders[k][s] = d.build_dataloader(sampler, num_workers=self.get_hparam('cpu_workers'),
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
                                                    tail=train_on_tail,
                                                    sample_size=sample_size)
                    d = dataset
                else:
                    sampler = dataset[s].build_sampler(batch_size_train, subset=None, persistent=True,
                                                       oversample=oversample, weight_factor=weight_factor,
                                                       expansion_size=expansion_size,
                                                       dynamic=dynamic, buffer_size=buffer_size,
                                                       probs_normalization=probs_normalization,
                                                       tail=train_on_tail,
                                                       sample_size=sample_size)
                    d = dataset[s]

                self.persistent_dataloaders[k][s] = {'dataloader': d.build_dataloader(sampler,
                                                                  num_workers=self.get_hparam('cpu_workers'),
                                                                  pin_memory=self.pin_memory,
                                                                  timeout=timeout, collate_fn=collate_fn,
                                                                  worker_init_fn=worker_init_fn,
                                                                  multiprocessing_context=multiprocessing_context,
                                                                  generator=generator,
                                                                  prefetch_factor=prefetch_factor),
                                                  'dataset': d,
                                                  'sampler': sampler,
                                                  }
                self.persistent_dataloaders[k][s]['iterator'] = iter(self.persistent_dataloaders[k][s]['dataloader'])

            if (set_epoch_length == 'first' and i == 0) or k == set_epoch_length:
                self.set_epoch_length(dataset)

    def set_epoch_length(self, dataset):

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
            self.epoch_length['train'] = len(dataset.indices['train'])

        if self.epoch_length[self.eval_subset] is None:
            self.epoch_length[self.eval_subset] = len(dataset.indices[self.eval_subset])

        if self.get_hparam('scale_epoch_by_batch_size'):
            self.epoch_length[self.eval_subset] = math.ceil(self.epoch_length[self.eval_subset] / self.batch_size_eval)
            self.epoch_length['train'] = math.ceil(self.epoch_length['train'] / self.batch_size_train)

        if self.n_epochs is None:
            self._n_epochs = self.get_hparam('total_steps') // self.epoch_length['train']

        self.set_hparam('epoch_length_train', self.epoch_length['train'])
        self.set_hparam('epoch_length_eval', self.epoch_length[self.eval_subset])

        for k, scheduler in self.schedulers_flat.items():
            if type(scheduler) is BeamScheduler:

                k, ki = self.split_names(k)

                state = self.schedulers_initial_state[k] if k in self.schedulers_initial_state else None
                if ki is not None and state is not None:
                    state = state[ki]

                scheduler.update_total_steps(epochs=self.n_epochs,
                                             steps_per_epochs=self.epoch_length['train'], initial_state=state)

    def prepare_accelerate(self):

        elements = []
        elements_origin = []
        elements_keys = []

        for k, v in self.networks.items():
            elements.append(v)
            elements_origin.append(self.networks)
            elements_keys.append((k,))
        for k, v in self.optimizers.items():
            elements.append(v)
            elements_origin.append(self.optimizers)
            elements_keys.append((k,))
        for k, v in self.schedulers.items():
            elements.append(v)
            elements_origin.append(self.schedulers)
            elements_keys.append((k,))
        # don't prepare dataloaders if you know the batch size
        # for k, v in self.persistent_dataloaders.items():
        #     for s, v in v.items():
        #         elements.append(v['dataloader'])
        #         elements_origin.append(self.persistent_dataloaders)
        #         elements_keys.append((k, s, 'dataloader'))

        modified_elements = self.accelerator.prepare(*elements)

        for e, o, kk in zip(modified_elements, elements_origin, elements_keys):
            set_item_with_tuple_key(o, kk, e)

        self.refresh_optimizers_and_schedulers_pointers()

    def prepare_deepspeed(self):

        import deepspeed
        for k, net in self.networks.items():
            opt = self.optimizers[k] if k in self.optimizers else None
            sch = self.schedulers[k] if k in self.schedulers else None

            # os.environ['LOCAL_RANK'] = str(self.rank)
            # # OMPI_COMM_WORLD_LOCAL_RANK
            # os.environ['OMPI_COMM_WORLD_RANK'] = str(self.rank)
            # # OMPI_COMM_WORLD_RANK
            # os.environ['OMPI_COMM_WORLD_SIZE'] = str(self.world_size)

            logger.critical(f"LOCAL_RANK: {os.environ['LOCAL_RANK']}")
            logger.critical(f"self.rank: {self.rank}")
            deepspeed_config = Namespace(local_rank=self.rank, device_rank=int(self.device.index),
                                         deepspeed_config=self.deepspeed_config(target='deepspeed'))

            net, opt, _, sch = deepspeed.initialize(deepspeed_config, model=net, optimizer=opt, lr_scheduler=sch)

            self.networks[k] = net

            self.optimizers[k] = opt
            if sch is not None:
                self.schedulers[k] = sch

        self.refresh_optimizers_and_schedulers_pointers()

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

    @property
    def is_cuda(self):
        if self.device is not None and self.device.type == 'cuda':
            return True
        return False

    @property
    def ddp(self):
        return self.distributed_training and self.distributed_training_framework == 'ddp'

    def register_network(self, net, name=None):

        net = BeamNN.from_module(net, name=name, hparams=self.hparams)

        if self.device is not None:
            if self.accelerate:
                net = net.to(self.device)
            else:
                net = net.to(self.device, dtype=self.model_dtype)

        if self.ddp:

            if self.is_cuda:
                device_ids = [self.device]
            else:
                device_ids = None

            from torch.nn.parallel import DistributedDataParallel as DDP
            from types import FunctionType, MethodType

            net_ddp = DDP(net, device_ids=device_ids,
                      find_unused_parameters=self.get_hparam('find_unused_parameters', specific=name),
                      broadcast_buffers=self.get_hparam('broadcast_buffers', specific=name))

            for attr_name in dir(net):
                if attr_name not in dir(net_ddp) and not attr_name.startswith('_'):
                    # setattr(net_ddp, a, getattr(net, a))
                    attr = getattr(net.__class__, attr_name, None)
                    if isinstance(attr, property):
                        # It's a property, copy it as a property
                        setattr(net_ddp.__class__, attr_name, attr)
                    elif attr_name not in dir(net_ddp) or isinstance(getattr(net, attr_name),
                                                                     (FunctionType, MethodType)):
                        # It's not a property, but a regular attribute or a function/method; copy it directly
                        setattr(net_ddp, attr_name, getattr(net, attr_name))
            net = net_ddp

            # net = BeamDDP(net, device_ids=device_ids,
            #               find_unused_parameters=self.get_hparam('find_unused_parameters', specific=name),
            #               broadcast_buffers=self.get_hparam('broadcast_buffers', specific=name))

        if not self.accelerate and self.get_hparam('compile_network'):
            net = net.optimize('compile')

        return net

    def return_dataset(self, subset):

        if type(subset) is str or isinstance(subset, torch.utils.data.DataLoader) \
                or isinstance(subset, torch.utils.data.Dataset):
            return True
        return False

    def build_dataloaders(self, subset):
        return {k: self.build_dataloader(subset, name=k) for k in self.dataloaders.keys()}

    def build_dataloader(self, subset, name='dataset'):

        if type(subset) is str:
            dataloader = self.dataloaders[name][subset]
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

            if 'DataBatch' in str(type(subset)):
                dataset = UniversalDataset(subset.data, index=subset.index, label=subset.label)
                index = subset.index
            elif subset_type.minor in [Types.list, Types.tuple]:
                dataset = UniversalDataset(*subset)
            elif subset_type.minor in [Types.dict]:
                dataset = UniversalDataset(**subset)
            else:
                dataset = UniversalDataset(subset)

            if index is None:
                if hasattr(dataset, 'index') and dataset.index is not None:
                    index = dataset.index
                else:
                    index = len(dataset)
            sampler = UniversalBatchSampler(index, self.get_hparam('batch_size_eval'), shuffle=False,
                                            tail=True, once=True)

            if str(dataset.device) == 'cpu' and self.pin_memory:
                pin_memory = True
            else:
                pin_memory = False

            dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=None,
                                                     num_workers=self.get_hparam('cpu_workers'),
                                                     timeout=self.get_hparam('data_fetch_timeout'),
                                                     pin_memory=pin_memory)
        return dataloader

    def schedulers_step(self, objective=None, step_type=None):

        # when using deepspeed, the scheduler is updated implicitly by the optimizer
        if self.deepspeed:
            return

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

    def data_generator(self, subset, max_iterations=None):

        dataloader = enumerate(self.build_dataloader(subset))
        for i, (ind, label, sample) in dataloader:
            if max_iterations is not None and i >= max_iterations:
                break
            sample, label = self.to_device(sample, label)
            yield i, DataBatch(index=ind, label=label, data=sample)

    def to_device(self, *args):
        return to_device(args, self.device, half=self.half, brain=self.brain)

        # if self.accelerator is None or not self.accelerator.device_placement:
        #     return to_device(args, self.device, half=self.half, brain=self.brain)
        # else:
        #     device = 'cuda' if self.is_cuda else 'cpu'
        #     return to_device(args, device, half=self.half, brain=self.brain)

    def finite_data_generator(self, subset, length):

        dataloaders = {k: self.persistent_dataloaders[k][subset]['iterator']
                      for k in self.persistent_dataloaders.keys()}

        for i, samples in enumerate(dictionary_iterator(dataloaders)):

            for k, (ind, label, sample) in samples.items():

                sample, label = self.to_device(sample, label)
                samples[k] = DataBatch(index=ind, label=label, data=sample)

            if len(samples) == 1:
                samples = list(samples.values())[0]
            yield i, samples
            if i + 1 == length:
                break

    def preprocess_epoch(self, epoch=None, subset=None, training=True, **kwargs):
        '''
        :param aux: auxiliary data dictionary - possibly from previous epochs
        :param epoch: epoch number
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        pass

    def train_iteration(self, sample=None, label=None, index=None, counter=None, subset=None, training=True, **kwargs):
        '''
        :param sample: the data fetched by the dataloader
        :param aux: a dictionary of auxiliary data
        :param subset: name of dataset subset (usually train/validation/test)
        :param training: train/test flag
        :return:
        loss: the loss fo this iteration
        aux: an auxiliary dictionary with all the calculated data needed for downstream computation (e.g. to calculate accuracy)
        '''
        pass

    def postprocess_epoch(self, sample=None, label=None, index=None, epoch=None, subset=None, training=True, **kwargs):
        '''
        :param epoch: epoch number
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        pass

    @cached_property
    def optimized_inner_train(self):
        def inner_train_with_cloned_ouptut(*args, **kwargs):
            res = self.inner_train(*args, **kwargs)
            return recursive_clone(res)

        if self.get_hparam('compile_train', False):
            return torch.compile(inner_train_with_cloned_ouptut, mode="reduce-overhead")

        return self.inner_train

    def inner_train(self, sample=None, label=None, index=None, counter=None, subset=None, training=True, **kwargs):
        raise NotImplementedError

    def iterate_epoch(self, subset, training, n):

        # if not training and self.rank > 0:
        #     return

        objective_name = self.objective_name
        batch_size = self.batch_size_train if training else self.batch_size_eval

        with self.reporter.track_epoch(subset, batch_size=batch_size, training=training):
            if training and (n == self.n_epochs + self.swa_epochs):
                logger.warning("This is an extra epoch to calculate BN statistics. "
                               "It is not used for training so we set training=False.")
                training = False
                bu_networks = self.networks
                self.networks = self.swa_networks
                self.set_mode(training=True)

            else:
                self.set_mode(training=training)

            self.preprocess_epoch(epoch=n, training=training)
            desc = f"{subset} (epoch {n+1}/{self.n_epochs + self.swa_epochs})"

            data_generator = self.finite_data_generator(subset, self.epoch_length[subset])
            for i, samples in self.reporter.iterate(data_generator,
                                  enable=self.enable_tqdm, notebook=(not self.ddp and self.is_notebook),
                                  threshold=self.get_hparam('tqdm_threshold'), stats_period=self.get_hparam('tqdm_stats'),
                                  desc=desc, total=self.epoch_length[subset]):

                if type(samples) is DataBatch:
                    ind = samples.index
                    label = samples.label
                    sample = samples.data
                else:
                    sample = samples
                    ind = None
                    label = None

                with torch.autocast(self.autocast_device, dtype=self.mixed_precision_dtype,
                                    enabled=self.amp):
                    self.train_iteration(sample=sample, counter=i, training=training, index=ind, label=label)

                    objective = self.reporter.get_scalar(objective_name, subset=subset, aggregate=False, index=-1)

                    if training and n < self.n_epochs:
                        self.schedulers_step(objective, step_type='iteration')

                    if self.amp and training:
                        if self.scaler._scale is not None:
                            self.scaler.update()
                        for k, scaler in self.scalers.items():
                            if scaler._scale is not None:
                                scaler.update()

            self.postprocess_epoch(sample=sample, index=ind, label=label, epoch=n, training=training)

            if n == self.n_epochs + self.swa_epochs:
                self.set_mode(training=False)
                self.networks = bu_networks

    def preprocess_inference(self, subset=None, predicting=False, **argv):
        '''
        :param aux: auxiliary data dictionary - possibly from previous epochs
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        pass

    def inference_iteration(self, sample=None, label=None, index=None, subset=None, predicting=False, **kwargs):
        '''
        :param sample: the data fetched by the dataloader
        :param aux: a dictionary of auxiliary data
        :param subset: name of dataset subset (usually train/validation/test)
        :return:
        loss: the loss fo this iteration
        aux: an auxiliary dictionary with all the calculated data needed for downstream computation (e.g. to calculate accuracy)
        '''
        self.train_iteration(sample=sample, label=label, index=index, subset=subset, counter=0, training=False, **kwargs)
        return {}

    def postprocess_inference(self, sample=None, label=None, index=None, subset=None, predicting=False, **kwargs):
        '''
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        pass

    def __call__(self, subset, dataset_name='dataset', predicting=False, enable_tqdm=None, max_iterations=None,
                 head=None, eval_mode=True, return_dataset=None, collate=True,  **kwargs):

        self.set_reporter(BeamReport(objective=self.get_hparam('objective'),
                                     optimization_mode=self.optimization_mode))

        with torch.no_grad():

            self.set_mode(training=not eval_mode)
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

            dataloader = self.build_dataloader(subset, name=dataset_name)
            dataset = dataloader.dataset

            batch_size = self.batch_size_eval
            if head is not None:
                max_iterations = math.ceil(head / batch_size)

            with (self.reporter.track_epoch(desc, batch_size=batch_size, training=not eval_mode)):
                self.preprocess_inference(subset=subset, predicting=predicting, dataset=dataset, **kwargs)
                data_generator = self.data_generator(dataloader, max_iterations=max_iterations)
                total_iterations = len(dataloader) if max_iterations is None else min(len(dataloader), max_iterations)
                for i, (ind, label, sample) in self.reporter.iterate(data_generator, enable=enable_tqdm,
                                      threshold=self.get_hparam('tqdm_threshold'), stats_period=self.get_hparam('tqdm_stats'),
                                      notebook=(not self.ddp and self.is_notebook), desc=desc, total=total_iterations):
                    transform = self.inference_iteration(sample=sample, subset=subset, predicting=predicting,
                                                         label=label, index=ind, **kwargs)
                    transforms.append(transform)
                    index.append(ind)

                if collate:
                    index = torch.cat(index)
                    transforms = recursive_concatenate(transforms)

                self.postprocess_inference(sample=sample, index=ind, transforms=transforms, label=label,
                                                     subset=subset, dataset=dataset, predicting=predicting, **kwargs)

            if return_dataset:
                dataset = UniversalDataset(transforms, index=index)
                dataset.set_statistics(self.reporter.data)
            else:
                dataset = DataBatch(index=index, data=transforms, label=None)

        return dataset

    def __iter__(self):

        self.t0 = timer()

        n_epochs = self.n_epochs + self.swa_epochs + int(self.swa_epochs > 0)
        self.refresh_optimizers_and_schedulers_pointers()

        if self.accelerate:
            self.prepare_accelerate()
        elif self.deepspeed:
            self.prepare_deepspeed()

        epoch_start = 0 if self.get_hparam("restart_epochs_count", default=True) else self.epoch
        self.set_train_reporter(first_epoch=epoch_start, n_epochs=n_epochs)

        for i in range(epoch_start, n_epochs):

            self.reporter.reset_epoch(i, total_epochs=self.epoch + 1)

            self.iterate_epoch(subset='train', training=True, n=i)
            with torch.no_grad():
                self.iterate_epoch(subset=self.eval_subset, training=False, n=i)

            # add learning rate and momentum of schedulers_steps
            for k, scheduler in self.schedulers_flat.items():

                if self.accelerate:
                    lr = scheduler.optimizers[0].param_groups[0]['lr']
                else:
                    lr = scheduler.optimizer.param_groups[0]['lr']

                self.report_scalar(f'lr_{k}', lr, subset='train')
                if type(scheduler) is BeamScheduler and scheduler.method in ['one_cycle']:
                    self.report_scalar(f'momentum_{k}', scheduler.get_current_state()['momentum'], subset='train')

            self.epoch += 1
            objective = self.calculate_objective_and_report(i)

            if i+1 == self.n_epochs and self.swa_epochs > 0:
                logger.warning("Switching to SWA training")

            if i+1 >= self.n_epochs and self.swa_epochs > 0:
                for k, swa_model in self.swa_networks.items():
                    swa_model.update_parameters(self.networks[k])
                for k, sch in self.swa_schedulers.items():
                    sch.step()

                    lr = sch.optimizer.param_groups[0]['lr']
                    self.report_scalar(f'swalr_{k}', lr, subset='train')
            else:
                self.schedulers_step(objective=objective, step_type='epoch')

            yield self.reporter

            if self.early_stopping(i):
                return

    def set_mode(self, training=True):

        for net in self.networks.values():

            if training:
                net.train()
            else:
                net.eval()

        for dataset in self.datasets.values():
            if hasattr(dataset, 'train'):
                if training:
                    dataset.train()
                else:
                    dataset.eval()

        self.training = training

    def state_dict(self):
        return self.save_checkpoint(networks=True, optimizers=False, schedulers=False,
                             processors=True, scaler=False, scalers=False, swa_schedulers=False, swa_networks=False,
                             aux=None, pickle_model=False, hparams=True)

    def load_state(self, path):
        self.load_checkpoint(path, networks=True, optimizers=False, schedulers=False, processors=True,
                        scaler=False, scalers=False, swa_schedulers=False, swa_networks=False, hparams=False)

    def save_checkpoint(self, path=None, networks=True, optimizers=True, schedulers=True,
                        processors=True, scaler=True, scalers=True, swa_schedulers=True, swa_networks=True,
                        hparams=True, aux=None, pickle_model=False):

        path = beam_path(path)
        state = {'aux': aux, 'epoch': self.epoch, 'best_objective': self.best_objective, 'best_epoch': self.best_epoch,}
        wrapper = copy.deepcopy if path is None else (lambda x: x)

        if hparams:
            state['hparams'] = self.hparams.namespace

        if not self.deepspeed:

            for k, net in filter_dict(self.networks, networks).items():
                state[f"{k}_parameters"] = wrapper(net.state_dict())
                if pickle_model:
                    state[f"{k}_model"] = net

            for k, optimizer in filter_dict(self.optimizers, optimizers).items():
                state[f"{k}_optimizer"] = wrapper(optimizer.state_dict())

            for k, scheduler in filter_dict(self.schedulers, schedulers).items():
                state[f"{k}_scheduler"] = wrapper(scheduler.state_dict())

            for k, processor in filter_dict(self.processors, processors).items():
                state[f"{k}_processor"] = wrapper(processor.state_dict())

            for k, swa_scheduler in filter_dict(self.swa_schedulers, swa_schedulers).items():
                state[f"{k}_swa_scheduler"] = wrapper(swa_scheduler.state_dict())

            for k, swa_network in filter_dict(self.swa_networks, swa_networks).items():
                state[f"{k}_swa_network"] = wrapper(swa_network.state_dict())

            if scaler:
                state['scaler'] = self.scaler.state_dict() if self.scaler is not None else None

            if scalers:
                state['scalers'] = {k: scaler.state_dict()
                                    if scaler is not None else None for k, scaler in self.scalers.items()}

            if path is not None:
                path.write(state, ext=self.file_format)
            else:
                return state

        else:

            def save_models(path):

                path.mkdir()
                for k, net in filter_dict(self.networks, networks).items():
                    net.save_checkpoint(str(path), client_state=state)

            if path.is_local:
                save_models(path)

            with local_copy(path) as path:
                save_models(path)

    @property
    def file_format(self):
        if self.get_hparam('safetensors', default=False):
            return '.safetensors'
        return '.pt'

    def load_checkpoint(self, path_or_state, strict=True, networks=True, optimizers=True, schedulers=True, hparams=True,
                        processors=True, scaler=True, scalers=True, swa_schedulers=True, swa_networks=True, load_epoch=True):

        path_or_state = beam_path(path_or_state)

        if self.deepspeed:

            def load_models(path):

                client_state = {}
                for k, net in filter_dict(self.networks, networks).items():
                    _, client_state = net.load_checkpoint(str(path), load_module_strict=strict,
                                                    load_optimizer_states=optimizers,
                                                    load_lr_scheduler_states=schedulers,
                                                    load_module_only=False)

                return client_state

            if path_or_state.is_local:
                state = load_models(path_or_state)
            else:
                with local_copy(path_or_state) as path:
                    state = load_models(path)

        else:

            if hasattr(path_or_state, 'read') and hasattr(path_or_state, 'as_uri') :
                logger.info(f"Loading network state from: {path_or_state}")
                state = path_or_state.read(ext=self.file_format, map_location=self.device)
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

        if hparams and 'hparams' in state.keys():
            self.hparams = state['hparams']

        return state.pop('aux', None)

    def set_auxiliary_state(self, aux):
        pass

    def optimize(self, method='compile', networks=True, **kwargs):

        for k, net in filter_dict(self.networks, networks).items():

            logger.warning(f"Optimizing model: {k} with torch compile")
            if self.accelerator is None:
                self.networks[k] = net.optimize(method, **kwargs)
            else:
                assert method=='compile', "Accelerate does not support other optimization methods than compile"
                self.networks[k] = self.accelerator.prepare_model(net, evaluate=True)

    def _fit(self, dataset=None, dataloaders=None, timeout=0, collate_fn=None,
                   worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2, **kwargs):
        '''
        For training purposes
        '''

        def algorithm_generator_single(experiment, *args, **kwargs):

            if dataset is not None:
                self.load_dataset(dataset=dataset, dataloaders=dataloaders, timeout=0, collate_fn=None,
                                  worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2)

            return self

        if self.get_hparam('n_gpus') == 1:
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

    def score(self, *args, **kwargs):
        '''
        For validation and test purposes (when labels are known)
        '''
        return self.evaluate(*args, predicting=False, **kwargs)

    def predict(self, dataset, *args, lazy=False, kpi=False, **kwargs):
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

    @beam_cache
    def cached_predict(self, *args, **kwargs):
        return self.predict(*args, **kwargs)