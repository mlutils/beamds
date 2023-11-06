import math
from collections import defaultdict
from torch import nn
import torch
import copy
from beam.utils import tqdm_beam as tqdm
from beam.logger import beam_logger as logger
import numpy as np
from beam.optim import BeamOptimizer, BeamScheduler, MultipleScheduler
from beam.utils import to_device, check_type, rate_string_format, recursive_concatenate, \
    as_numpy, beam_device, retrieve_name, filter_dict, BeamDict, \
    recursive_collate_chunks, is_notebook, DataBatch, pretty_format_number, nested_defaultdict, dictionary_iterator
from beam.config import beam_arguments, get_beam_parser
from beam.dataset import UniversalBatchSampler, UniversalDataset, TransformedDataset
from beam.experiment import Experiment
from beam.reporter import BeamReport
from timeit import default_timer as timer
from beam.path import beam_path, PureBeamPath
from beam.processor import Processor
from beam.logger import beam_kpi, BeamResult
from beam.data import BeamData


class Algorithm(Processor):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, processors=None, dataset=None,
                 name=None, **kwargs):

        super().__init__(hparams, name=name, **kwargs)

        self._experiment = None
        self._default_hparams = None
        self.hparams_warnings = []
        self.trial = None

        self.hparams = hparams

        # the following are set by the experiment
        self.device = None
        self.ddp = None
        self.hpo = None
        self.rank = None
        self.world_size = None

        # some experiment hyperparameters
        self.half = None
        self.enable_tqdm = None
        self.n_epochs = None
        self.swa_epochs = 0

        self.batch_size_train = None
        self.batch_size_eval = None

        self.cuda = None
        self.pin_memory = None
        self.autocast_device = None
        self.amp_dtype = None
        self.amp = None
        self.scaler = None

        self.set_experiment_properties()

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
        self.datasets = {}
        self.persistent_dataloaders = defaultdict(dict)
        self.dataloaders = defaultdict(dict)

        self.epoch_length = None
        self.eval_subset = None
        self.objective = None
        self.best_objective = None
        self.best_epoch = None
        self.best_state = False

        self.add_components(networks=networks, optimizers=optimizers, schedulers=schedulers, processors=processors)

        if self.get_hparam('reload_path') is not None:
            self.load_checkpoint(hparams.reload_path, hparams=False)

        if hparams.store_initial_weights:
            self.initial_weights = self.save_checkpoint()

        if dataset is not None:
            self.load_dataset(dataset)

        self.cb_model = None
        self._is_notebook = None
        self._predict_reporter = None
        self._train_reporter = None
        self.reporter = None
        self.training = False

    @property
    def state_attributes(self):
        return ['networks', 'optimizers', 'schedulers', 'processors', 'datasets', 'scaler', 'inference_networks',
                'swa_networks', 'swa_schedulers', 'schedulers_initial_state', 'optimizers_name_by_id',
                'schedulers_name_by_id', 'schedulers_flat', 'optimizers_flat', 'optimizers_steps']

    @property
    def train_reporter(self):
        if self._train_reporter is None:
            self._train_reporter = BeamReport(objective=self.get_hparam('objective'))
        return self._train_reporter

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            return self.networks[item]

    @property
    def dataset(self):
        if not len(self.datasets):
            return None
        if len(self.datasets) == 1:
            return list(self.datasets.values())[0]
        return self.datasets

    def set_reporter(self, reporter=None):
        self.reporter = reporter
        self.reporter.reset_time(None)
        self.reporter.reset_epoch(0, total_epochs=None)

    def set_train_reporter(self, first_epoch, n_epochs=None):

        if n_epochs is None:
            n_epochs = self.n_epochs

        self.reporter = self.train_reporter
        self.reporter.reset_time(first_epoch=first_epoch, n_epochs=n_epochs)

    def set_experiment_properties(self):

        self.device = beam_device(self.get_hparam('device'))

        # the following are set by the experiment
        self.ddp = self.get_hparam('ddp')
        self.hpo = self.get_hparam('hpo')
        self.rank = self.get_hparam('rank')
        self.world_size = self.get_hparam('world_size')

        # some experiment hyperparameters
        self.half = self.get_hparam('half')
        self.enable_tqdm = self.get_hparam('enable_tqdm') if (self.get_hparam('tqdm_threshold') == 0
                                                              or not self.get_hparam('enable_tqdm')) else None
        self.n_epochs = self.get_hparam('n_epochs')

        self.batch_size_train = self.get_hparam('batch_size_train')
        self.batch_size_eval = self.get_hparam('batch_size_eval')

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
        self.amp = self.get_hparam('amp') if self.cuda else False
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None

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
                logger.warning("Please pass to the algorithm the hparams object from the experiment: Algorithm(hparams=experiment.hparams).")
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

    @classmethod
    def from_path(cls, path):
        path = beam_path(path)
        state = path.read()
        alg = cls(state['hparams'])
        alg.load_state(state)
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

    def report_scalar(self, name, val, subset=None, aggregation=None, append=None, **kwargs):
        self.reporter.report_scalar(name, val, subset=subset, aggregation=aggregation, append=append, **kwargs)

    def report_data(self, name, val, subset=None, data_type=None, **kwargs):

        if '/' in name:
            dt, name = name.split('/')

            if data_type is None:
                data_type = dt
            else:
                data_type = f"{dt}_{data_type}"

        self.reporter.report_data(name, val, subset=subset, data_type=data_type, **kwargs)

    def report_image(self, name, val, subset=None, **kwargs):
        self.reporter.report_image(name, val, subset=subset, **kwargs)

    def report_images(self, name, val, subset=None, **kwargs):
        self.reporter.report_images(name, val, subset=subset, **kwargs)

    def report_scalars(self, name, val, subset=None, **kwargs):
        self.reporter.report_scalars(name, val, subset=subset, **kwargs)

    def report_histogram(self, name, val, subset=None, **kwargs):
        self.reporter.report_histogram(name, val, subset=subset, **kwargs)

    def report_figure(self, name, val, subset=None, **kwargs):
        self.reporter.report_figure(name, val, subset=subset, **kwargs)

    def report_video(self, name, val, subset=None, **kwargs):
        self.reporter.report_video(name, val, subset=subset, **kwargs)

    def report_audio(self, name, val, subset=None, **kwargs):
        self.reporter.report_audio(name, val, subset=subset, **kwargs)

    def report_embedding(self, name, val, subset=None, **kwargs):
        self.reporter.report_embedding(name, val, subset=subset, **kwargs)

    def report_text(self, name, val, subset=None, **kwargs):
        self.reporter.report_text(name, val, subset=subset, **kwargs)

    def report_mesh(self, name, val, subset=None, **kwargs):
        self.reporter.report_mesh(name, val, subset=subset, **kwargs)

    def report_pr_curve(self, name, val, subset=None, **kwargs):
        self.reporter.report_pr_curve(name, val, subset=subset, **kwargs)

    def get_scalar(self, name, subset=None, aggregate=False):
        v = self.reporter.get_scalar(name, subset=subset, aggregate=aggregate)
        return self.reporter.stack_scalar(v)

    def get_scalars(self, name, subset=None, aggregate=False):
        d = self.reporter.get_scalars(name, subset=subset, aggregate=aggregate)
        for k, v in d.items():
            d[k] = self.reporter.stack_scalar(v)
        return d

    def get_data(self, name, subset=None, data_type=None):

        if '/' in name:
            dt, name = name.split('/')

            if data_type is None:
                data_type = dt
            else:
                data_type = f"{dt}_{data_type}"

        return self.reporter.get_data(name, subset=subset, data_type=data_type)

    def get_image(self, name, subset=None):
        return self.reporter.get_image(name, subset=subset)

    def get_images(self, name, subset=None):
        return self.reporter.get_images(name, subset=subset)

    def get_histogram(self, name, subset=None):
        return self.reporter.get_histogram(name, subset=subset)

    def get_figure(self, name, subset=None):
        return self.reporter.get_figure(name, subset=subset)

    def get_video(self, name, subset=None):
        return self.reporter.get_video(name, subset=subset)

    def get_audio(self, name, subset=None):
        return self.reporter.get_audio(name, subset=subset)

    def get_embedding(self, name, subset=None):
        return self.reporter.get_embedding(name, subset=subset)

    def get_text(self, name, subset=None):
        return self.reporter.get_text(name, subset=subset)

    def get_mesh(self, name, subset=None):
        return self.reporter.get_mesh(name, subset=subset)

    def get_pr_curve(self, name, subset=None):
        return self.reporter.get_pr_curve(name, subset=subset)

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
        self.hparams = experiment.hparams
        self.set_experiment_properties()

        self.ddp = experiment.hparams.ddp
        self.hpo = experiment.hparams.hpo
        self.rank = experiment.hparams.rank
        self.world_size = experiment.hparams.world_size

        self._experiment = experiment

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

    def load_dataset(self, dataset, *args, name=None, **kwargs):
        if name is None:
            name = 'dataset'
        self.load_datasets({name: dataset}, *args, **kwargs)

    def load_datasets(self, datasets, set_epoch_length='first', batch_size_train=None, batch_size_eval=None,
                     oversample=None, weight_factor=None, expansion_size=None,timeout=0, collate_fn=None,
                     worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2,
                     dynamic=False, buffer_size=None, probs_normalization='sum', sample_size=100000):

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
                                                    sample_size=sample_size)
                    d = dataset
                else:
                    sampler = dataset[s].build_sampler(batch_size_train, subset=None, persistent=True,
                                                       oversample=oversample, weight_factor=weight_factor,
                                                       expansion_size=expansion_size,
                                                       dynamic=dynamic, buffer_size=buffer_size,
                                                       probs_normalization=probs_normalization,
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

            if (set_epoch_length == 'first' and i == 0) or k==set_epoch_length:
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

    def data_generator(self, subset, max_iterations=None):

        dataloader = enumerate(self.build_dataloader(subset))
        for i, (ind, label, sample) in dataloader:
            if max_iterations is not None and i >= max_iterations:
                break
            sample = to_device(sample, self.device, half=self.half)
            label = to_device(label, self.device, half=False)
            yield i, DataBatch(index=ind, label=label, data=sample)

    def finite_data_generator(self, subset, length):

        dataloaders = {k: self.persistent_dataloaders[k][subset]['iterator']
                      for k in self.persistent_dataloaders.keys()}

        for i, samples in enumerate(dictionary_iterator(dataloaders)):

            for k, (ind, label, sample) in samples.items():

                sample = to_device(sample, self.device, half=self.half)
                label = to_device(label, self.device, half=False)
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

    def iteration(self, sample=None, label=None, index=None, counter=None, subset=None, training=True, **kwargs):
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

    def iterate_epoch(self, subset, training, n):

        if not training and self.rank > 0:
            return

        objective_name = self.get_hparam('objective')
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

            data_generator = self.finite_data_generator(subset, self.epoch_length[subset])
            for i, samples in self.reporter.iterate(data_generator,
                                  enable=self.enable_tqdm, notebook=(not self.ddp and self.is_notebook),
                                  threshold=self.get_hparam('tqdm_threshold'), stats_period=self.get_hparam('tqdm_stats'),
                                  desc=subset, total=self.epoch_length[subset]):

                if type(samples) is DataBatch:
                    ind = samples.index
                    label = samples.label
                    sample = samples.data
                else:
                    sample = samples
                    ind = None
                    label = None

                with torch.autocast(self.autocast_device, dtype=self.amp_dtype, enabled=self.amp):
                    self.iteration(sample=sample, counter=i, training=training, index=ind, label=label)

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

    def inference(self, sample=None, label=None, index=None, subset=None, predicting=False, **kwargs):
        '''
        :param sample: the data fetched by the dataloader
        :param aux: a dictionary of auxiliary data
        :param subset: name of dataset subset (usually train/validation/test)
        :return:
        loss: the loss fo this iteration
        aux: an auxiliary dictionary with all the calculated data needed for downstream computation (e.g. to calculate accuracy)
        '''
        self.iteration(sample=sample, label=label, index=index, subset=subset, counter=0, training=False, **kwargs)
        return {}

    def postprocess_inference(self, sample=None, label=None, index=None, subset=None, predicting=False, **kwargs):
        '''
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        pass

    def calculate_objective(self):
        '''
        This function calculates the optimization non-differentiable objective. It is used for hyperparameter optimization
        and for ReduceOnPlateau scheduling. It is also responsible for tracking the best checkpoint
        '''

        self.best_objective = self.reporter.best_objective
        self.best_epoch = self.reporter.best_epoch
        self.objective = self.reporter.objective
        self.best_state = self.reporter.best_state

        return self.objective

    def report(self, objective, epoch=None):
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

    def early_stopping(self, epoch=None):
        '''
        Use this function to early stop your model based on the results or any other metric in the algorithm class
        '''

        stop_at = self.get_hparam('stop_at')
        early_stopping_patience = self.get_hparam('early_stopping_patience')
        if stop_at is not None or early_stopping_patience is not None:
            if self.objective is None:
                logger.warning("Early stopping is enabled but no objective is defined. set objective in the hparams")
                return False

        if stop_at is not None and stop_at > 0:
            if self.best_objective is not None:
                res = self.best_objective > stop_at
                if res:
                    logger.info(f"Stopping training at {self.best_objective} > {stop_at}")
                return res

        if early_stopping_patience is not None and early_stopping_patience > 0:
            res = self.epoch - self.best_epoch > early_stopping_patience
            if res:
                logger.info(f"Stopping training at epoch {self.epoch} - best epoch {self.best_epoch} > {early_stopping_patience}")
            return res

        return False

    def __call__(self, subset, dataset_name='dataset', predicting=False, enable_tqdm=None, max_iterations=None,
                 head=None, eval_mode=True, return_dataset=None, **kwargs):

        self.set_reporter(BeamReport(objective=self.get_hparam('objective')))

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
                    transform = self.inference(sample=sample, subset=subset, predicting=predicting,
                                                        label=label, index=ind, **kwargs)
                    transforms.append(transform)
                    index.append(ind)

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

        n_epochs = self.n_epochs + self.swa_epochs + int(self.swa_epochs > 0)
        self.refresh_optimizers_and_schedulers_pointers()

        epoch_start = 0 if self.get_hparam("restart_epochs_count", default=True) else self.epoch
        self.set_train_reporter(first_epoch=epoch_start, n_epochs=n_epochs)

        for i in range(epoch_start, n_epochs):

            self.reporter.reset_epoch(i, total_epochs=self.epoch + 1)

            self.iterate_epoch(subset='train', training=True, n=i)
            with torch.no_grad():
                self.iterate_epoch(subset=self.eval_subset, training=False, n=i)

            # add learning rate and momentum of schedulers_steps
            for k, scheduler in self.schedulers_flat.items():
                lr = scheduler.optimizer.param_groups[0]['lr']

                self.report_scalar(f'lr_{k}', lr, subset='train')
                if type(scheduler) is BeamScheduler and scheduler.method in ['one_cycle']:
                    self.report_scalar(f'momentum_{k}', scheduler.get_current_state()['momentum'], subset='train')

            self.epoch += 1
            
            objective = self.calculate_objective()
            self.report(objective, i)

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

    def save_state(self, path=None):
        self.save_checkpoint(path=path, networks=True, optimizers=False, schedulers=False,
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
            state['hparams'] = self.hparams

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
            path.write(state, ext='.pt')
        else:
            return state

    def load_checkpoint(self, path_or_state, strict=True, networks=True, optimizers=True, schedulers=True, hparams=True,
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

        if hparams and 'hparams' in state.keys():
            self.hparams = state['hparams']

        return state['aux']

    def set_auxiliary_state(self, aux):
        pass

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

    def score(self, *args, **kwargs):
        '''
        For validation and test purposes (when labels are known)
        '''
        return self.evaluate(*args, predicting=False, **kwargs)

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
