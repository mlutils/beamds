import math
from collections import defaultdict
from torch import nn
import torch
import copy
from .utils import tqdm_beam as tqdm
from .utils import logger
import numpy as np
from .model import BeamOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from .utils import finite_iterations, to_device, check_type, rate_string_format, concat_data, \
    stack_inference_results, to_numpy, stack_train_results
from .config import beam_arguments, get_beam_parser
from .dataset import UniversalBatchSampler, UniversalDataset, TransformedDataset, DataBatch
from .experiment import Experiment
from timeit import default_timer as timer
import torch_tensorrt as trt


class Algorithm(object):

    def __init__(self, hparams, networks=None, optimizers=None):

        self._experiment = None
        self.trial = None

        self.hparams = hparams

        self.device = hparams.device
        self.ddp = hparams.ddp
        self.hpo = hparams.hpo

        self.rank = hparams.rank
        self.world_size = hparams.world_size

        # some experiment hyperparameters
        self.half = hparams.half
        self.enable_tqdm = hparams.enable_tqdm if hparams.tqdm_threshold == 0 or not hparams.enable_tqdm else None
        self.n_epochs = hparams.n_epochs
        self.batch_size_train = hparams.batch_size_train
        self.batch_size_eval = hparams.batch_size_eval

        self.cuda = (self.device.type == 'cuda')
        self.pin_memory = self.cuda
        self.autocast_device = 'cuda' if self.cuda else 'cpu'
        self.amp = hparams.amp if self.cuda else False
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.scalers = {}
        self.epoch = 0

        self.networks = {}
        self.inference_networks = {}
        self.optimizers = {}
        self.add_networks_and_optmizers(networks=networks, optimizers=optimizers)

        if hparams.store_initial_weights:
            self.initial_weights = self.save_checkpoint()

        if hparams.reload_path is not None:
            self.load_checkpoint(hparams.reload_path)

        self.dataset = None
        self.persistent_dataloaders = {}
        self.dataloaders = {}
        self.eval_subset = None

    @staticmethod
    def get_parser():
        return get_beam_parser()

    @classmethod
    def from_pretrained(cls, path=None, hparams=None, **kwargs):
        if path is not None:
            experiment = Experiment.reload_from_path(path)
        elif hparams is not None:
            experiment = Experiment(hparams)
        else:
            hparams = beam_arguments(cls.get_parser(), **kwargs)
            experiment = Experiment(hparams)
        return experiment.algorithm_generator(cls)

    def add_networks_and_optmizers(self, networks=None, optimizers=None, build_optimizers=True, name='net'):

        if networks is None:
            networks = {}
        elif issubclass(type(networks), nn.Module):
            networks = {name: networks}
        elif check_type(networks).major == 'dict':
            pass
        else:
            raise NotImplementedError("Network type is unsupported")

        for k in networks.keys():
            networks[k] = self.register_network(networks[k])

        if build_optimizers:
            if optimizers is None:
                optimizers = {k: BeamOptimizer(v, dense_args={'lr': self.hparams.lr_dense,
                                                              'weight_decay': self.hparams.weight_decay,
                                                               'betas': (self.hparams.beta1, self.hparams.beta2),
                                                              'eps': self.hparams.eps,
                                                               'capturable': self.hparams.capturable},
                                               sparse_args={'lr': self.hparams.lr_sparse,
                                                            'betas': (self.hparams.beta1, self.hparams.beta2),
                                                            'eps': self.hparams.eps},
                                               clip=self.hparams.clip_gradient, amp=self.amp, accumulate=self.hparams.accumulate
                                               ) for k, v in networks.items()}

            elif issubclass(type(optimizers), dict):
                for k, o in optimizers.items():
                    if callable(o):
                        try:
                            optimizers[k] = o(networks[k])
                        except TypeError:
                            optimizers[k] = o(networks[k].parameters())
                    else:
                        o.load_state_dict(o.state_dict())
                        optimizers[k] = o

            elif issubclass(type(optimizers), torch.optim.Optimizer) or issubclass(type(optimizers), BeamOptimizer):
                optimizers.load_state_dict(optimizers.state_dict())
                optimizers = {name: optimizers}

            elif callable(optimizers):
                try:
                    optimizers = {name: optimizers(networks[name])}
                except TypeError:
                    optimizers = {name: optimizers(networks[name].parameters())}
            else:
                raise NotImplementedError

        else:
            optimizers = {}

        for k, net in networks.items():
            if k in self.networks:
                self.networks.pop(k)
                self.inference_networks.pop(k)
                logger.warning(f"Found network with identical keys: {k}. Overriding previous network.")
                if k in self.optimizers:
                    self.optimizers.pop(k)

            self.networks[k] = net
            self.inference_networks[k] = net
        for k, opt in optimizers.items():
            self.optimizers[k] = opt

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
        if reduction is None:
            reduction = self.hparams.reduction

        if len(losses) == 1 and issubclass(type(losses[0]), dict):
            losses = losses[0]
        elif len(losses) == 1:
            losses = {name: losses[0]}
        else:
            losses = {f'{name}_{i}': l for i, l in enumerate(losses)}

        if weights is None:
            weights = {k: 1 for k in losses.keys()}
        elif issubclass(type(weights), dict):
            pass
        else:
            weights_type = check_type(weights, check_minor=False, check_element=False)
            if weights_type.major == 'scalar':
                weights = {next(iter(losses.keys())): weights}
            else:
                weights = {f'{name}_{i}': l for i, l in enumerate(weights)}

        for k, loss in losses.items():
            n = torch.numel(loss)

            if n > 1:

                if reduction == 'sum':
                    r = 1
                elif reduction == 'mean':
                    r = n
                elif reduction == 'mean_batch':
                    r = len(loss)
                elif reduction == 'sqrt':
                    r = math.sqrt(n)
                elif reduction == 'sqrt_batch':
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
                    results['scalar'][f'{k}_s'].append(to_numpy(l))

                    if weights[k] > 1:
                        results['scalar'][f'{k}_w'].append(to_numpy(weights[k]))
                    elif weights[k] == 0:
                        results['scalar'][f'{k}_w'].append(0)
                    else:
                        results['scalar'][f'{k}_f'].append(to_numpy(1 / weights[k]))

            results['scalar'][name] = to_numpy(total_loss)

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
                optimizers = self.optimizers
            elif issubclass(type(optimizers), torch.optim.Optimizer) or issubclass(type(optimizers), BeamOptimizer):
                optimizers = [optimizers]

            optimizers_flat = []
            for op in optimizers:
                if issubclass(type(op), BeamOptimizer):
                    for opi in op.optimizers.values():
                        optimizers_flat.append(opi)
                else:
                    optimizers_flat.append(op)
            optimizers = optimizers_flat

            with torch.autocast(self.autocast_device, enabled=False):

                if self.amp:
                    scaler.scale(loss).backward(gradient=gradient, retain_graph=retain_graph,
                                                     create_graph=create_graph, inputs=inputs)
                else:
                    loss.backward(gradient=gradient, retain_graph=retain_graph,
                                  create_graph=create_graph, inputs=inputs)

                if self.hparams.clip_gradient > 0:
                    for op in optimizers.values():
                        if self.amp:
                            scaler.unscale_(op)
                        for pg in op.param_groups:
                            torch.nn.utils.clip_grad_norm_(iter(pg['params']), self.hparams.clip)

                if iteration is None or not (iteration % self.hparams.accumulate):
                    for op in optimizers:
                        if self.amp:
                            scaler.step(op)
                        else:
                            op.step()
                        op.zero_grad(set_to_none=set_to_none)
        return loss

    def load_dataset(self, dataset=None, batch_size_train=None, batch_size_eval=None,
                     oversample=None, weight_factor=None, expansion_size=None,timeout=0, collate_fn=None,
                     worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2,
                     dynamic=False, buffer_size=None, probs_normalization='sum', sample_size=100000):

        self.dataset = dataset

        batch_size_train = self.hparams.batch_size_train if batch_size_train is None else batch_size_train
        batch_size_eval = self.hparams.batch_size_eval if batch_size_eval is None else batch_size_eval
        oversample = (self.hparams.oversampling_factor > 0) if oversample is None else oversample
        weight_factor = self.hparams.oversampling_factor if weight_factor is None else weight_factor
        expansion_size = self.hparams.expansion_size if expansion_size is None else expansion_size
        dynamic = self.hparams.dynamic_sampler if dynamic is None else dynamic
        buffer_size = self.hparams.buffer_size if buffer_size is None else buffer_size
        probs_normalization = self.hparams.probs_normalization if probs_normalization is None else probs_normalization
        sample_size = self.hparams.sample_size if sample_size is None else sample_size

        self.persistent_dataloaders = {}
        self.dataloaders = {}

        subsets = dataset.indices.keys()
        self.eval_subset = 'validation' if 'validation' in subsets else 'test'

        for s in subsets:
            sampler = dataset.build_sampler(batch_size_eval, subset=s, persistent=False, oversample=oversample,
                                            weight_factor=weight_factor, expansion_size=expansion_size,
                                            dynamic=dynamic, buffer_size=buffer_size,
                                            probs_normalization=probs_normalization,
                                            sample_size=sample_size)

            self.dataloaders[s] = dataset.build_dataloader(sampler, num_workers=self.hparams.cpu_workers,
                                                            pin_memory=self.pin_memory,
                                                            timeout=timeout, collate_fn=collate_fn,
                                                            worker_init_fn=worker_init_fn,
                                                            multiprocessing_context=multiprocessing_context,
                                                            generator=generator,
                                                            prefetch_factor=prefetch_factor)

        for s in ['train', self.eval_subset]:

            sampler = dataset.build_sampler(batch_size_train, subset=s, persistent=True, oversample=oversample,
                                            weight_factor=weight_factor, expansion_size=expansion_size,
                                            dynamic=dynamic, buffer_size=buffer_size,
                                            probs_normalization=probs_normalization,
                                            sample_size=sample_size)

            self.persistent_dataloaders[s] = dataset.build_dataloader(sampler, num_workers=self.hparams.cpu_workers,
                                                        pin_memory=self.pin_memory,
                                                        timeout=timeout, collate_fn=collate_fn,
                                                        worker_init_fn=worker_init_fn,
                                                        multiprocessing_context=multiprocessing_context,
                                                        generator=generator,
                                                        prefetch_factor=prefetch_factor)

        self.epoch_length = {'train': None, self.eval_subset: None}

        if self.hparams.epoch_length is not None:
            l_train = len(dataset.indices['train'])
            l_eval = len(dataset.indices[self.eval_subset])

            self.epoch_length['train'] = int(self.hparams.epoch_length * l_train / (l_train + l_eval))
            self.epoch_length[self.eval_subset] = int(self.hparams.epoch_length * l_eval / (l_train + l_eval))

        if self.hparams.epoch_length_train is not None:
            self.epoch_length['train'] = self.hparams.epoch_length_train

        if self.hparams.epoch_length_eval is not None:
            self.epoch_length[self.eval_subset] = self.hparams.epoch_length_eval

        if self.epoch_length['train'] is None:
            dataset = self.persistent_dataloaders['train'].dataset
            self.epoch_length['train'] = len(dataset.indices['train'])

        if self.epoch_length[self.eval_subset] is None:
            dataset = self.persistent_dataloaders[self.eval_subset].dataset
            self.epoch_length[self.eval_subset] = len(dataset.indices[self.eval_subset])

        if self.hparams.scale_epoch_by_batch_size:
            self.epoch_length[self.eval_subset] = self.epoch_length[self.eval_subset] // self.batch_size_eval
            self.epoch_length['train'] = self.epoch_length['train'] // self.batch_size_train

        if self.n_epochs is None:
            self.n_epochs = self.hparams.total_steps // self.epoch_length['train']

    def register_network(self, net):

        if self.half:
            net = net.half()

        net = net.to(self.device)

        if self.ddp:
            net_ddp = DDP(net, device_ids=[self.device],
                      find_unused_parameters=self.hparams.find_unused_parameters,
                      broadcast_buffers=self.hparams.broadcast_buffers)

            for a in dir(net):
                if a not in dir(net_ddp) and not a.startswith('_'):
                    setattr(net_ddp, a, getattr(net, a))
            net = net_ddp

        return net

    def get_optimizers(self):
        return self.optimizers

    def get_networks(self):
        return self.networks

    def process_sample(self, sample):
        return to_device(sample, self.device, half=self.half)

    def build_dataloader(self, subset):

        subset_type = check_type(subset)

        if type(subset) is str:
            dataloader = self.dataloaders[subset]
        elif issubclass(type(subset), torch.utils.data.DataLoader):
            dataloader = subset
        elif issubclass(type(subset), torch.utils.data.Dataset):

            dataset = subset

            sampler = dataset.build_sampler(self.hparams.batch_size_eval, persistent=False)

            dataloader = dataset.build_dataloader(sampler, num_workers=self.hparams.cpu_workers,
                                                  pin_memory=self.pin_memory)

        else:

            if subset_type.minor == 'tuple' and check_type(subset.index).major == 'array' \
                    and check_type(subset.data).minor in ['list', 'dict']:
                if check_type(subset.data).minor == 'list':
                    dataset = UniversalDataset(*subset.data, index=subset.index)
                else:
                    dataset = UniversalDataset(**subset.data, index=subset.index)
            elif subset_type.minor in ['list', 'tuple']:
                dataset = UniversalDataset(*subset)
            elif subset_type.minor in ['dict']:
                dataset = UniversalDataset(**subset)
            else:
                dataset = UniversalDataset(subset)

            sampler = UniversalBatchSampler(len(dataset), self.hparams.batch_size_eval, shuffle=False,
                                            tail=True, once=True)
            dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=None,
                                                     num_workers=0, pin_memory=self.pin_memory)

        return dataloader

    def data_generator(self, subset, max_iterations=None, persistent=False):

        if persistent:
            dataloader = self.persistent_dataloaders[subset]
        else:
            dataloader = self.build_dataloader(subset)
        for i, (ind, sample) in enumerate(dataloader):
            if max_iterations is not None and i >= max_iterations:
                break
            sample = self.process_sample(sample)
            yield i, DataBatch(index=ind, data=sample)

    def preprocess_epoch(self, results=None, epoch=None, subset=None, training=True, **kwargs):
        '''
        :param aux: auxiliary data dictionary - possibly from previous epochs
        :param epoch: epoch number
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return results

    def iteration(self, sample=None, results=None, counter=None, subset=None, training=True, **kwargs):
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

    def postprocess_epoch(self, sample=None, results=None, epoch=None, subset=None, training=True, **kwargs):
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
            results = defaultdict(lambda: defaultdict(list))

            if not training and self.rank > 0:
                yield results
                continue

            self.set_mode(training=training)
            results = self.preprocess_epoch(results=results, epoch=n, training=training)

            data_generator = self.data_generator(subset, persistent=True)
            for i, (ind, sample) in tqdm(finite_iterations(data_generator, self.epoch_length[subset]),
                                  enable=self.enable_tqdm, notebook=(not self.ddp),
                                  threshold=self.hparams.tqdm_threshold, stats_period=self.hparams.tqdm_stats,
                                  desc=subset, total=self.epoch_length[subset]):

                with torch.autocast(self.autocast_device, enabled=self.amp):
                    results = self.iteration(sample=sample, results=results, counter=i, training=training, index=ind)

                    if self.amp and training:
                        if self.scaler._scale is not None:
                            self.scaler.update()
                        for k, scaler in self.scalers.items():
                            if scaler._scale is not None:
                                scaler.update()

            results = self.postprocess_epoch(sample=sample, index=ind, results=results, epoch=n, training=training)

            batch_size = self.batch_size_train if training else self.batch_size_eval
            results = stack_inference_results(results, batch_size=batch_size)

            delta = timer() - t0
            n_iter = i + 1

            results['stats']['seconds'] = delta
            results['stats']['batches'] = n_iter
            results['stats']['samples'] = n_iter * batch_size
            results['stats']['batch_rate'] = rate_string_format(n_iter, delta)
            results['stats']['sample_rate'] = rate_string_format(n_iter * batch_size, delta)

            yield results

    def preprocess_inference(self, results=None, subset=None, predicting=False, **argv):
        '''
        :param aux: auxiliary data dictionary - possibly from previous epochs
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return results

    def inference(self, sample=None, results=None, subset=None, predicting=False, **kwargs):
        '''
        :param sample: the data fetched by the dataloader
        :param aux: a dictionary of auxiliary data
        :param results: a dictionary of dictionary of lists containing results of
        :param subset: name of dataset subset (usually train/validation/test)
        :return:
        loss: the loss fo this iteration
        aux: an auxiliary dictionary with all the calculated data needed for downstream computation (e.g. to calculate accuracy)
        '''
        results = self.iteration(sample=sample, results=results, subset=subset, counter=0, training=False, **kwargs)
        return {}, results

    def postprocess_inference(self, sample=None, results=None, subset=None, predicting=False, **kwargs):
        '''
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return results

    def report(self, results=None, epoch=None, **argv):
        '''
        Use this function to report results to hyperparameter optimization frameworks
        also you can add key 'objective' to the results dictionary to report the final scores.
        '''
        return results

    def early_stopping(self, results=None, epoch=None, **kwargs):
        '''
        Use this function to early stop your model based on the results or any other metric in the algorithm class
        '''
        return False

    def __call__(self, subset, predicting=False, enable_tqdm=None, max_iterations=None, head=None, **kwargs):

        with torch.no_grad():

            self.set_mode(training=False)
            results = defaultdict(lambda: defaultdict(list))
            transforms = []
            index = []

            desc = subset if type(subset) is str else ('predict' if predicting else 'evaluate')

            if enable_tqdm is None:
                enable_tqdm = self.enable_tqdm

            dataloader = self.build_dataloader(subset)
            dataset = dataloader.dataset

            batch_size = self.batch_size_eval
            if head is not None:
                max_iterations = math.ceil(head / batch_size)

            results = self.preprocess_inference(results=results, subset=subset, predicting=predicting, dataset=dataset,
                                                **kwargs)
            data_generator = self.data_generator(dataloader, max_iterations=max_iterations)
            total_iterations = len(dataloader) if max_iterations is None else min(len(dataloader), max_iterations)
            for i, (ind, sample) in tqdm(data_generator, enable=enable_tqdm,
                                  threshold=self.hparams.tqdm_threshold, stats_period=self.hparams.tqdm_stats,
                                  notebook=(not self.ddp), desc=desc, total=total_iterations):
                transform, results = self.inference(sample=sample, results=results, subset=subset, predicting=predicting,
                                         index=ind, **kwargs)
                transforms.append(transform)
                index.append(ind)

            index = torch.cat(index)
            transforms = concat_data(transforms)
            results = self.postprocess_inference(sample=sample, index=ind, transforms=transforms,
                                                 results=results, subset=subset, dataset=dataset,
                                                 predicting=predicting, **kwargs)

            results = stack_inference_results(results, batch_size=batch_size)
            dataset = UniversalDataset(transforms, index=index)
            dataset.set_statistics(results)

        return dataset

    def __iter__(self):

        eval_generator = self.epoch_iterator(self.n_epochs, subset=self.eval_subset, training=False)
        for i, train_results in enumerate(self.epoch_iterator(self.n_epochs, subset='train', training=True)):

            train_results = stack_train_results(train_results, batch_size=self.batch_size_train)

            with torch.no_grad():

                eval_results = next(eval_generator)
                eval_results = stack_train_results(eval_results, batch_size=self.batch_size_eval)

            results = {'train': train_results, self.eval_subset: eval_results}

            if self.hpo is not None:
                results = self.report(results, i)

            self.epoch += 1
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

        state = {'aux': aux, 'epoch': self.epoch}

        wrapper = copy.deepcopy if path is None else (lambda x: x)

        for k, net in self.networks.items():
            state[f"{k}_parameters"] = wrapper(net.state_dict())
            if pickle_model:
                state[f"{k}_model"] = net

        for k, optimizer in self.optimizers.items():
            state[f"{k}_optimizer"] = wrapper(optimizer.state_dict())

        state['scaler'] = self.scaler.state_dict() if self.scaler is not None else None

        state['scalers'] = {k: scaler.state_dict() if scaler is not None else None for k, scaler in self.scalers.items()}

        if path is not None:
            torch.save(state, path)
        else:
            return state

    def load_checkpoint(self, path, strict=True):

        if type(path) is str:
            logger.info(f"Loading network state from: {path}")
            state = torch.load(path, map_location=self.device)
        else:
            state = path

        for k, net in self.networks.items():

            s = state[f"{k}_parameters"]

            if not self.ddp:
                torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(s, 'module.')

            net.load_state_dict(s, strict=strict)

        for k, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state[f"{k}_optimizer"])

        if self.scaler is not None and 'scaler' in state.keys():
            self.scaler.load_state_dict(state["scaler"])

        for k, s in state["scalers"].items():
            if k in self.scalers:
                self.scalers[k].load_state_dict(s)

        self.epoch = state['epoch']
        return state['aux']

    def optimize_for_inference(self, networks, half=True, eval=True):

        logger.warning("Currently we support only models on device=0")
        sample = self.dataset[0]

        self.inference_networks = {}

        for k, v in networks.items():

            v_type = check_type(v)
            if v_type.element == 'str':
                shape = sample.data[v].shape
            else:
                shape = v

            opt_shape = list((self.batch_size_eval, *shape))
            min_shape = list((1, *shape))

            net = copy.deepcopy(self.networks[k])
            if eval:
                net = net.eval()
            else:
                net = net.train()

            torch_script_module = torch.jit.optimize_for_inference(torch.jit.script(net.eval()))

            dtype = torch.half if half else torch.float

            # trt_ts_module = trt.compile(torch_script_module, inputs=[trt.Input(opt_shape=opt_shape,
            #                                                                     min_shape=min_shape,
            #                                                                     max_shape=opt_shape,
            #                                                                     dtype=dtype)],
            #                                        enabled_precisions={dtype},
            #                             require_full_compilation=True)

            trt_ts_module = trt.compile(torch_script_module, inputs=[trt.Input(shape=opt_shape,
                                                                                dtype=dtype)],
                                                   enabled_precisions={dtype},
                                        require_full_compilation=False)

            self.inference_networks[k] = trt_ts_module

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

        if self.hparams.parallel == 1:
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

    def predict(self, dataset, *args, lazy=False, **kwargs):
        '''
        For real data purposes (when labels are unknown)
        '''
        if lazy:
            return TransformedDataset(dataset, self, *args, **kwargs)
        return self(dataset, *args, predicting=True, **kwargs)
