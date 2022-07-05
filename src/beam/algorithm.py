from collections import defaultdict
from torch import nn
import torch
import copy
from .utils import tqdm_beam as tqdm
from .utils import logger
import numpy as np
from .model import BeamOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from .utils import finite_iterations, to_device, check_type
from .dataset import UniversalBatchSampler, UniversalDataset


class Algorithm(object):

    def __init__(self, experiment, networks=None, optimizers=None):

        self.experiment = experiment

        self.args = experiment.args

        self.device = experiment.device
        self.ddp = experiment.ddp
        self.hpo = experiment.hpo

        # some experiment hyperparameters
        self.half = experiment.args.half
        self.enable_tqdm = experiment.args.enable_tqdm
        self.trial = experiment.trial
        self.n_epochs = experiment.args.n_epochs
        self.batch_size_train = experiment.args.batch_size_train
        self.batch_size_eval = experiment.args.batch_size_eval

        self.pin_memory = ('cpu' not in str(self.device))
        self.amp = experiment.args.amp if self.pin_memory else False

        if networks is None:
            networks = {}
        elif issubclass(type(networks), nn.Module):
            networks = {'net': networks}
        elif check_type(networks).major == 'dict':
            pass
        else:
            raise NotImplementedError("Network type is unsupported")

        self.networks = networks

        if optimizers is None:
            self.networks = {k: self.register_network(v) for k, v in networks.items()}
            self.optimizers = {k: BeamOptimizer(v, dense_args={'lr': self.args.lr_dense,
                                                          'weight_decay': self.args.weight_decay,
                                                           'betas': (self.args.beta1, self.args.beta2),
                                                          'eps': self.args.eps},
                                           sparse_args={'lr': self.args.lr_sparse,
                                                        'betas': (self.args.beta1, self.args.beta2),
                                                        'eps': self.args.eps},
                                           clip=self.args.clip_gradient, amp=self.amp, accumulate=self.args.accumulate
                                           ) for k, v in self.networks.items()}

        elif issubclass(type(optimizers), dict):
            self.optimizers = {}
            for k, o in optimizers.items():
                if callable(o):
                    self.networks[k] = self.register_network(self.networks[k])
                    self.optimizers[k] = self.networks[k]
                else:
                    self.optimizers[k] = o

        elif issubclass(type(optimizers), torch.optim.Optimizer) or issubclass(type(optimizers), BeamOptimizer):
            self.optimizers = {'net': optimizers}

        elif callable(optimizers):
            self.networks['net'] = self.register_network(self.networks['net'])
            self.optimizers = {'net': optimizers(self.networks['net'])}
        else:
            raise NotImplementedError

        if experiment.args.store_initial_weights:
            self.initial_weights = self.save_checkpoint()

        if experiment.load_model:
            experiment.reload_checkpoint(self)

    def load_dataset(self, dataset=None, dataloaders=None, train_batch_size=None, eval_batch_size=None,
                     oversample=False, weight_factor=None, expansion_size=None,timeout=0, collate_fn=None,
                     worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2):

        assert dataloaders is not None or dataset is not None, "Either dataset or dataloader must be supplied"
        self.dataset = dataset

        if dataloaders is None:

            train_batch_size = self.experiment.args.train_batch_size if train_batch_size is None else train_batch_size
            eval_batch_size = self.experiment.args.eval_batch_size if eval_batch_size is None else eval_batch_size
            oversample = self.experiment.args.oversample if oversample is None else oversample
            weight_factor = self.experiment.args.weight_factor if weight_factor is None else weight_factor
            expansion_size = self.experiment.args.expansion_size if expansion_size is None else expansion_size

            dataset.build_samplers(train_batch_size, eval_batch_size=eval_batch_size,
                                   oversample=oversample, weight_factor=weight_factor, expansion_size=expansion_size)

            pin_memory = 'cpu' not in str(self.experiment.args.device)
            dataloaders = dataset.build_dataloaders(num_workers=self.experiment.args.cpu_workers, pin_memory=pin_memory,
                                                   timeout=timeout, collate_fn=collate_fn,
                                                   worker_init_fn=worker_init_fn,
                                                   multiprocessing_context=multiprocessing_context, generator=generator,
                                                   prefetch_factor=prefetch_factor)

        self.dataloaders = dataloaders
        self.epoch_length = {}

        self.eval_subset = 'validation' if 'validation' in dataloaders.keys() else 'test'
        self.epoch_length['train'] = self.experiment.args.epoch_length_train
        self.epoch_length[self.eval_subset] = self.experiment.args.epoch_length_eval

        if self.epoch_length['train'] is None:
            dataset = dataloaders['train'].dataset
            self.epoch_length['train'] = len(dataset.indices_split['train'])

        if self.epoch_length[self.eval_subset] is None:
            dataset = dataloaders[self.eval_subset].dataset
            self.epoch_length[self.eval_subset] = len(dataset.indices_split[self.eval_subset])

        if self.experiment.args.scale_epoch_by_batch_size:
            self.epoch_length[self.eval_subset] = self.epoch_length[self.eval_subset] // self.batch_size_eval
            self.epoch_length['train'] = self.epoch_length['train'] // self.batch_size_train

        if 'test' in dataloaders.keys():
            self.epoch_length['test'] = len(dataloaders['test'])

        if self.n_epochs is None:
            self.n_epochs = self.experiment.args.total_steps // self.epoch_length['train']

        if self.dataset is None:
            self.dataset = next(iter(self.dataloaders.values())).dataset

    def register_network(self, net):

        net = net.to(self.device)

        if self.half:
            net = net.half()

        if self.ddp:
            net = DDP(net, device_ids=[self.device])

        return net

    def get_optimizers(self):
        return self.optimizers

    def get_networks(self):
        return self.networks

    def process_sample(self, sample):
        return to_device(sample, self.device, half=self.half)

    def build_dataloader(self, subset):

        if type(subset) is str:
            dataloader = self.dataloaders[subset]
        elif issubclass(type(subset), torch.utils.data.DataLoader):
            dataloader = subset
        elif issubclass(type(subset), torch.utils.data.Dataset):

            dataset = subset

            pin_memory = self.pin_memory if 'cpu' in str(dataset.device) else False
            sampler = UniversalBatchSampler(len(dataset), self.experiment.args.batch_size_eval, shuffle=False,
                                            tail=True, once=True)
            dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=None,
                                                     num_workers=0, pin_memory=pin_memory)

        else:

            if check_type(subset).minor in ['list', 'tuple']:
                dataset = UniversalDataset(*subset)
            elif check_type(subset).minor in ['dict']:
                dataset = UniversalDataset(**subset)
            else:
                dataset = UniversalDataset(subset)

            pin_memory = self.pin_memory if 'cpu' in str(dataset.device) else False
            sampler = UniversalBatchSampler(len(dataset), self.experiment.args.batch_size_eval, shuffle=False,
                                            tail=True, once=True)
            dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=None,
                                                     num_workers=0, pin_memory=pin_memory)

        return dataloader

    def data_generator(self, subset):

        dataloader = self.build_dataloader(subset)
        for i, sample in enumerate(dataloader):
            sample = self.process_sample(sample)
            yield i, sample

    def preprocess_epoch(self, results=None, epoch=None, subset=None, training=True):
        '''
        :param aux: auxiliary data dictionary - possibly from previous epochs
        :param epoch: epoch number
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return results

    def iteration(self, sample=None, results=None, subset=None, training=True):
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

    def postprocess_epoch(self, sample=None, results=None, epoch=None, subset=None, training=True):
        '''
        :param epoch: epoch number
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return results

    def inner_loop(self, n_epochs, subset, training=True):

        for n in range(n_epochs):

            results = defaultdict(lambda: defaultdict(list))

            results = self.preprocess_epoch(results=results, epoch=n, subset=subset, training=training)
            self.set_mode(training=training)
            data_generator = self.data_generator(subset)
            for i, sample in tqdm(finite_iterations(data_generator, self.epoch_length[subset]),
                                  enable=self.enable_tqdm, notebook=(not self.ddp),
                                  desc=subset, total=self.epoch_length[subset] - 1):

                results = self.iteration(sample=sample, results=results, subset=subset, training=training)

            results = self.postprocess_epoch(sample=sample, results=results,
                                                  subset=subset, epoch=n, training=training)

            yield results

    def preprocess_inference(self, results=None, subset=None, with_labels=True):
        '''
        :param aux: auxiliary data dictionary - possibly from previous epochs
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return results

    def inference(self, sample=None, results=None, subset=None, with_labels=True):
        '''
        :param sample: the data fetched by the dataloader
        :param aux: a dictionary of auxiliary data
        :param results: a dictionary of dictionary of lists containing results of
        :param subset: name of dataset subset (usually train/validation/test)
        :return:
        loss: the loss fo this iteration
        aux: an auxiliary dictionary with all the calculated data needed for downstream computation (e.g. to calculate accuracy)
        '''
        results = self.iteration(sample=sample, results=results, subset=subset, training=False)
        return results

    def postprocess_inference(self, sample=None, results=None, subset=None, with_labels=True):
        '''
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return results

    def report(self, results, i):
        '''
        Use this function to report results to hyperparameter optimization frameworks
        also you can add key 'objective' to the results dictionary to report the final scores.
        '''
        return results

    def early_stopping(self, results, i):
        '''
        Use this function to early stop your model based on the results or any other metric in the algorithm class
        '''
        return False

    def __call__(self, subset='test', with_labels=True, enable_tqdm=None):

        with torch.no_grad():
            self.set_mode(training=False)

            results = defaultdict(lambda: defaultdict(list))

            results = self.preprocess_inference(results=results, subset=subset, with_labels=with_labels)

            if type(subset) is str:
                desc = subset
            else:
                desc = 'dataloader'

            if enable_tqdm is None:
                enable_tqdm = self.enable_tqdm

            dataloader = self.build_dataloader(subset)
            data_generator = self.data_generator(dataloader)
            for i, sample in tqdm(data_generator, enable=enable_tqdm, notebook=(not self.ddp), desc=desc, total=len(dataloader)):
                results = self.inference(sample=sample, results=results, subset=subset, with_labels=with_labels)

            results = self.postprocess_inference(sample=sample, results=results, subset=subset, with_labels=with_labels)

        return results

    def __iter__(self):

        all_train_results = defaultdict(dict)
        all_eval_results = defaultdict(dict)

        try:

            eval_generator = self.inner_loop(self.n_epochs + 1, subset=self.eval_subset, training=False)
            for i, train_results in enumerate(self.inner_loop(self.n_epochs, subset='train', training=True)):

                for k_type in train_results.keys():
                    for k_name, v in train_results[k_type].items():
                        all_train_results[k_type][k_name] = v

                with torch.no_grad():

                    validation_results = next(eval_generator)

                    for k_type in validation_results.keys():
                        for k_name, v in validation_results[k_type].items():
                            all_eval_results[k_type][k_name] = v

                results = {'train': all_train_results, self.eval_subset: all_eval_results}

                if self.hpo is not None:
                    results = self.report(results, i)

                yield results
                if self.early_stopping(results, i):
                    return

                all_train_results = defaultdict(dict)
                all_eval_results = defaultdict(dict)

        except KeyboardInterrupt:
            logger.error(f"KeyboardInterrupt: Training was interrupted, reloads last checkpoint and exits")
            self.experiment.reload_checkpoint(self)
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

        state = {'aux': aux}

        wrapper = copy.deepcopy if path is None else (lambda x: x)

        for k, net in self.networks.items():
            state[f"{k}_parameters"] = wrapper(net.state_dict())
            if pickle_model:
                state[f"{k}_model"] = net

        for k, optimizer in self.optimizers.items():
            state[f"{k}_optimizer"] = wrapper(optimizer.state_dict())

        if path is not None:
            torch.save(state, path)
        else:
            return state

    def load_checkpoint(self, path, strict=True):

        if type(path) is str:
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

        return state['aux']

    def fit(self, dataset=None, dataloaders=None, timeout=0, collate_fn=None,
                   worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2, **kwargs):
        '''
        For training purposes
        '''

        def algorithm_generator_single(experiment, *args, **kwargs):

            self.load_dataset(dataset=dataset, dataloaders=dataloaders, timeout=0, collate_fn=None,
                              worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2)

            return self

        if self.experiment.args.parallel == 1:
            algorithm_generator = algorithm_generator_single
        else:
            raise NotImplementedError("To continue training in parallel mode: please re-run experiment() with "
                                      "your own algorithm generator and a new dataset")

        return self.experiment(algorithm_generator, **kwargs)

    def evaluate(self, *args, **kwargs):
        '''
        For validation and test purposes (when labels are known)
        '''
        return self(*args, with_labels=True, **kwargs)

    def predict(self, *args, **kwargs):
        '''
        For real data purposes (when labels are unknown)
        '''

        return self(*args, with_labels=False, **kwargs)
