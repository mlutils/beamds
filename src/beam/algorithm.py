from collections import defaultdict
from torch import nn
import torch
import copy
from .utils import tqdm_beam as tqdm
import numpy as np
from .model import BeamOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from .utils import finite_iterations, to_device, check_type
from .dataset import UniversalBatchSampler, UniversalDataset


class Algorithm(object):

    def __init__(self, networks, dataloaders, experiment, dataset=None, optimizers=None, store_initial_weights=False):

        self.experiment = experiment

        self.device = experiment.device
        self.ddp = experiment.ddp
        self.half = experiment.half
        self.enable_tqdm = experiment.enable_tqdm
        self.amp = experiment.amp
        self.hpo = experiment.hpo
        self.trial = experiment.trial

        self.dataloaders = dataloaders
        self.dataset = dataset

        if type(networks) is not dict:
            networks = {'net': networks}

        self.networks = networks

        if optimizers is None:
            self.networks = {k: self.register_network(v) for k, v in networks.items()}
            self.optimizers = {k: BeamOptimizer(v, dense_args={'lr': experiment.lr_dense,
                                                          'weight_decay': experiment.weight_decay,
                                                           'betas': (experiment.beta1, experiment.beta2),
                                                          'eps': experiment.eps},
                                           sparse_args={'lr': experiment.lr_sparse,
                                                        'betas': (experiment.beta1, experiment.beta2),
                                                        'eps': experiment.eps},
                                           clip=experiment.clip, amp=self.amp, accumulate=experiment.accumulate
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

        self.batch_size_train = experiment.batch_size_train
        self.batch_size_eval = experiment.batch_size_eval

        self.epoch_length = {}

        self.eval_subset = 'validation' if 'validation' in self.dataloaders.keys() else 'test'
        self.epoch_length['train'] = experiment.epoch_length_train
        self.epoch_length[self.eval_subset] = experiment.epoch_length_eval

        if self.epoch_length['train'] is None:
            dataset = self.dataloaders['train'].dataset
            self.epoch_length['train'] = len(dataset.indices_split['train'])

        if self.epoch_length[self.eval_subset] is None:
            dataset = self.dataloaders[self.eval_subset].dataset
            self.epoch_length[self.eval_subset] = len(dataset.indices_split[self.eval_subset])

        self.n_epochs = experiment.n_epochs
        if self.n_epochs is None:
            self.n_epochs = experiment.total_steps // self.epoch_length['train']

        if experiment.scale_epoch_by_batch_size:
            self.epoch_length[self.eval_subset] = self.epoch_length[self.eval_subset] // self.batch_size_eval
            self.epoch_length['train'] = self.epoch_length['train'] // self.batch_size_train

        if 'test' in self.dataloaders.keys():
            self.epoch_length['test'] = len(self.dataloaders['test'])

        if store_initial_weights:
            self.initial_weights = self.save_checkpoint()

        if experiment.load_model:
            experiment.reload_checkpoint(self)

        if self.dataset is None:
            self.dataset = self.dataloaders['train'].dataset

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
            sampler = UniversalBatchSampler(len(dataset), self.experiment.batch_size_eval, shuffle=False,
                                            tail=True, once=True)
            dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=None,
                                                     num_workers=0, pin_memory=True)

        else:

            if check_type(subset).minor in ['list', 'tuple']:
                dataset = UniversalDataset(*subset)
            elif check_type(subset).minor in ['dict']:
                dataset = UniversalDataset(**subset)
            else:
                dataset = UniversalDataset(subset)

            sampler = UniversalBatchSampler(len(dataset), self.experiment.batch_size_eval, shuffle=False,
                                            tail=True, once=True)
            dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=None,
                                                     num_workers=0, pin_memory=True)

        return dataloader

    def data_generator(self, subset):

        dataloader = self.build_dataloader(subset)
        for i, sample in enumerate(dataloader):
            sample = self.process_sample(sample)
            yield i, sample

    def preprocess_epoch(self, aux=None, epoch=None, subset=None, training=True):
        '''
        :param aux: auxiliary data dictionary - possibly from previous epochs
        :param epoch: epoch number
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return aux

    def iteration(self, sample=None, aux=None, results=None, subset=None, training=True):
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
        return aux, results

    def postprocess_epoch(self, sample=None, aux=None, results=None, epoch=None, subset=None, training=True):
        '''
        :param epoch: epoch number
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return aux, results

    def inner_loop(self, n_epochs, subset, training=True):

        for n in range(n_epochs):

            aux = defaultdict(lambda: defaultdict(list))
            results = defaultdict(lambda: defaultdict(list))

            aux = self.preprocess_epoch(aux=aux, epoch=n, subset=subset, training=training)
            self.set_mode(training=training)
            data_generator = self.data_generator(subset)
            for i, sample in tqdm(finite_iterations(data_generator, self.epoch_length[subset]),
                                  enable=self.enable_tqdm, notebook=(not self.ddp),
                                  desc=subset, total=self.epoch_length[subset] - 1):
                # print(i)
                aux, results = self.iteration(sample=sample, aux=aux, results=results, subset=subset, training=training)

            aux, results = self.postprocess_epoch(sample=sample, aux=aux, results=results,
                                                  subset=subset, epoch=n, training=training)

            yield results

        return

    def preprocess_inference(self, aux=None, subset=None, with_labels=True):
        '''
        :param aux: auxiliary data dictionary - possibly from previous epochs
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return aux

    def inference(self, sample=None, aux=None, results=None, subset=None, with_labels=True):
        '''
        :param sample: the data fetched by the dataloader
        :param aux: a dictionary of auxiliary data
        :param results: a dictionary of dictionary of lists containing results of
        :param subset: name of dataset subset (usually train/validation/test)
        :return:
        loss: the loss fo this iteration
        aux: an auxiliary dictionary with all the calculated data needed for downstream computation (e.g. to calculate accuracy)
        '''
        aux, results = self.iteration(sample=sample, aux=aux, results=results, subset=subset, training=False)
        return aux, results

    def postprocess_inference(self, sample=None, aux=None, results=None, subset=None, with_labels=True):
        '''
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return aux, results

    def report(self, results, i):
        '''
        Use this function to report results to hyperparameter optimization frameworks
        also you can add key 'objective' to the results dictionary to report the final scores.
        '''
        return results

    def __call__(self, subset='test', with_labels=True, enable_tqdm=None):

        with torch.no_grad():
            self.set_mode(training=False)

            aux = defaultdict(lambda: defaultdict(list))
            results = defaultdict(lambda: defaultdict(list))

            aux = self.preprocess_inference(aux=aux, subset=subset, with_labels=with_labels)

            if type(subset) is str:
                desc = subset
            else:
                desc = 'dataloader'

            if enable_tqdm is None:
                enable_tqdm = self.enable_tqdm

            dataloader = self.build_dataloader(subset)
            data_generator = self.data_generator(dataloader)
            for i, sample in tqdm(data_generator, enable=enable_tqdm, notebook=(not self.ddp), desc=desc, total=len(dataloader)):
                aux, results = self.inference(sample=sample, aux=aux, results=results, subset=subset, with_labels=with_labels)

            aux, results = self.postprocess_inference(sample=sample, aux=aux, results=results, subset=subset, with_labels=with_labels)

        return results

    def __iter__(self):

        all_train_results = defaultdict(dict)
        all_eval_results = defaultdict(dict)

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

            all_train_results = defaultdict(dict)
            all_eval_results = defaultdict(dict)

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

    def fit(self, *args, **kwargs):
        '''
        For training purposes
        '''

        def algorithm_generator(experiment):
            return self
        return self.experiment(algorithm_generator, *args, **kwargs)

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
