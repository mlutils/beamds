from collections import defaultdict
from torch import nn
import torch
import copy
from .utils import tqdm_beam as tqdm
import numpy as np
from .model import BeamOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from .utils import finite_iterations, to_device


class Algorithm(object):

    def __init__(self, networks, dataloaders, experiment, dataset=None, optimizers=None, rank=0, store_initial_weights=False):

        self.experiment = experiment
        self.device = experiment.device
        self.rank = rank
        self.world_size = experiment.parallel
        self.dataloaders = dataloaders
        self.dataset = dataset

        if type(networks) is not dict:
            networks = {'net': networks}

        self.networks = networks

        if optimizers is None:
            self.networks = {k: self.register_network(v) for k, v in networks.items()}
            optimizers = {k: BeamOptimizer(v, dense_args={'lr': experiment.lr_d,
                                                     'weight_decay': experiment.weight_decay,
                                                      'eps': experiment.eps},
                                       sparse_args={'lr': experiment.lr_s, 'eps': experiment.eps},
                                       ) for k, v in self.networks.items()}

        self.optimizers = optimizers

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

        if store_initial_weights:
            self.initial_weights = self.save_checkpoint()

    def register_network(self, net):
        net = net.to(self.device)
        if self.world_size > 1:
            DDP(net, device_ids=[self.rank])
        return net

    def get_optimizers(self):
        return self.optimizers

    def get_networks(self):
        return self.networks

    def process_sample(self, sample):
        return to_device(sample, self.device)

    def data_generator(self, subset):
        for i, sample in enumerate(self.dataloaders[subset]):
            sample = self.process_sample(sample)
            yield i, sample

    def postprocess_epoch(self, sample=None, aux=None, results=None, epoch=None, subset=None, training=True):
        '''
        :param epoch: epoch number
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return aux, results

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

    def inner_loop(self, n_epochs, subset, training=True):

        for n in range(n_epochs):

            aux = self.preprocess_epoch(epoch=n, subset=subset, training=training)
            self.set_mode(training=training)
            results = defaultdict(lambda: defaultdict(list))

            data_generator = self.data_generator(subset)
            for i, sample in tqdm(finite_iterations(data_generator, self.epoch_length[subset]),
                                  enable=True, desc=subset):

                aux, results = self.iteration(sample=sample, aux=aux, results=results, subset=subset, training=training)

            aux, results = self.postprocess_epoch(sample=sample, aux=aux, results=results,
                                                  subset=subset, epoch=n, training=training)

            yield results

        return


    def postprocess_inference(self, sample, aux, results, subset):
        '''
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return aux, results

    def preprocess_inference(self, aux, subset):
        '''
        :param aux: auxiliary data dictionary - possibly from previous epochs
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return aux

    def inference(self, sample, aux, results, subset):
        '''
        :param sample: the data fetched by the dataloader
        :param aux: a dictionary of auxiliary data
        :param results: a dictionary of dictionary of lists containing results of
        :param subset: name of dataset subset (usually train/validation/test)
        :return:
        loss: the loss fo this iteration
        aux: an auxiliary dictionary with all the calculated data needed for downstream computation (e.g. to calculate accuracy)
        '''
        return aux, results

    def __call__(self, subset='test'):

        with torch.no_grad():
            self.set_mode(training=False)
            aux = self.preprocess_inference(None, subset)

            results = defaultdict(lambda: defaultdict(list))

            data_generator = self.data_generator(subset)
            for i, sample in tqdm(data_generator, enable=True, desc=subset):
                aux, results = self.inference(sample, aux, results, subset)

            aux, results = self.postprocess_inference(sample, aux, results, subset)

        return results

    def __iter__(self):

        all_train_results = defaultdict(dict)
        all_eval_results = defaultdict(dict)

        eval_generator = self.inner_loop(self.n_epochs + 1, subset=self.eval_subset, training=False)
        for train_results in self.inner_loop(self.n_epochs, subset='train', training=True):

            for k_type in train_results.keys():
                for k_name, v in train_results[k_type].items():
                    all_train_results[k_type][k_name] = v

            with torch.no_grad():

                validation_results = next(eval_generator)

                for k_type in validation_results.keys():
                    for k_name, v in validation_results[k_type].items():
                        all_eval_results[k_type][k_name] = v

            results = {'train': all_train_results, self.eval_subset: all_eval_results}
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
                if hasattr(dataloader, 'train'):
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

    def load_checkpoint(self, path, strict=False):

        if type(path) is str:
            state = torch.load(path, map_location=self.device)
        else:
            state = path

        for k, net in self.networks.items():
            net.load_state_dict(state[f"{k}_parameters"], strict=strict)

        for k, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state[f"{k}_optimizer"], strict=strict)

        return state['aux']

