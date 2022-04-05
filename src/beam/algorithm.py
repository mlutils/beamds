from collections import defaultdict
from torch import nn
import torch
import copy
from .utils import tqdm_beam as tqdm
import numpy as np
from .model import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP


class Algorithm(object):

    def __init__(self, networks, dataloader, experiment, rank=0, optimizers=None):

        self.device = experiment.device
        self.rank = rank
        self.world_size = experiment.parallel

        if type(networks) is not dict:
            networks = {'net': networks}
        self.networks = networks

        if optimizers is None:
            self.networks = {k: self.register_network(v) for k, v in networks.items()}
            optimizers = {k: Optimizer(v, dense_ars={'lr': experiment.lr_d,
                                                     'weight_decay': experiment.weight_decay, 'eps': 1e-4},
                                       sparse_args={'lr': experiment.lr_s, 'eps': 1e-4},
                                       ) for k, v in self.networks.items()}

        self.optimizers = optimizers
        self.dataloader = dataloader

        self.n_epochs = experiment.total_steps // experiment.epoch_length
        self.epoch_length = experiment.epoch_length

    def register_network(self, net):
        net = net.to(self.device)
        if self.world_size > 1:
            DDP(net, device_ids=[self.rank])
        return net

    def postprocess(self, sample):

        for name, var in sample.items():
            sample[name] = var.to(self.device)

        return sample

    def reset_opt(self, optimizer):

        if type(optimizer) is Optimizer:
            optimizer.reset()
        else:
            optimizer.state = defaultdict(dict)

    def reset_networks(self, networks_dict, optimizers_dict):

        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        for net in networks_dict:
            net = getattr(self, net)
            net.apply(init_weights)

        for optim in optimizers_dict:
            optim = getattr(self, optim)
            optim.state = defaultdict(dict)

    def get_optimizers(self):
        return self.optimizers

    def get_networks(self):
        return self.networks

    def process_sample(self, sample):

        if type(sample) is dict:
            sample = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in sample.items()}
        elif type(sample) is list:
            sample = [s.to(self.device) if hasattr(s, 'to') else s for s in sample]
        else:
            raise NotImplementedError
        return sample

    def data_generator(self, train):

        train = 'train' if train else 'test'
        for i, sample in enumerate(self.dataloader[train]):
            sample = self.process_sample(sample)
            yield i, sample

    def postprocess_epoch(self, sample, aux, results, epoch, train=True):
        '''
        :param epoch: epoch number
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return aux, results

    def preprocess_epoch(self, aux, epoch, train=True):
        '''
        :param aux: auxiliary data dictionary - possibly from previous epochs
        :param epoch: epoch number
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return aux

    def iteration(self, sample, aux, results, train=True):
        '''
        :param sample: the data fetched by the dataloader
        :param aux: a dictionary of auxiliary data
        :param results: a dictionary of dictionary of lists containing results of
        :param train: train/test flag
        :return:
        loss: the loss fo this iteration
        aux: an auxiliary dictionary with all the calculated data needed for downstream computation (e.g. to calculate accuracy)
        '''
        # loss = torch.tensor(0., device=self.device)
        aux = {}
        return aux, results

    def inner_loop(self, n_epochs, train=True):

        aux = self.preprocess_epoch(None, 0, train=train)
        self.set_mode(train=train)
        results = defaultdict(lambda: defaultdict(list))

        for i, sample in tqdm(self.data_generator(train), enable=train):

            aux, results = self.iteration(sample, aux, results, train=train)

            if not (i + 1) % self.epoch_length:

                n = (i + 1) // self.epoch_length
                aux, results = self.postprocess_epoch(sample, aux, results, n, train=train)

                if n >= n_epochs:
                    return results
                else:
                    yield results

                aux = self.preprocess_epoch(aux, n, train=train)
                self.set_mode(train=train)
                results = defaultdict(lambda: defaultdict(list))

    def __iter__(self):

        all_train_results = defaultdict(dict)
        all_test_results = defaultdict(dict)

        test_generator = self.inner_loop(self.n_epochs + 1, train=False)
        for train_results in self.inner_loop(self.n_epochs, train=True):

            for k_type in train_results.keys():
                for k_name, v in train_results[k_type].items():
                    all_train_results[k_type][k_name] = v

            with torch.no_grad():

                # if not self.rank:
                #     test_results = next(test_generator)
                # else:
                #     test_results = {}
                test_results = next(test_generator)

                for k_type in test_results.keys():
                    for k_name, v in test_results[k_type].items():
                        all_test_results[k_type][k_name] = v

            results = {'train': all_train_results, 'test': all_test_results}
            yield results

            all_train_results = defaultdict(dict)
            all_test_results = defaultdict(dict)

    def set_mode(self, train=True):

        for net in self.networks.values():

            if train:
                net.train()
            else:
                net.eval()

            for dataloader in self.dataloader.values():
                if hasattr(dataloader, 'train'):
                    if train:
                        dataloader.dataset.train()
                    else:
                        dataloader.dataset.eval()

    def state_dict(self, net):
        return copy.deepcopy(net.state_dict())

    def load_state_dict(self, net, state):
        net.load_state_dict(state, strict=False)

    def store_net_0(self):

        self.net_0 = {}

        for k, net in self.networks.items():
            net.eval()
            self.net_0[k] = self.state_dict(net)

    def save_checkpoint(self, path=None, aux=None, pickle_model=False):

        state = {'aux': aux}

        for k, net in self.networks.items():
            state[k] = self.state_dict(net)
            state[f"{k}_model"] = net

        for k, optimizer in self.optimizers.items():
            state[f"{k}_optimizer"] = copy.deepcopy(optimizer.state_dict())

        if path is not None:
            torch.save(state, path)

        return state

    def load_checkpoint(self, path):

        if type(path) is str:
            state = torch.load(path, map_location=self.device)
        else:
            state = path

        for k, net in self.networks.items():
            self.load_state_dict(net, state[k])

        for k, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state[k])

        return state['aux']

