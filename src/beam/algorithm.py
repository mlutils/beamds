from collections import defaultdict
from torch import nn
import torch
import copy
from .utils import tqdm_beam as tqdm
import numpy as np
from .model import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from .utils import finite_iterations


class Algorithm(object):

    def __init__(self, networks, experiment, dataloader=None, dataset=None, optimizers=None, rank=0):

        self.device = experiment.device
        self.rank = rank
        self.world_size = experiment.parallel

        if type(networks) is not dict:
            networks = {'net': networks}
        self.networks = networks

        if optimizers is None:
            self.networks = {k: self.register_network(v) for k, v in networks.items()}
            optimizers = {k: Optimizer(v, dense_args={'lr': experiment.lr_d,
                                                     'weight_decay': experiment.eps,
                                                      'eps': experiment.weight_decay},
                                       sparse_args={'lr': experiment.lr_s, 'eps': experiment.eps},
                                       ) for k, v in self.networks.items()}

        if dataloader is None:
            assert dataset is not None, 'If dataloader is not provided, you must provide a dataset instance'

            dataloader = {}
            for subset, index in dataset.indices.items():

                batch_size = experiment.batch_size_train if 'train' in subset else experiment.batch_size_test
                batch_size =  batch_size if batch_size is not None else experiment.batch_size

                dataloader[subset] = dataset.dataloader(subset=subset, batch_size=batch_size,
                                                        num_workers=experiment.cpu_workers, pin_memory=True)

        self.dataloader = dataloader

        self.optimizers = optimizers
        self.batch_size_train = self.batch_size if self.batch_size_train is None else self.batch_size_train
        self.batch_size_test = self.batch_size if self.batch_size_test is None else self.batch_size_test

        self.epoch_length = {}

        for subset, index in dataset.indices.items():

            epoch_length = experiment.epoch_length_train if 'train' in subset else experiment.epoch_length_test
            epoch_length = experiment.epoch_length if epoch_length is None else epoch_length
            self.epoch_length[subset] = len(index) if epoch_length is None else epoch_length

        self.n_epochs = experiment.total_steps // experiment.epoch_length_train if experiment.n_epochs is None else experiment.n_epochs


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

    def data_generator(self, subset):
        for i, sample in enumerate(self.dataloader[subset]):
            sample = self.process_sample(sample)
            yield i, sample

    def postprocess_epoch(self, sample, aux, results, epoch, subset, training=True):
        '''
        :param epoch: epoch number
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return aux, results

    def preprocess_epoch(self, aux, epoch, subset, training=True):
        '''
        :param aux: auxiliary data dictionary - possibly from previous epochs
        :param epoch: epoch number
        :param subset: name of dataset subset (usually train/validation/test)
        :return: None
        a placeholder for operations to execute before each epoch, e.g. shuffling/augmenting the dataset
        '''
        return aux

    def iteration(self, sample, aux, results, subset, training=True):
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
        # loss = torch.tensor(0., device=self.device)
        aux = {}
        return aux, results

    def inner_loop(self, n_epochs, subset, training=True):

        for n in range(n_epochs):

            aux = self.preprocess_epoch(None, 0, subset, training=training)
            self.set_mode(training=training)
            results = defaultdict(lambda: defaultdict(list))

            data_generator = self.data_generator(subset)
            for i, sample in tqdm(finite_iterations(data_generator, self.epoch_length[subset]),
                                  enable=True, desc=subset):

                aux, results = self.iteration(sample, aux, results, subset, training=training)

            aux, results = self.postprocess_epoch(sample, aux, results, subset, n, training=training)

            yield results

        return

    def evaluate(self, subset='test', training=False):
        with torch.no_grad():
            pass

    def __iter__(self):

        all_train_results = defaultdict(dict)
        all_validation_results = defaultdict(dict)

        validation_generator = self.inner_loop(self.n_epochs + 1, subset='validation', training=False)
        for train_results in self.inner_loop(self.n_epochs, subset='train', training=True):

            for k_type in train_results.keys():
                for k_name, v in train_results[k_type].items():
                    all_train_results[k_type][k_name] = v

            with torch.no_grad():

                validation_results = next(validation_generator)

                for k_type in validation_results.keys():
                    for k_name, v in validation_results[k_type].items():
                        all_validation_results[k_type][k_name] = v

            results = {'train': all_train_results, 'validation': all_validation_results}
            yield results

            all_train_results = defaultdict(dict)
            all_validation_results = defaultdict(dict)

    def set_mode(self, training=True):

        for net in self.networks.values():

            if training:
                net.train()
            else:
                net.eval()

            for dataloader in self.dataloader.values():
                if hasattr(dataloader, 'train'):
                    if training:
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

