from collections import defaultdict
from torch import nn
import torch
import copy
from .utils import tqdm_beam as tqdm
from .utils import logger
import numpy as np
from .model import BeamOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from .utils import finite_iterations, to_device, check_type, rate_string_format, concat_data
from .dataset import UniversalBatchSampler, UniversalDataset, TransformedDataset
from timeit import default_timer as timer


class Algorithm(object):

    def __init__(self, hparams, networks=None, optimizers=None):

        self._experiment = None
        self.trial = None

        self.hparams = hparams

        self.device = hparams.device
        self.ddp = hparams.ddp
        self.hpo = hparams.hpo

        # some experiment hyperparameters
        self.half = hparams.half
        self.enable_tqdm = hparams.enable_tqdm if hparams.tqdm_threshold == 0 else None
        self.n_epochs = hparams.n_epochs
        self.batch_size_train = hparams.batch_size_train
        self.batch_size_eval = hparams.batch_size_eval

        self.cuda = ('cpu' not in str(self.device))
        self.pin_memory = self.cuda
        self.autocast_device = 'cuda' if self.cuda else 'cpu'
        self.amp = hparams.amp if self.cuda else False
        self.epoch = 0

        if networks is None:
            networks = {}
        elif issubclass(type(networks), nn.Module):
            networks = {'net': networks}
        elif check_type(networks).major == 'dict':
            pass
        else:
            raise NotImplementedError("Network type is unsupported")

        for k in networks.keys():
            networks[k] = self.register_network(networks[k])
        self.networks = networks

        if optimizers is None:
            self.optimizers = {k: BeamOptimizer(v, dense_args={'lr': self.hparams.lr_dense,
                                                          'weight_decay': self.hparams.weight_decay,
                                                           'betas': (self.hparams.beta1, self.hparams.beta2),
                                                          'eps': self.hparams.eps},
                                           sparse_args={'lr': self.hparams.lr_sparse,
                                                        'betas': (self.hparams.beta1, self.hparams.beta2),
                                                        'eps': self.hparams.eps},
                                           clip=self.hparams.clip_gradient, amp=self.amp, accumulate=self.hparams.accumulate
                                           ) for k, v in self.networks.items()}

        elif issubclass(type(optimizers), dict):
            self.optimizers = {}
            for k, o in optimizers.items():
                if callable(o):
                    self.optimizers[k] = self.networks[k]
                else:
                    o.load_state_dict(o.state_dict())
                    self.optimizers[k] = o

        elif issubclass(type(optimizers), torch.optim.Optimizer) or issubclass(type(optimizers), BeamOptimizer):
            optimizers.load_state_dict(optimizers.state_dict())
            self.optimizers = {'net': optimizers}

        elif callable(optimizers):
            self.optimizers = {'net': optimizers(self.networks['net'])}
        else:
            raise NotImplementedError

        if hparams.store_initial_weights:
            self.initial_weights = self.save_checkpoint()

        if hparams.reload_path is not None:
            self.load_checkpoint(hparams.reload_path)

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

    def load_dataset(self, dataset=None, dataloaders=None, batch_size_train=None, batch_size_eval=None,
                     oversample=None, weight_factor=None, expansion_size=None,timeout=0, collate_fn=None,
                     worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2):

        assert dataloaders is not None or dataset is not None, "Either dataset or dataloader must be supplied"
        self.dataset = dataset

        if dataloaders is None:

            batch_size_train = self.hparams.batch_size_train if batch_size_train is None else batch_size_train
            batch_size_eval = self.hparams.batch_size_eval if batch_size_eval is None else batch_size_eval
            oversample = (self.hparams.oversampling_factor > 0) if oversample is None else oversample
            weight_factor = self.hparams.oversampling_factor if weight_factor is None else weight_factor
            expansion_size = self.hparams.expansion_size if expansion_size is None else expansion_size

            dataset.build_samplers(batch_size_train, eval_batch_size=batch_size_eval,
                                   oversample=oversample, weight_factor=weight_factor, expansion_size=expansion_size)

            dataloaders = dataset.build_dataloaders(num_workers=self.hparams.cpu_workers,
                                                    pin_memory=self.pin_memory,
                                                   timeout=timeout, collate_fn=collate_fn,
                                                   worker_init_fn=worker_init_fn,
                                                   multiprocessing_context=multiprocessing_context, generator=generator,
                                                   prefetch_factor=prefetch_factor)

        self.dataloaders = dataloaders
        self.epoch_length = {}

        self.eval_subset = 'validation' if 'validation' in dataloaders.keys() else 'test'
        self.epoch_length['train'] = self.hparams.epoch_length_train
        self.epoch_length[self.eval_subset] = self.hparams.epoch_length_eval

        if self.epoch_length['train'] is None:
            dataset = dataloaders['train'].dataset
            self.epoch_length['train'] = len(dataset.indices_split['train'])

        if self.epoch_length[self.eval_subset] is None:
            dataset = dataloaders[self.eval_subset].dataset
            self.epoch_length[self.eval_subset] = len(dataset.indices_split[self.eval_subset])

        if self.hparams.scale_epoch_by_batch_size:
            self.epoch_length[self.eval_subset] = self.epoch_length[self.eval_subset] // self.batch_size_eval
            self.epoch_length['train'] = self.epoch_length['train'] // self.batch_size_train

        if 'test' in dataloaders.keys():
            self.epoch_length['test'] = len(dataloaders['test'])

        if self.n_epochs is None:
            self.n_epochs = self.hparams.total_steps // self.epoch_length['train']

        if self.dataset is None:
            self.dataset = next(iter(self.dataloaders.values())).dataset

    def register_network(self, net):

        if self.half:
            net = net.half()

        net = net.to(self.device)

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

            if hasattr(dataset, 'index') and dataset.index is not None:
                index = dataset.index.index.values
            else:
                index = len(dataset)

            sampler = UniversalBatchSampler(index, self.hparams.batch_size_eval, shuffle=False,
                                            tail=True, once=True)
            dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=None,
                                                     num_workers=0, pin_memory=self.pin_memory)

        else:

            if check_type(subset).minor in ['list', 'tuple']:
                dataset = UniversalDataset(*subset)
            elif check_type(subset).minor in ['dict']:
                dataset = UniversalDataset(**subset)
            else:
                dataset = UniversalDataset(subset)

            sampler = UniversalBatchSampler(len(dataset), self.hparams.batch_size_eval, shuffle=False,
                                            tail=True, once=True)
            dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=None,
                                                     num_workers=0, pin_memory=self.pin_memory)

        return dataloader

    def data_generator(self, subset):

        dataloader = self.build_dataloader(subset)
        for i, (ind, sample) in enumerate(dataloader):
            sample = self.process_sample(sample)
            yield i, (ind, sample)

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

            self.set_mode(training=training)
            results = self.preprocess_epoch(results=results, epoch=n, training=training)

            data_generator = self.data_generator(subset)
            for i, (ind, sample) in tqdm(finite_iterations(data_generator, self.epoch_length[subset]),
                                  enable=self.enable_tqdm, notebook=(not self.ddp),
                                  threshold=self.hparams.tqdm_threshold, stats_period=self.hparams.tqdm_stats,
                                  desc=subset, total=self.epoch_length[subset]):

                with torch.autocast(self.autocast_device, enabled=self.amp):
                    results = self.iteration(sample=sample, results=results, counter=i, training=training, index=ind)

            results = self.postprocess_epoch(sample=sample, index=ind, results=results, epoch=n, training=training)

            delta = timer() - t0
            n_iter = i + 1
            batch_size = self.batch_size_train if training else self.batch_size_eval

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

    def __call__(self, subset, predicting=False, enable_tqdm=None):

        with torch.no_grad():

            self.set_mode(training=False)
            results = defaultdict(lambda: defaultdict(list))
            transforms = []
            index = []
            results = self.preprocess_inference(results=results, subset=subset, predicting=predicting)

            desc = subset if type(subset) is str else ('predict' if predicting else 'evaluate')

            if enable_tqdm is None:
                enable_tqdm = self.enable_tqdm

            dataloader = self.build_dataloader(subset)
            data_generator = self.data_generator(dataloader)
            for i, (ind, sample) in tqdm(data_generator, enable=enable_tqdm,
                                  threshold=self.hparams.tqdm_threshold, stats_period=self.hparams.tqdm_stats,
                                  notebook=(not self.ddp), desc=desc, total=len(dataloader)):
                transform, results = self.inference(sample=sample, results=results, subset=subset, predicting=predicting,
                                         index=ind)
                transforms.append(transform)
                index.append(ind)

            index = torch.cat(index)
            transforms = concat_data(transforms)
            results = self.postprocess_inference(sample=sample, index=ind, transforms=transforms,
                                                 results=results, subset=subset, predicting=predicting)

            dataset = UniversalDataset(transforms, index=index)
            # results = concat_results(results)
            dataset.set_statistics(results)

        return dataset

    def __iter__(self):

        all_train_results = defaultdict(dict)
        all_eval_results = defaultdict(dict)

        eval_generator = self.epoch_iterator(self.n_epochs, subset=self.eval_subset, training=False)
        for i, train_results in enumerate(self.epoch_iterator(self.n_epochs, subset='train', training=True)):

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

            self.epoch += 1
            yield results

            if self.early_stopping(results, i):
                return

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

        state = {'aux': aux, 'epoch': self.epoch}

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

        self.epoch = state['epoch']
        return state['aux']

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
