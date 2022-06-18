import itertools
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from .utils import check_type, slice_to_array, as_tensor
import pandas as pd
import math
import hashlib
import sys
import warnings
from .data_tensor import DataTensor


class PackedFolds(object):

    def __init__(self, data, index=None, names=None, fold=None, fold_index=None, device='cpu'):

        self.names_dict = None
        self.names = None
        
        if names is not None:
            self.names = names
            self.names_dict = {n: i for i, n in enumerate(self.names)}

        data_type = check_type(data)

        if data_type.minor == 'list':
            self.data = [as_tensor(v, device=device) for v in data]
            
        elif data_type.minor == 'dict':
            if names is None:
                self.names = list(data.keys())
                self.names_dict = {n: i for i, n in enumerate(self.names)}
                
            self.data = [as_tensor(data[k], device=device) for k in self.names]

        else:
            raise ValueError

        fold_type = check_type(fold)

        if fold_type.element == 'str':
            fold, names_map = pd.factorize(fold)

            if self.names_dict is not None:
                assert all([i == self.names_dict[n] for i, n in enumerate(names_map)]), "fold and data maps must match"
            
            else:
                self.names = list(names_map)
                self.names_dict = {n: i for i, n in enumerate(self.names)}

            fold = as_tensor(fold)
            
        elif fold_type.element == 'int':   
            fold = as_tensor(fold)
            
        elif fold is None:
            fold = torch.cat([i * torch.ones(len(d), dtype=torch.int64) for i, d in enumerate(self.data)])

        else:
            raise ValueError

        if names is None:
            self.names = list(range(len(self.data)))
            self.names_dict = {n: i for i, n in enumerate(self.names)}

    
        if fold_index is not None:
            fold_index = as_tensor(fold_index)
        else:
            fold_index = torch.cat([torch.arange(len(d)) for d in self.data])

        index_type = check_type(index)

        if index_type.minor == 'list':
            index = torch.stack([as_tensor(v) for v in index])

        elif index_type.minor == 'dict':
            index = [as_tensor(index[k]) for k in self.names]

        elif  index_type.major == 'array':
            index = as_tensor(index)
            
        elif index is None:
            index = torch.arange(sum([len(d) for d in self.data]))
            
        else:
            raise ValueError

        lengths = torch.LongTensor([len(self.get_fold(k)) for k in self.names])
        cumsum =  torch.cumsum(lengths, dim=0)
        offset =  cumsum - lengths
        offset = offset[fold] + fold_index

        self.device = device

        info = {'fold': fold, 'fold_index': fold_index, 'offset': offset}
        self.info = DataTensor(info, index=index, device=device)

    def __len__(self):
        return len(self.info)

    def get_fold(self, name):
        return self.data[self.names_dict[name]]

    def apply(self, functions):

        functions_type = check_type(functions)

        if functions_type.minor == 'list':
            data = [f(d) for d, f in zip(self.data, functions)]

        elif functions_type.minor == 'dict':
            data = [f(self.get_fold(k)) for k, f in functions.items()]
        else:
            raise ValueError

        return PackedFolds(data=data, index=self.index, names=self.names, device=self.device)

    @property
    def values(self):

        data = torch.cat(self.data)
        return data[self.info['offset'].values]

    @property
    def fold(self):
        if len(self.names) == 1:
            return self.names[0]
        return 'hetrogenous'

    def to(self, device):

        self.data = [di.to(device) for di in self.data]
        self.info = self.info.to(device)
        self.device = device

    def __repr__(self):
        return repr({k: self.get_fold(k) for k in self.names})

    def __getitem__(self, ind):

        ind = slice_to_array(ind, l=len(self))
        info = self.info.loc[ind]

        fold, fold_index = info[['fold', 'fold_index']].values.T

        uq = torch.sort(torch.unique(fold)).values
        names = [self.names[i] for i in uq]

        if len(uq) == 1:
            return PackedFolds(data=[self.data[uq[0]][fold_index]], names=names, index=ind, device=self.device)

        fold_index = [fold_index[fold==i] for i in uq]
        data = [self.data[i][j] for i, j in zip(uq, fold_index)]

        return PackedFolds(data=data, names=names, index=ind, device=self.device)


class UniversalDataset(torch.utils.data.Dataset):

    def __init__(self, *args, device='cpu', **kwargs):
        super().__init__()
        self.indices_split = {}
        self.samplers = {}
        self.labels_split = {}

        # The training label is to be used when one wants to apply some data transformations/augmentations only in training mode
        self.training = False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(args):
                self.data = [as_tensor(v, device=device) for v in args]
            elif len(kwargs):
                self.data = {k: as_tensor(v, device=device) for k, v in kwargs.items()}
            else:
                self.data = None

    def train(self):
        self.training = True
    def eval(self):
        self.training = False

    def __getitem__(self, ind):

        if type(self.data) is dict:
            return {k: v[ind] for k, v in self.data.items()}
        elif len(self.data) == 1:
            return self.data[ind]
        else:
            return [v[ind] for v in self.data]

    @property
    def device(self):

        if type(self.data) is dict:
            t = next(iter(self.data.values()))
        elif type(self.data) is list:
            t = self.data[0]
        elif check_type(self.data).major == 'array':
            t = self.data
        else:
            raise NotImplementedError

        if check_type(self.data).minor == 'tensor':
            device = t.device
        else:
            raise NotImplementedError

        return device


    def __len__(self):
        if issubclass(type(self.data), dict):
            return len(next(iter(self.data.values())))
        elif issubclass(type(self.data), list):
            return len(self.data[0])
        elif check_type(self.data).major == 'array':
            return len(self.data)
        else:
            raise NotImplementedError

    def split(self, validation=None, test=None, seed=5782, stratify=False, labels=None,
                    test_split_method='uniform', time_index=None, window=None):
        '''
        validation, test can be lists of indices, relative size or absolute size

        The seed is fixed unless explicitly given, to make sure datasets are split the same in all instances.
        If one wish to obtain random splitting, seed should be None

        test_split_method should be uniform/time_based

        # TODO: add window_based split
        '''

        indices = np.arange(len(self))
        if time_index is None:
            time_index = indices

        if test is None:
            pass
        elif check_type(test).major == 'array':
            self.indices_split['test'] = torch.LongTensor(test)
            indices = np.sort(np.array(list(set(indices).difference(set(np.array(test))))))

            if labels is not None:
                self.labels_split['test'] = labels[self.indices_split['test']]
                labels = labels[indices]

        elif test_split_method == 'uniform':
            if labels is not None:
                indices, test, labels, self.labels_split['test'] = train_test_split(indices, labels, random_state=seed,
                                                                                    test_size=test, stratify=labels if stratify else None)
            else:
                indices, test = train_test_split(indices, random_state=seed, test_size=test)

            self.indices_split['test'] = torch.LongTensor(test)
            if seed is not None:
                seed = seed + 1

        elif test_split_method == 'time_based':
            ind_sort = np.argsort(time_index)
            indices = indices[ind_sort]

            self.indices_split['test'] = torch.LongTensor(indices[-int(test * len(indices)):])
            indices = indices[:-int(test * len(indices))]

            if labels is not None:
                labels = labels[ind_sort]
                self.labels_split['test'] = labels[self.indices_split['test']]
                labels = labels[indices]

        if validation is None:
            pass
        elif check_type(validation).major == 'array':
            self.indices_split['validation'] = torch.LongTensor(validation)
            indices = np.sort(np.array(list(set(indices).difference(set(np.array(validation))))))

            if labels is not None:
                self.labels_split['validation'] = labels[self.indices_split['validation']]
                labels = labels[indices]

        else:
            if type(validation) is float:
                validation = len(self) / len(indices) * validation

            if stratify and labels is not None:
                indices, validation, labels, self.labels_split['validation'] = train_test_split(indices, labels, random_state=seed,
                                                                                                test_size=validation, stratify=labels if stratify else None)
            else:
                indices, validation = train_test_split(indices, random_state=seed, test_size=validation)

            self.indices_split['validation'] = torch.LongTensor(validation)

        self.indices_split['train'] = torch.LongTensor(indices)
        self.labels_split['train'] = labels

    def build_samplers(self, batch_size, eval_batch_size=None, oversample=False, weight_factor=1., expansion_size=int(1e7)):

        if eval_batch_size is None:
            eval_batch_size = batch_size

        if 'test' in self.indices_split:
            self.samplers['test'] = UniversalBatchSampler(self.indices_split['test'],
                                                          eval_batch_size, shuffle=False, tail=True, once=True)

        if 'validation' in self.indices_split:
            probs = None
            if oversample and 'validation' in self.labels_split and self.labels_split['validation'] is not None:
                probs = compute_sample_weight('balanced', y=self.labels_split['validation']) ** weight_factor

            self.samplers['validation'] = UniversalBatchSampler(self.indices_split['validation'],
                                                          eval_batch_size, probs=probs, shuffle=True, tail=True, once=False)

        if 'train' in self.indices_split:
            probs = None
            if oversample and 'train' in self.labels_split and self.labels_split['train'] is not None:
                probs = compute_sample_weight('balanced', y=self.labels_split['train']) ** weight_factor

            self.samplers['train'] = UniversalBatchSampler(self.indices_split['train'],
                                                           batch_size, probs=probs, shuffle=True, tail=True,
                                                           once=False, expansion_size=expansion_size)

    def build_dataloaders(self, num_workers=0, pin_memory=None, timeout=0, collate_fn=None,
                   worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2):

        dataloaders = {}
        if pin_memory is None:
            try:
                d = str(self.device)
                pin_memory = False if 'cuda' in d else True
            except NotImplementedError:
                pin_memory = True

        if 'test' in self.samplers:
            sampler = self.samplers['test']
            persistent_workers = True if num_workers > 0 else False
            dataloaders['test'] = torch.utils.data.DataLoader(self, sampler=sampler, batch_size = None,
                                                 num_workers=num_workers, pin_memory=pin_memory, timeout=timeout,
                                                 worker_init_fn=worker_init_fn, collate_fn=collate_fn,
                                                 multiprocessing_context=multiprocessing_context, generator=generator,
                                                 prefetch_factor=prefetch_factor, persistent_workers=persistent_workers
                                                 )

        if 'validation' in self.samplers:
            sampler = self.samplers['validation']
            dataloaders['validation'] = torch.utils.data.DataLoader(self, sampler=sampler, batch_size = None,
                                                 num_workers=num_workers, pin_memory=pin_memory, timeout=timeout,
                                                 worker_init_fn=worker_init_fn, collate_fn=collate_fn,
                                                 multiprocessing_context=multiprocessing_context, generator=generator,
                                                 prefetch_factor=prefetch_factor)

        if 'train' in self.samplers:
            sampler = self.samplers['train']
            dataloaders['train'] = torch.utils.data.DataLoader(self, sampler=sampler,
                                                                    batch_size = None,
                                                                    num_workers=num_workers,
                                                                    pin_memory=pin_memory,
                                                                    timeout=timeout,
                                                                    worker_init_fn=worker_init_fn,
                                                                    collate_fn=collate_fn,
                                                                    multiprocessing_context=multiprocessing_context,
                                                                    generator=generator,
                                                                    prefetch_factor=prefetch_factor)

        return dataloaders

    def dataloader(self, batch_size, subset='train', length=None, shuffle=True, tail=True, once=False,
                   num_workers=0, pin_memory=True, timeout=0, collate_fn=None,
                   worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2,
                   persistent_workers=False):

        indices = self.indices_split[subset]

        sampler = UniversalBatchSampler(indices, batch_size, length=length, shuffle=shuffle, tail=tail, once=once)
        dataloader = torch.utils.data.DataLoader(self, sampler=sampler, batch_size=None,
                                                 num_workers=num_workers, pin_memory=pin_memory, timeout=timeout,
                                                 worker_init_fn=worker_init_fn, collate_fn=collate_fn,
                                                 multiprocessing_context=multiprocessing_context,
                                                 generator=generator,
                                                 prefetch_factor=prefetch_factor,
                                                 persistent_workers=persistent_workers
                                                 )
        return dataloader


class UniversalBatchSampler(object):

    def __init__(self, dataset_size, batch_size, probs=None, length=None, shuffle=True, tail=True,
                 once=False, expansion_size=int(1e7)):

        self.length = sys.maxsize if length is None else int(length)

        if check_type(dataset_size).major == 'array':
            self.indices = torch.LongTensor(dataset_size)
        else:
            self.indices = torch.arange(dataset_size)

        if probs is not None:

            probs = np.array(probs)
            probs = probs / probs.sum()

            grow_factor = max(expansion_size, len(probs)) / len(probs)

            probs = (probs * len(probs) * grow_factor).round().astype(np.int)
            m = np.gcd.reduce(probs)
            reps = probs // m
            indices = pd.DataFrame({'index': self.indices, 'times': reps})
            self.indices = torch.LongTensor(indices.loc[indices.index.repeat(indices['times'])]['index'].values)

        self.size = len(self.indices)

        if once:
            self.length = math.ceil(self.size / batch_size) if tail else self.size // batch_size

        self.once = once

        self.batch = batch_size
        self.minibatches = int(self.size / self.batch)

        self.shuffle = shuffle
        self.tail = tail

    def __iter__(self):

        self.n = 0
        indices = self.indices.clone()

        for _ in itertools.count():

            if self.shuffle:
                indices = indices[torch.randperm(len(indices))]

            indices_batched = indices[:self.minibatches * self.batch]
            indices_tail = indices[self.minibatches * self.batch:]

            if self.tail and not self.once:

                to_sample = max(0, self.batch - (self.size - self.minibatches * self.batch))

                fill_batch = np.random.choice(len(indices_batched), to_sample, replace=(to_sample > self.size))
                fill_batch = indices_batched[torch.LongTensor(fill_batch)]
                indices_tail = torch.cat([indices_tail, fill_batch])

                indices_batched = torch.cat([indices_batched, indices_tail])

            indices_batched = indices_batched.reshape((-1, self.batch))

            for samples in indices_batched:
                self.n += 1
                if self.n >= self.length:
                    yield samples
                    return
                else:
                    yield samples

            if self.once:
                if self.tail:
                    yield indices_tail
                return

    def __len__(self):
        return self.length


class HashSplit(object):

    def __init__(self, seed=None, granularity=.001, **argv):

        s = pd.Series(index=list(argv.keys()), data=list(argv.values()))
        s = s / s.sum() / granularity
        self.subsets = s.cumsum()
        self.n = int(1 / granularity)
        self.seed = seed

    def __call__(self, x):

        if type(x) is pd.Series:
            return x.apply(self._call)
        elif type(x) is list:
            return [self._call(xi) for xi in x]
        else:
            return self._call(x)

    def _call(self, x):

        x = f'{x}/{self.seed}'
        x = int(hashlib.sha1(x.encode('utf-8')).hexdigest(), 16) % self.n
        subset = self.subsets.index[x < self.subsets][0]

        return subset
