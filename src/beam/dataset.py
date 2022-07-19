import itertools
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from .utils import check_type, slice_to_index, as_tensor
import pandas as pd
import math
import hashlib
import sys
import warnings
from .data_tensor import DataTensor


class PackedFolds(object):

    def __init__(self, data, index=None, names=None, fold=None, fold_index=None, device=None,
                 sort_index=False, quick_getitem=True):

        self.quick_getitem = quick_getitem
        self.names_dict = None
        self.names = None
        self.sampling_method = 'folds'
        
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

        elif data_type.major == 'array':
            self.data = as_tensor(data, device=device)
            self.sampling_method = 'no_folds'
        else:
            raise ValueError("data should be either dict/list/array")

        fold_type = check_type(fold)

        if fold_type.element == 'str':
            fold, names_map = pd.factorize(fold)

            if self.names_dict is not None:
                assert all([i == self.names_dict[n] for i, n in enumerate(names_map)]), "fold and data maps must match"
            
            else:
                self.names = list(names_map)
                self.names_dict = {n: i for i, n in enumerate(self.names)}

            fold = as_tensor(fold)

            if self.sampling_method == 'no_folds':
                self.sampling_method = 'foldable'
            
        elif fold_type.element == 'int':   
            fold = as_tensor(fold)

            if self.sampling_method == 'no_folds':
                self.sampling_method = 'foldable'

        elif fold is None:
            if self.sampling_method == 'no_folds':
                assert len(names) == 1, "this is the single fold case"
                fold = torch.zeros(len(self.data), dtype=torch.int64)
                self.sampling_method = 'offset'
            else:
                fold = torch.cat([i * torch.ones(len(d), dtype=torch.int64) for i, d in enumerate(self.data)])

        else:
            raise ValueError

        if data_type.minor in ['dict', 'list']:
            lengths = torch.LongTensor([len(di) for di in self.data])
        else:
            lengths = torch.bincount(fold)
    
        if fold_index is not None:
            fold_index = as_tensor(fold_index)
        else:
            fold_index = torch.cat([torch.arange(l) for l in lengths])

        # update names
        if self.names is None:
            if data_type.minor in ['dict', 'list']:
                self.names = list(range(len(self.data)))
            else:
                self.names = torch.sort(torch.unique(fold)).values

            self.names_dict = {n: i for i, n in enumerate(self.names)}

        # merge data if possible for faster slicing
        if data_type.minor in ['dict', 'list']:
            dtype = [di.dtype for di in self.data]
            shape = [di.shape[1:] for di in self.data]
            if all([d == dtype[0] for d in dtype]) and all([s == shape[0] for s in shape]):
                self.data = torch.cat(self.data)
                self.sampling_method = 'offset'

        index_type = check_type(index)

        if index_type.minor == 'list':
            index = torch.concat([as_tensor(v, return_vector=True) for v in index])

        elif index_type.minor == 'dict':
            index = torch.concat([as_tensor(index[k], return_vector=True) for k in self.names])

        elif index_type.major == 'array':
            index = as_tensor(index)
            
        elif index is None:

            if self.sampling_method in ['offset', 'foldable']:
                index = torch.arange(len(self.data))
                self.sampling_method = 'index'
            else:
                index = torch.arange(sum([len(d) for d in self.data]))
            
        else:
            raise ValueError

        cumsum = torch.cumsum(lengths, dim=0)
        offset = cumsum - lengths
        offset = offset[fold] + fold_index

        self.device = device

        info = {'fold': fold, 'fold_index': fold_index, 'offset': offset}

        if self.sampling_method == 'index':
            index = None

        self.info = DataTensor(info, index=index, device=device)

        if sort_index:
            self.sort_index()

    def sort_index(self, ascending=True):

        self.info = self.info.sort_index(ascending=ascending)

        return self

    def __len__(self):
        return len(self.info)

    def get_fold(self, name):

        fold = self.names_dict[name]
        info = self.info[self.info['fold'] == fold]
        index = info.index

        if self.sampling_method == 'folds':
            data = self.data[fold]
        elif  self.sampling_method == 'index':
            data = self.data[index]
        elif self.sampling_method == 'offset':
            data = self.data[info['offset'].values]
        else:
            raise Exception(f"Sampling method unsupported: {self.sampling_method}")

        return PackedFolds(data=data, index=index, names=[name], device=self.device)

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
    def index(self):
        return self.info.index

    @property
    def fold(self):
        return self.info['fold'].values

    @property
    def fold_index(self):
        return self.info['fold_index'].values

    @property
    def offset(self):
        return self.info['offset'].values

    @property
    def shape(self):

        if self.sampling_method == 'fold':
            shape = {k: d.shape for k, d in zip(self.names, self.data)}
        else:
            shape = self.data.shape
        return shape

    @property
    def values(self):

        if self.sampling_method == 'fold':
            data = torch.cat(self.data)
        else:
            data = self.data

        return data[self.info['offset'].values]

    @property
    def tag(self):
        if len(self.names) == 1:
            return self.names[0]
        return 'hetrogenous'

    def to(self, device):

        self.data = [di.to(device) for di in self.data]
        self.info = self.info.to(device)
        self.device = device

    def __repr__(self):
        if self.sampling_method == 'folds':
            data = {k: self.data[self.names_dict[k]] for k in self.names}
        else:
            data = {k: self.get_fold(k).data for k in self.names}

        if len(data) == 1:
            data = next(iter(data.values()))

        return repr(data)

    def __getitem__(self, ind):

        ind_type = check_type(ind, check_minor=False)
        if ind_type.major == 'scalar' and ind_type.element == 'str':
            return self.get_fold(ind)
        if self.sampling_method == 'index' and self.quick_getitem:
            return self.data[ind]

        if type(ind) is tuple:
            ind_rest = ind[1:]
            ind = ind[0]
        else:
            ind_rest = tuple()

        ind = slice_to_index(ind, l=len(self), sliced=self.index)
        if ind_type.major == 'scalar':
            ind = [ind]

        info = self.info.loc[ind]

        if self.sampling_method == 'folds':
            fold, fold_index = info[['fold', 'fold_index']].values.T

            uq = torch.sort(torch.unique(fold)).values
            names = [self.names[i] for i in uq]

            if len(uq) == 1:

                data = self.data[uq[0]].__getitem__((fold_index, *ind_rest))
                if len(ind) == 1:
                    return data[0]

                return PackedFolds(data=[data], names=names, index=ind, device=self.device)

            fold_index = [fold_index[fold == i] for i in uq]
            data = [self.data[i].__getitem__((j, *ind_rest)) for i, j in zip(uq, fold_index)]

            fold = None

        else:

            fold, offset = info[['fold', 'offset']].values.T
            index = info.index
            ind = index if self.sampling_method == 'index' else offset

            data = self.data.__getitem__((ind, *ind_rest))

            uq = torch.sort(torch.unique(fold)).values
            names = [self.names[i] for i in uq]

            if self.sampling_method == 'index':
                ind = None
            else:
                ind = index

        return PackedFolds(data=data, names=names, index=ind, fold=fold, device=self.device)


class UniversalDataset(torch.utils.data.Dataset):

    def __init__(self, *args, index=None, device='cpu', **kwargs):
        super().__init__()

        self.index = None
        self.set_index(index)

        if not hasattr(self, 'indices_split'):
            self.indices_split = {}
        if not hasattr(self, 'samplers'):
            self.samplers = {}
        if not hasattr(self, 'labels_split'):
            self.labels_split = {}

        # The training label is to be used when one wants to apply some data transformations/augmentations
        # only in training mode
        self.training = False
        self.data_type = None
        self.statistics = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(args) == 1:
                d = args[0]
                if issubclass(type(d), dict):
                    self.data = {k: as_tensor(v, device=device) for k, v in d.items()}
                    self.data_type = 'dict'
                elif (type(d) is list) or (type(d) is tuple):
                    self.data = [as_tensor(v, device=device) for v in d]
                    self.data_type = 'list'
                else:
                    self.data = d
                    self.data_type = 'simple'
            elif len(args):
                self.data = [as_tensor(v, device=device) for v in args]
                self.data_type = 'list'
            elif len(kwargs):
                self.data = {k: as_tensor(v, device=device) for k, v in kwargs.items()}
                self.data_type = 'dict'
            else:
                self.data = None

    def set_index(self, index):

        self.index = None
        if index is not None:
            index_type = check_type(index)
            if index_type.minor == 'tensor':
                index = index.detach().cpu().numpy()
            index = pd.Series(data=np.arange(len(index)), index=index)
            # check if index is not a simple arange
            if np.abs(index.index.values - np.arange(len(index))).sum() > 0:
                self.index = index

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def getitem(self, ind):

        if self.data_type is None:
            self.data_type = check_type(self.data).minor

        if self.data_type == 'dict':
            return {k: v[ind] for k, v in self.data.items()}
        elif self.data_type == 'list':
            return [v[ind] for v in self.data]
        elif self.data_type == 'simple':
            return self.data[ind]
        else:
            return self.data[ind]

    def __getitem__(self, ind):

        if self.index is not None:

            ind = slice_to_index(ind, l=self.index.index.max()+1)

            ind_type = check_type(ind, check_element=False)
            if ind_type.minor == 'tensor':
                loc = ind.detach().cpu().numpy()
            else:
                loc = ind
                ind = as_tensor(ind)

            iloc = self.index.loc[loc].values

        else:

            ind = slice_to_index(ind, l=len(self))
            iloc = ind

        return ind, self.getitem(iloc)

    @property
    def device(self):

        if self.data_type is None:
            self.data_type = check_type(self.data).minor

        if self.data_type == 'dict':
            return next(iter(self.data.values())).device
        elif self.data_type == 'list':
            return self.data[0].device
        elif self.data_type == 'simple':
            return self.data.device
        elif hasattr(self.data, 'device'):
            return self.data.device
        else:
            raise NotImplementedError(f"For data type: {type(self.data)}")

    def __repr__(self):
        return repr(self.data)

    @property
    def values(self):
        return self.data

    def __len__(self):

        if self.data_type is None:
            self.data_type = check_type(self.data).minor

        if self.data_type == 'dict':
            return len(next(iter(self.data.values())))
        elif self.data_type == 'list':
            return len(self.data[0])
        elif self.data_type == 'simple':
            return len(self.data)
        elif hasattr(self.data, '__len__'):
            return len(self.data)
        else:
            raise NotImplementedError(f"For data type: {type(self.data)}")

    def split(self, validation=None, test=None, seed=5782, stratify=False, labels=None,
                    test_split_method='uniform', time_index=None, window=None):
        """
                partition the data into train/validation/split folds.
                Parameters
                ----------
                validation : float/int/array/tensor
                    If float, the ratio of the data to be used for validation. If int, should represent the total number of
                    validation samples. If array or tensor, the elements are the indices for the validation part of the data
                test :  float/int/array/tensor
                   If float, the ratio of the data to be used for test. If int, should represent the total number of
                   test samples. If array or tensor, the elements are the indices for the test part of the data
                seed : int
                    The random seed passed to sklearn's train_test_split function to ensure reproducibility. Passing seed=None
                    will produce randomized results.
                stratify: bool
                    If True, and labels is not None, partition the data such that the distribution of the labels in each part
                    is the same as the distribution of the labels in the whole dataset.
                labels: iterable
                    The corresponding ground truth for the examples in data
                """

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
                # labels = labels[indices]

        elif test_split_method == 'uniform':

            if labels is not None:
                labels_to_split = labels[indices]
                indices, test, _, self.labels_split['test'] = train_test_split(indices, labels_to_split,
                                                                               random_state=seed,
                                                                               test_size=test,
                                                                               stratify=labels_to_split if stratify else None)
            else:
                indices, test = train_test_split(indices, random_state=seed, test_size=test)

            self.indices_split['test'] = torch.LongTensor(test)
            if seed is not None:
                seed = seed + 1

        elif test_split_method == 'time_based':
            ind_sort = np.argsort(time_index)
            indices = indices[ind_sort]

            test_size = int(test * len(self)) if type(test) is float else test
            self.indices_split['test'] = torch.LongTensor(indices[-test_size:])
            indices = indices[:-test_size]

            if labels is not None:
                labels = labels[ind_sort]
                self.labels_split['test'] = labels[self.indices_split['test']]

        if validation is None:
            pass
        elif check_type(validation).major == 'array':
            self.indices_split['validation'] = torch.LongTensor(validation)
            indices = np.sort(np.array(list(set(indices).difference(set(np.array(validation))))))

            if labels is not None:
                self.labels_split['validation'] = labels[self.indices_split['validation']]

        else:
            if type(validation) is float:
                validation = len(self) / len(indices) * validation

            if labels is not None:

                labels_to_split = labels[indices]
                indices, validation, _, self.labels_split['validation'] = train_test_split(indices, labels_to_split, random_state=seed,
                                                                                                test_size=validation, stratify=labels_to_split if stratify else None)
            else:
                indices, validation = train_test_split(indices, random_state=seed, test_size=validation)

            self.indices_split['validation'] = torch.LongTensor(validation)

        self.indices_split['train'] = torch.LongTensor(indices)
        if labels is not None:
            self.labels_split['train'] = labels[indices]

    def set_statistics(self, stats):
        self.statistics = stats

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

        try:
            d = str(self.device)
            pin_memory_ = ('cpu' in str(d))
        except NotImplementedError:
            pin_memory_ = True

        if pin_memory is None:
            pin_memory = pin_memory_
        else:
            pin_memory = pin_memory and pin_memory_

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


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, alg, *args, **kwargs):
        super().__init__()

        if type(dataset) != UniversalDataset:
            dataset = UniversalDataset(dataset)

        self.dataset = dataset
        self.alg = alg
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, ind):

        ind_type = check_type(ind, check_element=False)
        if ind_type.major == 'scalar':
            ind = [ind]

        ind, data = self.dataset[ind]
        dataset = UniversalDataset(data)
        res = self.alg.predict(dataset, *self.args, **self.kwargs)
        # res.set_index(ind)

        return ind, res.values


class UniversalBatchSampler(object):
    """
         A class used to generate batches of indices, to be used in drawing samples from a dataset
         ...
         Attributes
         ----------
         indices : tensor
             The array of indices that can be sampled.
         length : int
               Maximum number of batches that can be returned by the sampler
         size : int
               The length of indices
         batch: int
               size of batch
         minibatches : int
             number of batches in one iteration over the array of indices
         once : bool
             If true, perform only 1 iteration over the indices array.
         tail : bool
             If true, run over the tail elements of indices array (the remainder left
             when dividing len(indices) by batch size). If once, return a minibatch. Else
             sample elements from the rest of the array to supplement the tail elements.
          shuffle : bool
             If true, shuffle the indices after each epoch
         """

    def __init__(self, dataset_size, batch_size, probs=None, length=None, shuffle=True, tail=True,
                 once=False, expansion_size=int(1e7)):

        """
               Parameters
               ----------
               dataset_size : array/tensor/int
                   If array or tensor, represents the indices of the examples contained in a subset of the whole data
                   (train/validation/test). If int, generates an array of indices [0, ..., dataset_size].
               batch_size : int
                   number of elements in a batch
               probs : array, optional
                   An array the length of indices, with probability/"importance" values to determine
                   how to perform oversampling (duplication of indices to change data distribution).
               length : int, optional
                  see descrtiption in class docstring
               shuffle : bool, optional
                  see description in class docstring
               tail : bool, optional
                  see description in class docstring
               once: bool, optional
                  see description in class docstring
               expansion_size : int
                    Limit on the length of indices (when oversampling, the final result can't be longer than
                    expansion_size).``
         """

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
