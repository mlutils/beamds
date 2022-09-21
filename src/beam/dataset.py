import itertools
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from .utils import check_type, slice_to_index, as_tensor, to_device, recursive_batch, as_numpy, beam_device, \
    recursive_device, recursive_len
import pandas as pd
import math
import hashlib
import sys
import warnings
from .data_tensor import DataTensor
import argparse
from collections import namedtuple

DataBatch = namedtuple("DataBatch", "index data")


class UniversalDataset(torch.utils.data.Dataset):

    def __init__(self, *args, index=None, device='cpu', target_device=None, **kwargs):
        """
        Universal Beam dataset class

        @param args:
        @param index:
        @param device:
        @param target_device: if not None, the dataset is responsible to transform samples into this dataset.
        This is useful when we want to transform a sample to the GPU during the getitem routine in order to speed-up the
        computation.
        @param kwargs:
        """
        super().__init__()

        device = beam_device(device)
        target_device = beam_device(target_device)

        self.index = None
        self.set_index(index)

        if not hasattr(self, 'indices_split'):
            self.indices = {}
        if not hasattr(self, 'labels_split'):
            self.labels_split = {}
        if not hasattr(self, 'probs'):
            self.probs = {}

        # The training label is to be used when one wants to apply some data transformations/augmentations
        # only in training mode
        self.training = False
        self.data_type = None
        self.statistics = None
        self.target_device = target_device

        if len(args) >= 1 and type(args[0]) is argparse.Namespace:
            self.hparams = args[0]
            args = args[1:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(args) == 1:
                d = args[0]
                if isinstance(d, dict):
                    self.data = {k: as_tensor(v, device=device) for k, v in d.items()}
                    self.data_type = 'dict'
                elif isinstance(d, list) or isinstance(d, tuple):
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
                index = as_numpy(index)
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

            ind_type = check_type(ind, check_minor=False)
            if ind_type.element == 'str':
                if ind_type.major == 'scalar':
                    return self.data[ind]
                return [self.data[k] for k in ind]

            return {k: recursive_batch(v, ind) for k, v in self.data.items()}

        elif self.data_type == 'list':
            return [recursive_batch(v, ind) for v in self.data]
        elif self.data_type == 'simple':
            return self.data[ind]
        else:
            return self.data[ind]

    def __getitem__(self, ind):

        if self.index is not None:

            ind = slice_to_index(ind, l=self.index.index.max()+1)

            ind_type = check_type(ind, check_element=False)
            if ind_type.minor == 'tensor':
                loc = as_numpy(ind)
            else:
                loc = ind
                ind = as_tensor(ind)

            if ind_type.major == 'scalar':
                loc = [loc]

            iloc = self.index.loc[loc].values

        else:

            ind = slice_to_index(ind, l=len(self))
            iloc = ind

        sample = self.getitem(iloc)
        if self.target_device is not None:
            sample = to_device(sample, device=self.target_device)

        return DataBatch(index=ind, data=sample)

    def __device__(self):
        raise NotImplementedError(f"For data type: {type(self.data)}")

    @property
    def device(self):

        if self.data_type is None:
            self.data_type = check_type(self.data).minor

        if self.data_type == 'dict':
            return recursive_device(next(iter(self.data.values())))
        elif self.data_type == 'list':
            return recursive_device(self.data[0])
        elif self.data_type == 'simple':
            return self.data.device
        elif hasattr(self.data, 'device'):
            return self.data.device
        else:
            return self.__device__()

    def __repr__(self):
        return repr(self.data)

    @property
    def values(self):
        return self.data

    def __len__(self):

        if self.data_type is None:
            self.data_type = check_type(self.data).minor

        if self.data_type == 'dict':
            return recursive_len(next(iter(self.data.values())))
        elif self.data_type == 'list':
            return recursive_len(self.data[0])
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
            self.indices['test'] = torch.LongTensor(test)
            indices = np.sort(np.array(list(set(indices).difference(set(np.array(test))))))

            if labels is not None:
                self.labels_split['test'] = labels[self.indices['test']]
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

            self.indices['test'] = torch.LongTensor(test)
            if seed is not None:
                seed = seed + 1

        elif test_split_method == 'time_based':
            ind_sort = np.argsort(time_index)
            indices = indices[ind_sort]

            test_size = int(test * len(self)) if type(test) is float else test
            self.indices['test'] = torch.LongTensor(indices[-test_size:])
            indices = indices[:-test_size]

            if labels is not None:
                labels = labels[ind_sort]
                self.labels_split['test'] = labels[self.indices['test']]

        if validation is None:
            pass
        elif check_type(validation).major == 'array':
            self.indices['validation'] = torch.LongTensor(validation)
            indices = np.sort(np.array(list(set(indices).difference(set(np.array(validation))))))

            if labels is not None:
                self.labels_split['validation'] = labels[self.indices['validation']]

        else:
            if type(validation) is float:
                validation = len(self) / len(indices) * validation

            if labels is not None:

                labels_to_split = labels[indices]
                indices, validation, _, self.labels_split['validation'] = train_test_split(indices, labels_to_split, random_state=seed,
                                                                                                test_size=validation, stratify=labels_to_split if stratify else None)
            else:
                indices, validation = train_test_split(indices, random_state=seed, test_size=validation)

            self.indices['validation'] = torch.LongTensor(validation)

        self.indices['train'] = torch.LongTensor(indices)
        if labels is not None:
            self.labels_split['train'] = labels[indices]

    def set_statistics(self, stats):
        self.statistics = stats

    def build_sampler(self, batch_size, subset=None, persistent=True, oversample=False, weight_factor=1., expansion_size=int(1e7),
                       dynamic=False, buffer_size=None, probs_normalization='sum', sample_size=100000):

        if subset is None:
            if self.index is not None:
                indices = self.index.index.values
            else:
                indices = torch.arange(len(self))
        else:
            indices = self.indices[subset]

        if not persistent:
            return UniversalBatchSampler(indices, batch_size, shuffle=False,
                                         tail=True, once=True, dynamic=False)

        probs = None
        if oversample and subset in self.labels_split and self.labels_split[subset] is not None:
            probs = compute_sample_weight('balanced', y=self.labels_split[subset]) ** weight_factor
            probs_normalization = 'sum'
        elif subset in self.probs:
            probs = self.probs[subset]

        return UniversalBatchSampler(self.indices[subset],
                                     batch_size, probs=probs, shuffle=True, tail=True,
                                     once=False, expansion_size=expansion_size,
                                     dynamic=dynamic, buffer_size=buffer_size,
                                     probs_normalization=probs_normalization,
                                     sample_size=sample_size)

    def build_dataloader(self, sampler, num_workers=0, pin_memory=None, timeout=0, collate_fn=None,
                   worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2):

        try:
            d = self.device.type if self.target_device is None else self.target_device
            pin_memory_ = ('cpu' == d)
        except NotImplementedError:
            pin_memory_ = True

        if pin_memory is None:
            pin_memory = pin_memory_
        else:
            pin_memory = pin_memory and pin_memory_

        persistent_workers = (num_workers > 0 and sampler.once)
        return torch.utils.data.DataLoader(self, sampler=sampler, batch_size = None,
                                             num_workers=num_workers, pin_memory=pin_memory, timeout=timeout,
                                             worker_init_fn=worker_init_fn, collate_fn=collate_fn,
                                             multiprocessing_context=multiprocessing_context, generator=generator,
                                             prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)


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

        return ind, res.values


class LazyReplayBuffer(UniversalDataset):

    def __init__(self, size, device='cpu'):
        self.max_size = size
        self.size = 0
        self.ptr = 0
        self.target_device = device
        super().__init__(device=device)

    def build_buffer(self, x):
        x_type = check_type(x)
        if x_type.element == 'int':
            dtype = torch.int64
        elif x_type.element == 'float16':
            dtype = torch.float16
        else:
            dtype = torch.float32

        return torch.zeros(self.size, *x.shape, device=self.target_device, dtype=dtype)

    def store(self, *args, **kwargs):

        if len(args) == 1:
            d = args[0]
        elif len(args):
            d = args
        else:
            d = kwargs

        if self.data is None:
            if isinstance(d, dict):
                self.data = {k: self.build_buffer(v) for k, v in d.items()}
                self.data_type = 'dict'
            elif isinstance(d, list) or isinstance(d, tuple):
                self.data = [self.build_buffer(v) for v in d]
                self.data_type = 'list'
            else:
                self.data = self.build_buffer(d)
                self.data_type = 'simple'

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.data_type == 'dict':
                for k, v in d.items():
                    self.data[k][self.ptr] = as_tensor(v, device=self.target_device)
            elif self.data_type == 'list':
                for i, v in enumerate(self.data):
                    self.data[i][self.ptr] = as_tensor(v, device=self.target_device)
            else:
                self.data[self.ptr] = as_tensor(d, device=self.target_device)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def reset(self):
        self.ptr = 0
        self.data = None
        self.size = 0

    def __len__(self):
        return self.size


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
         batch_size: int
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

    def __init__(self, indices, batch_size, probs=None, length=None, shuffle=True, tail=True,
                 once=False, expansion_size=int(1e7), dynamic=False, buffer_size=None,
                 probs_normalization='sum', sample_size=100000):

        """
               Parameters
               ----------
               indices : array/tensor/int
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
        self.once = once
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tail = tail
        self.probs_normalization = probs_normalization
        self.buffer_size = buffer_size
        self.refreshed = False
        self.size = None
        self.minibatches = None
        self.sample_size = sample_size

        if check_type(indices).major == 'array':
            self.indices = torch.LongTensor(indices)
        else:
            self.indices = torch.arange(indices)
        self.probs = np.array(probs) if probs is not None else None

        if dynamic:
            self.samples_iterator = self.dynamic_samples_iterator
            self.indices = np.array(self.indices)

        else:
            self.samples_iterator = self.static_samples_iterator
            if probs is not None:

                probs = np.array(self.normalize_probabilities(probs))
                grow_factor = max(expansion_size, len(probs)) / len(probs)

                probs = (probs * len(probs) * grow_factor).round().astype(np.int)
                m = np.gcd.reduce(probs)
                reps = probs // m
                indices = pd.DataFrame({'index': self.indices, 'times': reps})
                self.indices = torch.LongTensor(indices.loc[indices.index.repeat(indices['times'])]['index'].values)

        self.size = len(self.indices)
        self.minibatches = int(self.size / self.batch_size)
        if once:
            self.length = math.ceil(self.size / batch_size) if tail else self.size // batch_size

    def normalize_probabilities(self, p):

        if p is None:
            return None

        if self.probs_normalization == 'softmax':
            return torch.softmax(as_tensor(p, device='cpu'), dim=0)

        return p / p.sum()

    def update_fifo(self):
        if self.buffer_size is not None:
            self.indices = self.indices[-self.buffer_size:]
            self.probs = self.probs[-self.buffer_size:]
            self.unnormalized_probs = self.unnormalized_probs[-self.buffer_size:]

    def dynamic_samples_iterator(self):

        self.n = 0
        for _ in itertools.count():

            self.update_fifo()
            probs = np.array(self.normalize_probabilities(self.probs))
            size = min(self.size, self.sample_size) if self.sample_size is not None else self.size
            minibatches = math.ceil(size / self.batch_size)
            indices_batched = torch.LongTensor(np.random.choice(self.indices, size=(minibatches, self.batch_size),
                                                        replace=True, p=probs))

            for samples in indices_batched:
                self.n += 1
                yield samples
                if self.n >= self.length:
                    return
                if self.refreshed:
                    self.refreshed = False
                    continue

    def replace_indices(self, indices, probs=None):
        if check_type(indices).major == 'array':
            self.indices = np.array(indices)
        else:
            self.indices = np.arange(indices)
        self.probs = np.array(probs) if probs is not None else None
        self.refreshed = True

    def append_indices(self, indices, probs=None):
        self.indices = np.concatenate([self.indices, np.array(indices)])
        if probs is not None:
            self.probs = torch.cat([self.probs, as_tensor(probs, device='cpu')])

    def append_index(self, index, prob=None):
        self.indices = np.concatenate([self.indices, np.array([index])])
        if prob is not None:
            self.probs = torch.cat([self.probs, as_tensor([prob], device='cpu')])

    def pop_index(self, index):
        v = self.indices != index
        self.indices = self.indices[v]
        if self.probs is not None:
            self.probs = self.probs[torch.BoolTensor(v)]

    def pop_indices(self, indices):
        v = ~np.isin(self.indices, np.array(indices))
        self.indices = self.indices[v]
        if self.probs is not None:
            self.probs = self.probs[v]

    def static_samples_iterator(self):

        self.n = 0
        indices = self.indices.clone()

        for _ in itertools.count():

            if self.shuffle:
                indices = indices[torch.randperm(len(indices))]

            indices_batched = indices[:self.minibatches * self.batch_size]
            indices_tail = indices[self.minibatches * self.batch_size:]

            if self.tail and not self.once:

                to_sample = max(0, self.batch_size - (self.size - self.minibatches * self.batch_size))

                try:
                    fill_batch = np.random.choice(len(indices_batched), to_sample, replace=(to_sample > self.size))
                except ValueError:
                    raise ValueError("Looks like your dataset is smaller than a single batch. Try to make it larger.")

                fill_batch = indices_batched[torch.LongTensor(fill_batch)]
                indices_tail = torch.cat([indices_tail, fill_batch])

                indices_batched = torch.cat([indices_batched, indices_tail])

            indices_batched = indices_batched.reshape((-1, self.batch_size))

            for samples in indices_batched:
                self.n += 1
                yield samples
                if self.n >= self.length:
                    return

            if self.once:
                if self.tail:
                    yield indices_tail
                return

    def __iter__(self):
        return self.samples_iterator()

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
