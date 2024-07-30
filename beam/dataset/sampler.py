import numpy as np
import torch
import sys
import itertools
import pandas as pd
import math
import hashlib

from ..logging import beam_logger as logger
from ..utils import as_tensor, check_type, as_numpy, beam_device
from ..type import Types


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
                 probs_normalization='sum', sample_size=100000, device=None):

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

        indices_type = check_type(indices)
        if indices_type.minor == Types.tensor:
            device = device if device is not None else indices.device
        else:
            device = device if device is not None else 'cpu'\

        self.device = beam_device(device)

        if indices_type.major == Types.array:
            self.indices = as_tensor(indices, device=device, dtype=torch.int64)
        else:
            self.indices = torch.arange(indices, device=device)
        self.probs = as_numpy(probs) if probs is not None else None

        if dynamic:
            self.samples_iterator = self.dynamic_samples_iterator
            self.indices = as_numpy(self.indices)

        else:
            self.samples_iterator = self.static_samples_iterator
            if probs is not None:

                logger.info("UniversalBatchSampler: Building expanded indices array based on given probabilities")
                probs = as_numpy(self.normalize_probabilities(probs))
                grow_factor = max(expansion_size, len(probs)) / len(probs)

                probs = (probs * len(probs) * grow_factor).round().astype(np.int)
                m = np.gcd.reduce(probs)
                reps = np.clip(np.round(probs / m).astype(np.int), 1, None)

                logger.info(f"Expansion size: {expansion_size}, before expansion: {len(probs)}, "
                            f"after expansion: {np.sum(reps)}")
                indices = pd.DataFrame({'index': as_numpy(self.indices), 'times': reps})
                self.indices = as_tensor(indices.loc[indices.index.repeat(indices['times'])]['index'].values,
                                         device=self.device, dtype=torch.int64)

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
            probs = as_numpy(self.normalize_probabilities(self.probs))
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
        if check_type(indices).major == Types.array:
            self.indices = as_numpy(indices)
        else:
            self.indices = np.arange(indices)
        self.probs = as_numpy(probs) if probs is not None else None
        self.refreshed = True

    def append_indices(self, indices, probs=None):
        self.indices = np.concatenate([self.indices, as_numpy(indices)])
        if probs is not None:
            self.probs = torch.cat([self.probs, as_tensor(probs, device='cpu')])

    def append_index(self, index, prob=None):
        self.indices = np.concatenate([self.indices, as_numpy([index])])
        if prob is not None:
            self.probs = torch.cat([self.probs, as_tensor([prob], device='cpu')])

    def pop_index(self, index):
        v = self.indices != index
        self.indices = self.indices[v]
        if self.probs is not None:
            self.probs = self.probs[torch.BoolTensor(v)]

    def pop_indices(self, indices):
        v = ~np.isin(self.indices, as_numpy(indices))
        self.indices = self.indices[v]
        if self.probs is not None:
            self.probs = self.probs[v]

    def static_samples_iterator(self):

        self.n = 0
        indices = self.indices.clone()

        for _ in itertools.count():

            if self.shuffle:
                indices = indices[torch.randperm(len(indices), device=self.device)]

            indices_batched = indices[:self.minibatches * self.batch_size]
            indices_tail = indices[self.minibatches * self.batch_size:]

            if self.tail and not self.once:

                to_sample = max(0, self.batch_size - (self.size - self.minibatches * self.batch_size))

                try:
                    fill_batch = np.random.choice(len(indices_batched), to_sample, replace=(to_sample > self.size))
                except ValueError:
                    raise ValueError("Looks like your dataset is smaller than a single batch. Try to make it larger.")

                fill_batch = indices_batched[as_tensor(fill_batch, device=self.device)]
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
