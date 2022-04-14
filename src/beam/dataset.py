import itertools
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from .utils import check_type
import pandas as pd
import math

class UniversalDataset(torch.utils.data.Dataset):

    def __init__(self, indices, validation=None, test=None, seed=None,):
        super().__init__()

        '''
        validation, test can be lists of indices, relative size or absolute size
        '''

        if check_type(indices) != 'array':
            indices = torch.arange(indices)

        self.indices = {}

        if check_type(test) == 'array':
            self.indices['test'] = np.array(test)
            indices = np.sort(np.array(list(set(indices).difference(set(self.indices['test'])))))
        else:
            indices, self.indices['test'] = train_test_split(indices, random_state=seed, test_size=test)
            if seed is not None:
                seed = seed + 1

        if validation is None:
            pass
        elif check_type(validation) == 'array':
            self.indices['test'] = np.array(index)
            indices = np.sort(np.array(list(set(indices).difference(set(self.indices['test'])))))
        else:
            if type(validation) is float:
                validation = len(data) / len(indices) * validation

            indices, self.indices['validation'] = train_test_split(indices, random_state=seed, test_size=validation)

        self.indices['train'] = indices

    def dataloader(self, batch_size, subset='train', length=None, shuffle=True, tail=True, once=False,
                   num_workers=0, pin_memory=False, timeout=0, collate_fn=None,
                   worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2,
                   persistent_workers=False):

        indices = self.indices[subset]

        sampler = UniversalBatchSampler(indices, batch_size, length=length, shuffle=shuffle, tail=tail, once=once)
        dataloader = torch.utils.data.DataLoader(self, sampler=sampler, batch_size=None,
                                                 num_workers=num_workers, pin_memory=pin_memory, timeout=timeout,
                                                 worker_init_fn=worker_init_fn, collate_fn=collate_fn,
                                                 multiprocessing_context=multiprocessing_context, generator=generator,
                                                 prefetch_factor=prefetch_factor, persistent_workers=persistent_workers
                                                 )
        return dataloader


class UniversalBatchSampler(object):

    def __init__(self, dataset_size, batch_size, probs=None, length=None, shuffle=True, tail=True, once=False):

        self.length = int(1e20) if length is None else int(length)

        if check_type(dataset_size) == 'array':
            self.indices = dataset_size
        else:
            self.indices = np.arange(dataset_size)

        if probs is not None:
            probs = probs / probs.sum()

            # TODO: Start here
            probs = (probs * len(probs) * 60).round().long()
            m = np.gcd.reduce(probs)
            reps = probs // m
            indices = pd.DataFrame({'index': self.indices, 'times': reps})
            self.indices = indices.loc[indices.index.repeat(x['times'])]['index'].values

        self.size = len(self.indices)

        if once:
            self.length = math.ceil(self.size / batch_size) if tail else self.size // batch_size

        self.once = once

        self.batch = batch_size
        self.minibatches = int(self.size / self.batch)

        self.shuffle = shuffle
        self.tail = tail

        print(self.indices)

    def __iter__(self):

        self.n = 0
        indices = self.indices.copy()

        for _ in itertools.count():

            if self.shuffle:
                indices = indices[np.random.permutation(len(indices))]

            indices_batched = indices[:self.minibatches * self.batch]
            indices_tail = indices[self.minibatches * self.batch:]

            if self.tail and not self.once:

                to_sample = max(0, self.batch - (self.size - self.minibatches * self.batch))

                indices_tail = np.concatenate([indices_tail, np.random.choice(indices_batched, to_sample,
                                                                              replace=(to_sample > self.size))])

                indices_batched = np.concatenate([indices_batched, indices_tail])

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
