import itertools
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from .utils import check_type
import pandas as pd
import math

class UniversalDataset(torch.utils.data.Dataset):

    def __init__(self):
        super().__init__()
        self.indices_split = {}
        self.samplers = {}
        self.labels_split = {}

    def __len__(self):
        try:
            return len(self.data)
        except:
            raise NotImplementedError

    def split(self, validation=None, test=None, seed=5782, stratify=False, labels=None):
        '''
        validation, test can be lists of indices, relative size or absolute size

        The seed is fixed unless explicitly given, to make sure datasets are split the same in all instances.
        If one wish to obtain random splitting, seed should be None

        '''

        indices = np.arange(len(self))

        if test is None:
            pass
        elif check_type(test) in ['array', 'tensor']:
            self.indices_split['test'] = torch.LongTensor(test)
            indices = np.sort(np.array(list(set(indices).difference(set(np.array(test))))))
        else:
            if labels is not None:
                indices, test, labels, self.labels_split['test'] = train_test_split(indices, labels, random_state=seed,
                                                                                    test_size=test, stratify=labels if stratify else None)
            else:
                indices, test = train_test_split(indices, random_state=seed, test_size=test)

            self.indices_split['test'] = torch.LongTensor(test)
            if seed is not None:
                seed = seed + 1

        if validation is None:
            pass
        elif check_type(validation) in ['array', 'tensor']:
            self.indices_split['validation'] = torch.LongTensor(validation)
            indices = np.sort(np.array(list(set(indices).difference(set(validation)))))
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
            self.samplers['validation'] = UniversalBatchSampler(self.indices_split['validation'],
                                                          eval_batch_size, shuffle=True, tail=True, once=False)

        if 'train' in self.indices_split:
            probs = None
            if oversample and 'train' in self.labels_split and self.labels_split['train'] is not None:
                probs = compute_sample_weight('balanced', y=self.labels_split['train']) ** weight_factor

            self.samplers['train'] = UniversalBatchSampler(self.indices_split['train'],
                                                           batch_size, probs=probs, shuffle=True, tail=True,
                                                           once=False, expansion_size=expansion_size)

    def build_dataloaders(self, num_workers=0, pin_memory=True, timeout=0, collate_fn=None,
                   worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2):

        dataloaders = {}

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

        self.length = int(1e20) if length is None else int(length)

        if check_type(dataset_size) in ['array', 'tensor']:
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
