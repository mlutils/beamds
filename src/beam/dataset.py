import itertools
import numpy as np
import torch


class UniversalDataset(torch.utils.data.Dataset):

    def __init__(self):
        super().__init__()

    def dataloader(self, batch_size, length=None, shuffle=True, tail=True, once=False,
                   num_workers=0, pin_memory=False, timeout=0, collate_fn=None,
                   worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2, persistent_workers=False):

        sampler = UniversalBatchSampler(len(self), batch_size, length=length, shuffle=shuffle, tail=tail, once=once)
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

        if hasattr(dataset_size, '__len__') and len(dataset_size) > 1:
            self.size = len(dataset_size)
            self.indices = dataset_size
        else:
            self.size = dataset_size
            self.indices = np.arange(self.size)

        self.replace = False if probs is None else True
        self.probs = probs

        self.batch = batch_size
        self.minibatches = int(self.size / self.batch)
        self.n = 0
        self.shuffle = shuffle
        self.tail = tail
        self.once = once

    def __iter__(self):

        for _ in itertools.count():

            if self.shuffle:
                shuffle_indices = np.random.choice(self.indices, self.size, replace=self.replace)
            else:
                shuffle_indices = self.indices

            if self.tail and not self.shuffle:

                self.n += 1
                i = 0
                while i < len(shuffle_indices):
                    yield shuffle_indices[i:i + self.batch]
                    i += self.batch

                if self.n >= self.length:
                    return

            else:
                shuffle_indices_minibatches = shuffle_indices[:self.minibatches * self.batch]
                shuffle_indices_tail = shuffle_indices[self.minibatches * self.batch:]

                to_sample = max(0, self.batch - (self.size - self.minibatches * self.batch))
                p = np.ones(self.size)
                p[shuffle_indices_tail] = 0
                p = p / np.sum(p)

                shuffle_indices = shuffle_indices_minibatches

                if not self.once:

                    shuffle_indices_tail = np.concatenate([shuffle_indices_tail,
                                                           np.random.choice(self.size, to_sample,
                                                                            replace=(to_sample > self.size), p=p)])
                    if self.tail:
                        shuffle_indices = np.concatenate([shuffle_indices_minibatches, shuffle_indices_tail])

                shuffle_indices = shuffle_indices.reshape((-1, self.batch))

                for samples in shuffle_indices:
                    self.n += 1
                    if self.n >= self.length:
                        self.n = 0
                        yield samples
                        return
                    else:
                        yield samples

                if self.once:
                    if self.tail:
                        yield shuffle_indices_tail
                    return

    def __len__(self):
        return self.length
