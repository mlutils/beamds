import argparse
import warnings

import numpy as np
import pandas as pd
import torch

from .sampler import UniversalBatchSampler
from ..utils import (recursive_batch, to_device, recursive_device, container_len, beam_device, as_tensor, check_type,
                     as_numpy, slice_to_index, DataBatch, lazy_property)

from ..path import beam_path
from ..data import BeamData


class UniversalDataset(torch.utils.data.Dataset):

    def __init__(self, *args, index=None, label=None, device=None, target_device=None, to_torch=True,
                 index_mapping='backward', **kwargs):
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

        if device is None:
            device = 'cpu'
        device = beam_device(device)
        target_device = beam_device(target_device)

        self.index = None
        self.set_index(index, mapping=index_mapping)

        if not hasattr(self, 'indices_split'):
            self.indices = {}
        if not hasattr(self, 'labels_split'):
            self.labels_split = {}
        if not hasattr(self, 'probs'):
            self.probs = {}

        # The training label is to be used when one wants to apply some data transformations/augmentations
        # only in training mode
        self.training = False
        self.statistics = None
        self._target_device = target_device
        self.to_torch = to_torch

        if len(args) >= 1 and isinstance(args[0], argparse.Namespace):
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

        self.label = as_tensor(label, device=self.device)

    @lazy_property
    def target_device(self):
        if self._target_device is not None:
            return self._target_device

        if hasattr(self, 'hparams'):
            if self.hparams.get('accelerate', False) and self.hparams.get('device_placement', False):
                return None
            if self.hparams.get('device', None) is not None and self.hparams.get('n_gpus', 1) <= 1:
                return beam_device(self.hparams.get('device', None))

        return None

    def set_index(self, index, mapping='backward'):

        self.index = None
        if index is not None:
            index_type = check_type(index)
            if index_type.minor == 'tensor':
                index = as_numpy(index)
            if mapping == 'backward':
                index = pd.Series(data=np.arange(len(index)), index=index)
                # check if index is not a simple arange
                if np.abs(index.index.values - np.arange(len(index))).sum() > 0:
                    self.index = index
            elif mapping == 'forward':
                index = pd.Series(data=index, index=np.arange(len(index)))
                self.index = index
            else:
                raise NotImplementedError(f"Mapping type: {mapping} not supported")


    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    @lazy_property
    def data_type(self):
        return check_type(self.data).minor

    def getitem(self, ind):

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

    @classmethod
    def get_subset(cls, self, subset):
        index = self.indices[subset]
        return cls(self.data, index=index, label=self.label, device=self.device,
                     target_device=self.target_device, to_torch=self.to_torch, index_mapping='forward')

    def __getitem__(self, ind):

        if type(ind) is str:
            return UniversalDataset.get_subset(self, ind)

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
        if self.to_torch:
            sample = as_tensor(sample, device=self.target_device)
        elif self.target_device is not None:
            sample = to_device(sample, device=self.target_device)

        label = None
        if self.label is not None:
            label = self.label[iloc]

        return DataBatch(index=ind, data=sample, label=label)

    @lazy_property
    def device(self):

        if self.data_type == 'dict':
            device = recursive_device(next(iter(self.data.values())))
        elif self.data_type == 'list':
            device = recursive_device(self.data[0])
        elif self.data_type == 'simple':
            device = self.data.device
        elif hasattr(self.data, 'device') and self.data.device is not None:
            device = self.data.device
        else:
            device = 'cpu'

        return beam_device(device)

    def __repr__(self):
        return repr(self.data)

    @property
    def values(self):
        return self.data

    # def save(self, path):
    #
    #     bd_path = beam_path(path).joinpath('beam_data')
    #     bd = BeamData(self.data, index=self.index, label=self.label, path=bd_path, device=self.device)
    #     bd.store()
    #
    #     state = self.state
    #     if has_beam_ds and isinstance(state, BeamData):
    #         state.store(path=path)
    #     else:
    #         path = beam_path(path)
    #         path.write(state)

    def __len__(self):

        if self.index is not None:
            return len(self.index)

        if self.data_type is None:
            self.data_type = check_type(self.data).minor

        if self.data_type == 'dict':
            return container_len(next(iter(self.data.values())))
        elif self.data_type == 'list':
            return container_len(self.data[0])
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

        from sklearn.model_selection import train_test_split

        if labels is None:
            labels = self.label
        if self.label is None:
            self.label = labels

        indices = np.arange(len(self))
        if time_index is None:
            time_index = indices

        if test is None:
            pass
        elif check_type(test).major == 'array':
            self.indices['test'] = as_tensor(test, dtype=torch.long)
            indices = np.sort(list(set(indices).difference(set(as_numpy(test)))))

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

            self.indices['test'] = as_tensor(test, dtype=torch.long)
            if seed is not None:
                seed = seed + 1

        elif test_split_method == 'time_based':
            ind_sort = np.argsort(time_index)
            indices = indices[ind_sort]

            test_size = int(test * len(self)) if type(test) is float else test
            self.indices['test'] = as_tensor(indices[-test_size:], dtype=torch.long)
            indices = indices[:-test_size]

            if labels is not None:
                labels = labels[ind_sort]
                self.labels_split['test'] = labels[self.indices['test']]

        if validation is None:
            pass
        elif check_type(validation).major == 'array':
            self.indices['validation'] = as_tensor(validation, dtype=torch.long)
            indices = np.sort(list(set(indices).difference(set(as_numpy(validation)))))

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

            self.indices['validation'] = as_tensor(validation, dtype=torch.long)

        self.indices['train'] = as_tensor(indices, dtype=torch.long)
        if labels is not None:
            self.labels_split['train'] = labels[indices]

    def set_statistics(self, stats):
        self.statistics = stats

    def build_sampler(self, batch_size, subset=None, persistent=True, oversample=False, weight_factor=1., expansion_size=int(1e7),
                       dynamic=False, buffer_size=None, probs_normalization='sum', tail=True, sample_size=100000):

        from sklearn.utils.class_weight import compute_sample_weight

        if subset is None:
            if self.index is not None:
                indices = self.index.index.values
            else:
                indices = torch.arange(len(self))
        else:
            indices = self.indices[subset]

        if not persistent:
            return UniversalBatchSampler(indices, batch_size, shuffle=False,
                                         tail=tail, once=True, dynamic=False)

        probs = None
        if oversample and subset in self.labels_split and self.labels_split[subset] is not None:
            probs = compute_sample_weight('balanced', y=self.labels_split[subset]) ** weight_factor
            probs_normalization = 'sum'
        elif subset is None and check_type(self.probs).major == 'array':
            probs = self.probs
        elif subset in self.probs:
            probs = self.probs[subset]

        return UniversalBatchSampler(indices,
                                     batch_size, probs=probs, shuffle=True, tail=tail,
                                     once=False, expansion_size=expansion_size,
                                     dynamic=dynamic, buffer_size=buffer_size,
                                     probs_normalization=probs_normalization,
                                     sample_size=sample_size, device=self.device)

    def build_dataloader(self, sampler, num_workers=0, pin_memory=None, timeout=0, collate_fn=None,
                   worker_init_fn=None, multiprocessing_context=None, generator=None, prefetch_factor=2):

        kwargs = {}
        if num_workers > 0:
            kwargs['prefetch_factor'] = prefetch_factor

        try:
            d = self.device.type if self.target_device is None else self.target_device
            pin_memory_ = ('cpu' == d)
        except NotImplementedError:
            pin_memory_ = True

        if pin_memory is None:
            pin_memory = pin_memory_
        else:
            pin_memory = pin_memory and pin_memory_

        persistent_workers = (num_workers > 0 and not sampler.once)

        return torch.utils.data.DataLoader(self, sampler=sampler, batch_size=None,
                                             num_workers=num_workers, pin_memory=pin_memory, timeout=timeout,
                                             worker_init_fn=worker_init_fn, collate_fn=collate_fn,
                                             multiprocessing_context=multiprocessing_context, generator=generator,
                                             persistent_workers=persistent_workers, **kwargs)
