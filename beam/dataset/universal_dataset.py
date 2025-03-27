import argparse
import warnings
import numpy as np
import pandas as pd
import torch

from ..type import Types
from ..utils import cached_property, slice_array
from ..path import beam_path
from ..data import BeamData
from ..base import BeamBase

from .sampler import UniversalBatchSampler
from ..utils import (recursive_batch, to_device, recursive_device, container_len, beam_device, as_tensor, check_type,
                     as_numpy, slice_to_index, DataBatch)


class UniversalDataset(torch.utils.data.Dataset, BeamBase):

    def __init__(self, *args, index=None, label=None, device=None, target_device=None, to_torch=True,
                 index_mapping='backward', preprocess=True, hparams=None, **kwargs):
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
        torch.utils.data.Dataset.__init__(self)
        BeamBase.__init__(self, *args, hparams=hparams, device=device, target_device=target_device, to_torch=to_torch,
                          index_mapping=index_mapping, preprocess=preprocess, **kwargs)

        device = beam_device(self.hparams.device)

        self.index = None
        self.reversed_index = None
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
        self.preprocess = self.hparams.preprocess
        self.statistics = None
        self._target_device = beam_device(self.hparams.target_device)
        self.to_torch = self.hparams.to_torch

        if len(args) >= 1 and isinstance(args[0], argparse.Namespace):
            self.hparams = args[0]
            args = args[1:]

        self._data_type = None
        self._device = None
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('_')}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(args) == 1:
                d = args[0]
                if isinstance(d, dict):
                    self.data = {k: self.as_something(v, device=device) for k, v in d.items()}
                elif isinstance(d, list) or isinstance(d, tuple):
                    self.data = [self.as_something(v, device=device) for v in d]
                else:
                    self.data = d
            elif len(args):
                self.data = [self.as_something(v, device=device) for v in args]
            elif len(kwargs):
                self.data = {k: self.as_something(v, device=device) for k, v in kwargs.items()}
            else:
                self.data = None

        self.label = self.as_something(label, device=self.device)

    def as_something(self, x, device=None, dtype=None):
        if self.to_torch:
            return as_tensor(x, device=device, dtype=dtype)
        elif self.preprocess:
            return as_numpy(x)
        return x

    @cached_property
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
            if index_type.element != Types.int:
                mapping = 'forward'
            if index_type.minor == Types.tensor:
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

            r = pd.Series(index)
            self.reversed_index = pd.Series(data=r.index, index=r.values)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    @property
    def data_type(self):
        if self._data_type is None or self._data_type.major == Types.none:
            self._data_type = check_type(self.data)
        return self._data_type

    def getitem(self, ind):

        if self.data_type.minor == Types.dict:

            ind_type = check_type(ind, minor=False)
            if ind_type.element == Types.str:
                if ind_type.major == Types.scalar:
                    return self.data[ind]
                return [self.data[k] for k in ind]

            return {k: recursive_batch(v, ind) for k, v in self.data.items()}

        elif self.data_type.minor == Types.list:
            return [recursive_batch(v, ind) for v in self.data]

        return slice_array(self.data, ind, x_type=self.data_type)

    def get_subset(self, subset):
        index = self.indices[subset]
        data = recursive_batch(self.data, index)
        label = recursive_batch(self.label, index) if self.label is not None else None

        return UniversalDataset(data, label=label, device=self.device,
                                target_device=self.target_device, to_torch=self.to_torch, index_mapping='forward')

    def __getitem__(self, ind):

        if type(ind) is str:
            return UniversalDataset.get_subset(self, ind)

        if self.index is not None:
            ind = slice_to_index(ind, l=self.index.index.max()+1)

            ind_type = check_type(ind, element=False)
            if ind_type.minor == Types.tensor:
                loc = as_numpy(ind)
            else:
                loc = ind
                ind = as_tensor(ind)

            if ind_type.major == Types.scalar:
                loc = [loc]

            iloc = self.index.loc[loc].values

        else:

            ind = slice_to_index(ind, l=len(self))
            iloc = ind

        if type(iloc) is int:
            iloc = [iloc]
        sample = self.getitem(iloc)
        if self.to_torch:
            sample = as_tensor(sample, device=self.target_device)
        elif self.target_device is not None:
            sample = to_device(sample, device=self.target_device)

        label = None
        if self.label is not None:
            label = self.label[iloc]

        return DataBatch(index=ind, data=sample, label=label)

    @property
    def device(self):

        if self._device is None:
            if self.data_type.minor == Types.dict:
                device = recursive_device(next(iter(self.data.values())))
            elif self.data_type.minor == Types.list:
                device = recursive_device(self.data[0])
            elif hasattr(self.data, 'device') and self.data.device is not None:
                device = self.data.device
            else:
                device = None
            self._device = beam_device(device)

        return self._device

    def __repr__(self):
        return repr(self.data)

    @property
    def values(self):
        return self.data

    def save(self, path):

        bd_path = beam_path(path)
        bd = BeamData(self.data, index=self.index, label=self.label, path=bd_path, device=self.device)
        bd.store()

    def __len__(self):

        if self.index is not None:
            return len(self.index)

        if self.data_type.minor == Types.dict:
            return container_len(next(iter(self.data.values())))
        elif self.data_type.minor == Types.list:
            return container_len(self.data[0])
        elif self.data_type.is_data_array:
            return len(self.data)
        elif hasattr(self.data, '__len__'):
            return len(self.data)
        else:
            raise NotImplementedError(f"For data type: {type(self.data)}")

    def split(self, labels=None, validation=None, test=None, seed=None, stratify=None,
                    test_split_method=None, time_index=None, window=None):
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

        if validation is None:
            validation = self.get_hparam('validation_size', None)
        if test is None:
            test = self.get_hparam('test_size', None)
        if seed is None:
            seed = self.get_hparam('split_dataset_seed', 5782)
        if stratify is None:
            stratify = self.get_hparam('stratify_dataset', False)
        if test_split_method is None:
            test_split_method = self.get_hparam('test_split_method', 'uniform')
        if time_index is None:
            time_index = self.get_hparam('time_index', None)

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
        elif check_type(test).major == Types.array:
            self.indices['test'] = self.as_something(test, dtype=torch.long)
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

            self.indices['test'] = self.as_something(test, dtype=torch.long)
            if seed is not None:
                seed = seed + 1

        elif test_split_method == 'time_based':
            ind_sort = np.argsort(time_index)
            indices = indices[ind_sort]

            test_size = int(test * len(self)) if type(test) is float else test
            self.indices['test'] = self.as_something(indices[-test_size:], dtype=torch.long)
            indices = indices[:-test_size]

            if labels is not None:
                labels = labels[ind_sort]
                self.labels_split['test'] = labels[self.indices['test']]

        if validation is None:
            pass
        elif check_type(validation).major == Types.array:
            self.indices['validation'] = self.as_something(validation, dtype=torch.long)
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

            self.indices['validation'] = self.as_something(validation, dtype=torch.long)

        self.indices['train'] = self.as_something(indices, dtype=torch.long)
        if labels is not None:
            self.labels_split['train'] = labels[indices]

    def set_statistics(self, stats):
        self.statistics = stats

    def build_sampler(self, batch_size, subset=None, indices=None, persistent=True, oversample=False, weight_factor=1., expansion_size=int(1e7),
                       dynamic=False, buffer_size=None, probs_normalization='sum', tail=True, sample_size=100000):

        from sklearn.utils.class_weight import compute_sample_weight

        if indices is None:
            if subset is None:
                if self.index is not None:
                    indices = self.index.index.values
                else:
                    indices = torch.arange(len(self))
            else:
                indices = self.indices[subset]
        else:
            indices = self.as_something(indices, dtype=torch.long)

        if not persistent:
            return UniversalBatchSampler(indices, batch_size, shuffle=False,
                                         tail=tail, once=True, dynamic=False)

        probs = None
        if oversample and subset in self.labels_split and self.labels_split[subset] is not None:
            probs = compute_sample_weight('balanced', y=self.labels_split[subset]) ** weight_factor
            probs_normalization = 'sum'
        elif subset is None and check_type(self.probs).major == Types.array:
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
