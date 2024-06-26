import spacy
import pandas as pd
import numpy as np

from beam.transformer import Transformer
from beam.algorithm import TextGroupExpansionAlgorithm
from beam.algorithm.group_expansion import InvMap
from beam.utils import cached_property

from .utils import (replace_entity_over_series, build_dataset, split_dataset)


class EnronTicketSimilarity(TextGroupExpansionAlgorithm):

    @cached_property
    def entity_remover(self):
        return Transformer(self.hparams, func=replace_entity_over_series)

    @cached_property
    def nlp_model(self):
        nlp = spacy.load(self.get_hparam('nlp-model'))
        nlp.max_length = self.get_hparam('nlp-max-length')
        return nlp

    def preprocess_body(self):
        v = self.metadata['body'].str.split('-----Original Message-----').str[0]
        self.entity_remover.transform(v, nlp=self.nlp_model, transform_kwargs={
            'store_path': self.root_path.joinpath('enron_mails_without_entities_body')})

    def preprocess_title(self):
        self.entity_remover.transform(self.metadata['subject'], nlp=self.nlp_model, transform_kwargs={
            'store_path': self.root_path.joinpath('enron_mails_without_entities_title')})

    def split_dataset(self):
        split_dataset(self.metadata, self.root_path.joinpath('split_dataset'),
                      train_ratio=self.get_hparam('train-ratio'),
                      val_ratio=self.get_hparam('val-ratio'),
                      gap_days=self.get_hparam('gap-days'))

    def build_dataset(self):
        build_dataset(self.subsets, self.root_path)

    @classmethod
    @property
    def excluded_attributes(cls):
        return super(EnronTicketSimilarity, cls).excluded_attributes.union(['nlp_model'])

    @cached_property
    def x(self):
        vals = {'train': self.dataset[f'x_train'].values,
                'validation': self.dataset['x_val'].values,
                'test': self.dataset['x_test'].values}

        if self.get_hparam('subset-labels', None):
            vals = {k: [v[i] for i in self.full_invmap[k][self.ind[k]]] for k, v in vals.items()}

        return vals

    @cached_property
    def y(self):
        vals = {'train': self.dataset[f'y_train'].values,
                'validation': self.dataset['y_val'].values,
                'test': self.dataset['y_test'].values}

        if self.get_hparam('subset-labels', None):
            vals = {k: v[self.full_invmap[k][self.ind[k]]] for k, v in vals.items()}

        return vals

    @cached_property
    def ind(self):

        subset_labels = self.get_hparam('subset-labels', None)
        if subset_labels:
            _ind = {}
            v = self.subsets['train'].values
            _ind['train'] = v[v['label'].isin(subset_labels)].index
            v = self.subsets['validation'].values
            _ind['validation'] = v[v['label'].isin(subset_labels)].index
            v = self.subsets['test'].values
            _ind['test'] = v[v['label'].isin(subset_labels)].index

        else:
            _ind = {'train': self.subsets['train'].values.index,
                    'validation': self.subsets['validation'].values.index,
                    'test': self.subsets['test'].values.index}

        return _ind


    @cached_property
    def _full_invmap(self):
        im = {}
        for k, v in self.full_ind.items():
            s = pd.Series(np.arange(len(v.values)), index=v.values)
            im[k] = s.sort_index()
        return im

    @cached_property
    def full_invmap(self):

        return {k: InvMap(v) for k, v in self._full_invmap.items()}


    @cached_property
    def full_ind(self):
        return {'train': self.subsets['train'].values.index,
                'validation': self.subsets['validation'].values.index,
                'test': self.subsets['test'].values.index}