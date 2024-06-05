from functools import cached_property
import spacy

from src.beam.transformer import Transformer
from src.beam.algorithm import TextGroupExpansionAlgorithm

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

