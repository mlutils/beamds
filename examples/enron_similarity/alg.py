from functools import cached_property

import numpy as np
import pandas as pd
import spacy

from src.beam import resource, BeamData
from src.beam.transformer import Transformer
from src.beam import beam_logger as logger
from src.beam.algorithm import Algorithm

from .utils import replace_entity_over_series, build_dataset, split_dataset


class GroupExpansionAlgorithm(Algorithm):

    @cached_property
    def base_classifier(self):
        from sklearn.ensemble import RandomForestClassifier
        alg = RandomForestClassifier(n_estimators=100)
        return alg

    @cached_property
    def pu_classifier(self):
        from pulearn import BaggingPuClassifier
        alg = BaggingPuClassifier(
            base_estimator=self.base_classifier, n_estimators=15)
        return alg

    def expand(self, group):
        raise NotImplementedError

    def data_of_index(self, index):
        raise NotImplementedError

    def predict(self, group):
        candidates = self.expand(group)

        x_positives = self.data_of_index(group)
        x_unlabeled = self.data_of_index(candidates)


class TicketSimilarity(GroupExpansionAlgorithm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitted_subset_tfidf = None
        self.fitted_subset_dense = None

    @cached_property
    def entity_remover(self):
        return Transformer(self.hparams, func=replace_entity_over_series)

    @cached_property
    def root_path(self):
        return resource(self.get_hparam('root-path'))

    @cached_property
    def dataset(self):
        bd = BeamData.from_path(self.root_path.joinpath('dataset'))
        bd.cache()
        return bd

    @cached_property
    def metadata(self):
        df = resource(self.get_hparam('path-to-data')).read(target='pandas')
        return df

    @cached_property
    def nlp_model(self):
        nlp = spacy.load(self.get_hparam('nlp-model'))
        nlp.max_length = self.get_hparam('nlp-max-length')
        return nlp

    def preprocess_body(self):
        self.entity_remover.transform(self.metadata['body'], nlp=self.nlp_model, transform_kwargs={
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

    @cached_property
    def subsets(self):
        subsets = BeamData.from_path(self.root_path.joinpath('split_dataset'))
        subsets.cache()
        return subsets

    @cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.get_hparam('tokenizer'))

    def tokenize(self, x):
        return self.tokenizer(x)['input_ids']

    @cached_property
    def tfidf_sim(self):
        from src.beam.similarity import TFIDF
        # for now fix the metric as it is the only supported metric in tfidf sim
        sim = TFIDF(preprocessor=self.tokenize, metric='bm25',
                    chunksize=self.get_hparam('tokenizer-chunksize'),
                    hparams=self.hparams)
        return sim

    @cached_property
    def dense_sim(self):
        from src.beam.similarity import TextSimilarity
        sim = TextSimilarity(hparams=self.hparams)
        return sim

    @cached_property
    def invmap(self):
        im = {}
        for k, v in self.subsets.items():
            im[k] = pd.Series(np.arange(len(v.values)), index=v.values.index)
        return im

    @cached_property
    def x(self):
        return {'train': self.dataset[f'x_train'].values,
                'validation': self.dataset['x_val'].values,
                'test': self.dataset['x_test'].values}

    @cached_property
    def y(self):
        return {'train': self.dataset[f'y_train'].values,
                'validation': self.dataset['y_val'].values,
                'test': self.dataset['y_test'].values}

    @cached_property
    def ind(self):
        return {'train': self.subsets['train'].values.index,
                'validation': self.subsets['validation'].values.index,
                'test': self.subsets['test'].values.index}

    def reset(self):
        self.tfidf_sim.reset()
        self.dense_sim.reset()
        self.fitted_subset_tfidf = None
        self.fitted_subset_dense = None

    def fit_tfidf(self, subset='validation'):
        self.tfidf_sim.fit_transform(self.x[subset], index=self.ind[subset])
        self.fitted_subset_tfidf = subset

    def fit_dense(self, subset='validation'):
        self.dense_sim.add(self.x[subset], index=self.ind[subset])
        self.fitted_subset_dense = subset

    def search_tfidf(self, query, k=5):
        return self.tfidf_sim.search(query, k=k)

    def search_dense(self, query, k=5):
        return self.dense_sim.search(query, k=k)

    def save_state(self, path, ext=None):

        path = resource(path)
        super().save_state(path.joinpath('internal_state'), ext=ext)
        self.tfidf_sim.save_state(path.joinpath('tfidf_sim'))
        self.dense_sim.save_state(path.joinpath('dense_sim'))

    def load_state(self, path):

        path = resource(path)
        super().load_state(path.joinpath('internal_state'))
        try:
            self.tfidf_sim.load_state(path.joinpath('tfidf_sim'))
        except FileNotFoundError:
            logger.warning("TFIDF model not found")
        try:
            self.dense_sim.load_state(path.joinpath('dense_sim'))
        except FileNotFoundError:
            logger.warning("Dense model not found")

    @property
    def state_attributes(self):
        return ['nlp_model', 'entity_remover', 'root_path',
                'dataset', 'metadata', 'nlp_model', 'tokenizer', 'tfidf_sim'] + super().state_attributes

    def build_group_dataset(self, group_label,
                            known_subset='train',
                            unknown_subset='validation', k_sparse=None, k_dense=None):

        k_sparse = k_sparse or self.get_hparam('k-sparse')
        k_dense = k_dense or self.get_hparam('k-dense')

        if self.fitted_subset_tfidf != known_subset:
            if self.fitted_subset_tfidf is not None:
                logger.warning(f"TFIDF model not fitted for {known_subset}. Fitting now and overriding existing fit")
            else:
                logger.info(f"TFIDF model not fitted for {known_subset}. Fitting now")
            self.fit_tfidf(known_subset)

        if self.fitted_subset_dense != known_subset:
            if self.fitted_subset_dense is not None:
                logger.warning(f"Dense model not fitted for {known_subset}. Fitting now and overriding existing fit")
            else:
                logger.info(f"Dense model not fitted for {known_subset}. Fitting now")
            self.fit_dense(known_subset)

        ind_pos = np.where(self.y[known_subset] == group_label)[0]
        v = self.x[known_subset]
        x_pos = [v[i] for i in ind_pos]
        y_pos = np.ones(len(ind_pos), dtype=int)

        res_sparse = self.search_tfidf(x_pos, k=k_sparse)
        res_dense = self.search_dense(x_pos, k=k_dense)



