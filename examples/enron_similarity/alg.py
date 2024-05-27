from collections import defaultdict
from functools import cached_property
from typing import List

import numpy as np
import pandas as pd
import spacy

from src.beam import resource, BeamData, Timer, as_numpy
from src.beam.transformer import Transformer
from src.beam import beam_logger as logger
from src.beam.algorithm import Algorithm
from src.beam.type import BeamType

from .utils import (replace_entity_over_series, build_dataset, split_dataset, extract_textstat_features,
                    InvMap, Tokenizer)


class GroupExpansionAlgorithm(Algorithm):

    @cached_property
    def base_classifier(self):
        alg = None
        if self.get_hparam('classifier') == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            alg = RandomForestClassifier(n_estimators=100)
        elif self.get_hparam('classifier') == 'catboost':
            from src.beam.algorithm import CBAlgorithm
            alg = CBAlgorithm(self.hparams)
        return alg

    @cached_property
    def pu_classifier(self):
        from pulearn import BaggingPuClassifier
        alg = BaggingPuClassifier(estimator=self.base_classifier,
                                  verbose=self.get_hparam('pu_verbose', 10),
                                  n_estimators=self.get_hparam('pu_n_estimators', 15),)
        return alg

    def expand(self, group):
        raise NotImplementedError

    def predict(self, group):
        raise NotImplementedError


class TicketSimilarity(GroupExpansionAlgorithm):

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

    @cached_property
    def svd_transformer(self):
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=self.get_hparam('svd-components', 128))
        return svd

    @cached_property
    def pca_transformer(self):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.get_hparam('pca-components', 128))
        return pca

    @staticmethod
    def svd_preprocess(x):

        x_type = BeamType.check_minor(x)

        if x_type.minor == 'tensor':
            crow_indices = x.crow_indices().numpy()
            col_indices = x.col_indices().numpy()
            values = x.values().numpy()

            # Create a SciPy CSR matrix
            from scipy.sparse import csr_matrix
            x = csr_matrix((values, col_indices, crow_indices), shape=x.size())
        return x

    def svd_fit_transform(self, x):
        x = self.svd_preprocess(x)
        return self.svd_transformer.fit_transform(x)

    def svd_transform(self, x):
        x = self.svd_preprocess(x)
        return self.svd_transformer.transform(x)

    def pca_fit_transform(self, x):
        return self.pca_transformer.fit_transform(as_numpy(x))

    def pca_transform(self, x):
        return self.pca_transformer.transform(as_numpy(x))

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

    @cached_property
    def subsets(self):
        subsets = BeamData.from_path(self.root_path.joinpath('split_dataset'))
        subsets.cache()
        return subsets

    @cached_property
    def tfidf_sim(self):
        from src.beam.similarity import TFIDF
        # for now fix the metric as it is the only supported metric in tfidf sim
        sim = {}
        for k in ['validation', 'test']:
            tokenizer = Tokenizer(self.hparams)
            sim[k] = TFIDF(preprocessor=tokenizer.tokenize, metric='bm25',
                        chunksize=self.get_hparam('tokenizer-chunksize'),
                        hparams=self.hparams)
        return sim

    @cached_property
    def dense_sim(self):
        from src.beam.similarity import TextSimilarity
        sim = {}
        for k in ['validation', 'test']:
            sim[k] = TextSimilarity(hparams=self.hparams)
        return sim

    @cached_property
    def _invmap(self):
        im = {}
        for k, v in self.subsets.items():
            s = pd.Series(np.arange(len(v.values)), index=v.values.index)
            im[k] = s.sort_index()
        return im

    @cached_property
    def invmap(self):

        return {k: InvMap(v) for k, v in self._invmap.items()}

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

    @cached_property
    def robust_scaler(self):
        from sklearn.preprocessing import RobustScaler
        return RobustScaler()

    def robust_scale_fit_transform(self, x):
        return self.robust_scaler.fit_transform(as_numpy(x))

    def robust_scale_transform(self, x):
        return self.robust_scaler.transform(as_numpy(x))

    def reset(self):
        for k in ['validation', 'test']:
            self.tfidf_sim[k].reset()
            self.dense_sim[k].reset()

    def fit_tfidf(self, subset='validation'):
        # we need to fit the tfidf model and also apply the transformation in order to
        # calculate the doc_len_sparse attribute
        self.tfidf_sim[subset].fit_transform(self.x[subset], index=self.ind[subset])

    def fit_dense(self, subset='validation'):
        self.dense_sim[subset].add(self.x[subset], index=self.ind[subset])

    def search_tfidf(self, query, subset='validation', k=5):
        return self.tfidf_sim[subset].search(query, k=k)

    def search_dense(self, query, subset='validation', k=5):
        return self.dense_sim[subset].search(query, k=k)

    @classmethod
    @property
    def special_state_attributes(cls):
        return super().special_state_attributes + ['tfidf_sim', 'dense_sim', 'features']

    @classmethod
    @property
    def excluded_attributes(cls):
        return (['dataset', 'metadata', 'nlp_model', 'subsets', 'x', 'y', 'ind'] +
                super().excluded_attributes)

    def load_state_dict(self, path, ext=None, exclude: List = None, **kwargs):
        super().load_state_dict(path, ext=ext, exclude=exclude, **kwargs)
        tokenizer = Tokenizer(self.hparams)
        for k in self.tfidf_sim.keys():
            self.tfidf_sim[k].preprocessor = tokenizer.tokenize

    def build_group_dataset(self, group_label,
                            known_subset='train',
                            unknown_subset='validation', k_sparse=None, k_dense=None):

        k_sparse = k_sparse or self.get_hparam('k-sparse')
        k_dense = k_dense or self.get_hparam('k-dense')

        if not self.tfidf_sim[unknown_subset].is_trained:
            logger.warning(f"TFIDF model not fitted for {unknown_subset}. Fitting now")
            self.fit_tfidf(subset=unknown_subset)

        if not self.dense_sim[unknown_subset].is_trained:
            logger.warning(f"Dense model not fitted for {unknown_subset}. Fitting now")
            self.fit_dense(subset=unknown_subset)

        ind_pos = np.where(self.y[known_subset] == group_label)[0]
        v = self.x[known_subset]
        x_pos = [v[i] for i in ind_pos]
        y_pos = np.ones(len(ind_pos), dtype=int)

        res_sparse = self.search_tfidf(x_pos, subset=unknown_subset, k=k_sparse)
        res_dense = self.search_dense(x_pos, subset=unknown_subset, k=k_dense)

        ind_sparse = self.invmap[unknown_subset][res_sparse.index.flatten()]
        ind_dense = self.invmap[unknown_subset][res_dense.index.flatten()]

        ind_unlabeled = np.unique(np.concatenate([ind_sparse, ind_dense], axis=0))

        x_unlabeled = [self.x[unknown_subset][k] for k in ind_unlabeled]

        y_unlabeled = np.zeros(len(ind_unlabeled), dtype=int)

        y_unlabeled_true = self.y[unknown_subset][ind_unlabeled]

        return {'x_pos': x_pos, 'y_pos': y_pos, 'ind_pos': ind_pos,
                'x_unlabeled': x_unlabeled,
                'y_unlabeled': y_unlabeled, 'y_unlabeled_true': y_unlabeled_true, 'ind_unlabeled': ind_unlabeled}

    def _build_features(self, x, is_train=False, n_workers=None):

        transform_kwargs = {}
        if n_workers is not None:
            transform_kwargs['n_workers'] = n_workers
        x_tfidf = self.tfidf_sim['validation'].transform(x, transform_kwargs=transform_kwargs)
        x_dense = self.dense_sim['validation'].encode(x)
        # x_textstat = extract_textstat_features(x, n_workers=self.get_hparam('n_workers'))
        x_textstat = extract_textstat_features(x, n_workers=1)

        if is_train:
            with Timer(name='svd_transform', logger=logger):
                x_svd = self.svd_fit_transform(x_tfidf)
            with Timer(name='pca_transform', logger=logger):
                x_pca = self.pca_fit_transform(x_dense)
            with Timer(name='extract_textstat_features', logger=logger):
                x_textstat = self.robust_scale_fit_transform(x_textstat)
        else:
            x_svd = self.svd_transform(x_tfidf)
            x_pca = self.pca_transform(x_dense)
            x_textstat = self.robust_scale_transform(x_textstat)

        x = np.concatenate([x_pca, x_svd, x_textstat], axis=1)

        return x

    @cached_property
    def features(self):

        # p = '/home/shared/data/results/enron/tmp/features'
        # bd = BeamData.from_path(p)
        # bd.cache()
        # return bd.values

        f = {'validation': self._build_features(self.x['validation'], is_train=True),
             'train': self._build_features(self.x['train'], is_train=False),
             'test': self._build_features(self.x['test'], is_train=False)}
        # the "validation" set which is the second set is the true trained set.
        return f

    def build_features(self):
        _ = self.features

    def build_classification_datasets(self, group_label,
                                  known_subset='train',
                                  unknown_subset='validation',
                                  test_subset='test',
                                  k_sparse=None, k_dense=None):
        res_train = self.build_group_dataset(group_label, known_subset=known_subset, unknown_subset=unknown_subset,
                                       k_sparse=k_sparse, k_dense=k_dense)
        ind = np.concatenate([res_train['ind_unlabeled'], res_train['ind_pos']])
        x_train = self.features[known_subset][ind]
        y_train = np.concatenate([res_train['y_unlabeled'], res_train['y_pos']])

        res_test = self.build_group_dataset(group_label, known_subset=known_subset, unknown_subset=test_subset,
                                      k_sparse=k_sparse, k_dense=k_dense)

        x_test = self.features[test_subset][res_test['ind_unlabeled']]
        y_test = (res_test['y_unlabeled_true'] == group_label).astype(int)

        return {'x_train': x_train, 'y_train': y_train, 'ind_train': ind,
                'y_train_true': res_train['y_unlabeled_true'],
                'x_test': x_test, 'y_test': y_test, 'ind_test': res_test['ind_unlabeled'],
                'y_test_true': res_test['y_unlabeled_true']}

    def calculate_evaluation_metrics(self, group_label, datasets, y_pred_train, y_pred_test,
                                     unknown_subset='validation', test_subset='test'):

        y_pred = {'train': y_pred_train, 'test': y_pred_test}
        # calculate metrics:
        results = defaultdict(dict)
        for part, org_set in (('train', unknown_subset), ('test', test_subset)):

            results[part]['original_pool'] = len(self.y[org_set])
            results[part]['prevalence_count'] = (self.y[org_set] == group_label).sum()
            results[part]['prevalence'] = results[part]['prevalence_count'] / results[part]['original_pool']
            results[part]['expansion_recall_count'] = (datasets[f'y_{part}_true'] == group_label).sum()
            results[part]['expansion_recall'] = (results[part]['expansion_recall_count']
                                                 / results[part]['prevalence_count'])
            results[part]['expansion_pool'] = (datasets[f'y_{part}'] == 0).sum()
            results[part]['expansion_precision'] = (results[part]['expansion_recall_count'] /
                                                    results[part]['expansion_pool'])

            if part == 'train':
                y_train_true = datasets[f'y_{part}_true'] == group_label
                results[part]['final_recall_count'] = (y_pred['train'] * y_train_true == 1).sum()
            else:
                results[part]['final_recall_count'] = (y_pred['test'] * datasets[f'y_test'] == 1).sum()

            results[part]['final_recall'] = (results[part]['final_recall_count']
                                             / results[part]['prevalence_count'])
            results[part]['final_pool'] = (y_pred[part] == 1).sum()
            results[part]['final_precision'] = (results[part]['final_recall_count'] /
                                                results[part]['final_pool'])

        return results

    def evaluate(self, group_label, known_subset='train', unknown_subset='validation',
                 test_subset='test', k_sparse=None, k_dense=None, threshold=0.5):

        datasets = self.build_classification_datasets(group_label, known_subset=known_subset,
                                                unknown_subset=unknown_subset, test_subset=test_subset,
                                                k_sparse=k_sparse, k_dense=k_dense)

        self.pu_classifier.fit(datasets['x_train'], datasets['y_train'])

        x_train_unlabeled = datasets['x_train'][datasets['y_train'] == 0]
        y_pred = {'train': self.pu_classifier.predict_proba(x_train_unlabeled)[:, 1],
                  'test': self.pu_classifier.predict_proba(datasets['x_test'])[:, 1]}

        metrics = self.calculate_evaluation_metrics(group_label, datasets, y_pred['train'] > threshold,
                                                    y_pred['test'] > threshold,
                                                    unknown_subset=unknown_subset,
                                                    test_subset=test_subset)

        results = {'metrics': metrics, 'datasets': datasets, 'y_pred': y_pred}

        return results
