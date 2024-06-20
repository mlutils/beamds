from collections import defaultdict
from functools import cached_property
from typing import List, Union

import numpy as np
import pandas as pd

from .. import resource, BeamData, Timer, as_numpy
from ..logging import beam_logger as logger
from .core_algorithm import Algorithm
from ..type import BeamType


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


class TextGroupExpansionAlgorithm(GroupExpansionAlgorithm):

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
        for k in ['train', 'validation', 'test']:
            tokenizer = Tokenizer(self.hparams)
            sim[k] = TFIDF(preprocessor=tokenizer.tokenize, metric='bm25',
                        chunksize=self.get_hparam('tokenizer-chunksize'),
                        hparams=self.hparams)
        return sim

    def build_dense_model(self):
        from src.beam.similarity import TextSimilarity
        return TextSimilarity.load_dense_model(dense_model_path=self.get_hparam('dense-model-path'),
                                               dense_model_device=self.get_hparam('dense_model_device'),
                                               **self.get_hparam('st_kwargs', {}))

    @cached_property
    def dense_sim(self):
        from src.beam.similarity import TextSimilarity
        dense_model = self.build_dense_model()
        sim = {}
        for k in ['train', 'validation', 'test']:
            sim[k] = TextSimilarity(dense_model=dense_model, hparams=self.hparams, metric='l2')
        return sim

    @cached_property
    def _invmap(self):
        im = {}
        for k, v in self.ind.items():
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
        for k in ['train', 'validation', 'test']:
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
        return super(TextGroupExpansionAlgorithm, cls).special_state_attributes.union(['tfidf_sim', 'dense_sim'
                                                                                       'features'])

    @classmethod
    @property
    def excluded_attributes(cls):
        return super(TextGroupExpansionAlgorithm, cls).excluded_attributes.union(['dataset', 'metadata', 'subsets', 'x',
                                                                                  'y', 'ind'])

    def load_state_dict(self, path, ext=None, exclude: List = None, **kwargs):
        super().load_state_dict(path, ext=ext, exclude=exclude, **kwargs)
        tokenizer = Tokenizer(self.hparams)
        for k in self.tfidf_sim.keys():
            self.tfidf_sim[k].preprocessor = tokenizer.tokenize

        dense_model = self.build_dense_model()
        for k in self.dense_sim.keys():
            self.dense_sim[k].set_dense_model(dense_model)

    def search_dual(self, query, subset='validation', k_sparse=5, k_dense=5):
        res_sparse = self.search_tfidf(query, subset=subset, k=k_sparse)
        res_dense = self.search_dense(query, subset=subset, k=k_dense)

        loc_sparse = res_sparse.index.flatten()
        loc_dense = res_dense.index.flatten()
        iloc_sparse = self.invmap[subset][loc_sparse]
        iloc_dense = self.invmap[subset][loc_dense]
        source_sparse = np.repeat(np.arange(len(query))[:, None], k_sparse, axis=1).flatten()
        source_dense = np.repeat(np.arange(len(query))[:, None], k_dense, axis=1).flatten()
        val_sparse = res_sparse.values.flatten()
        val_dense = res_dense.values.flatten()

        df_sparse = pd.DataFrame({'val': val_sparse, 'source': source_sparse,
                                 'loc': loc_sparse, 'iloc': iloc_sparse})

        df_sparse = df_sparse.groupby('iloc').agg({'val': 'max', 'source': list, 'loc': 'first'})

        df_dense = pd.DataFrame({'val': val_dense, 'source': source_dense,
                                'loc': loc_dense, 'iloc': iloc_dense})

        df_dense = df_dense.groupby('iloc').agg({'val': 'max', 'source': 'list', 'loc': 'first'})

        df = pd.merge(df_sparse, df_dense, how='outer', left_index=True, right_index=True,
                      suffixes=('_sparse', '_dense'), indicator=True)

        return df

    def build_expansion_dataset(self, group_label, seed_subsets: Union[str, List[str]] = 'train',
                                expansion_subset='validation', k_sparse=None, k_dense=None):

        if isinstance(seed_subsets, str):
            seed_subsets = [seed_subsets]

        k_sparse = k_sparse or self.get_hparam('k-sparse')
        k_dense = k_dense or self.get_hparam('k-dense')

        if not self.tfidf_sim[expansion_subset].is_trained:
            logger.warning(f"TFIDF model not fitted for {expansion_subset}. Fitting now")
            self.fit_tfidf(subset=expansion_subset)

        if not self.dense_sim[expansion_subset].is_trained:
            logger.warning(f"Dense model not fitted for {expansion_subset}. Fitting now")
            self.fit_dense(subset=expansion_subset)

        x_seed = []
        x_seed_features = []
        iloc_seed = {}

        for seed_subset in seed_subsets:

            iloc_seed_j = np.where(self.y[seed_subset] == group_label)[0]
            x_seed.extend([self.x[seed_subset][i] for i in iloc_seed_j])
            x_seed_features.append(self.features[seed_subset][iloc_seed_j])
            iloc_seed[seed_subset] = iloc_seed_j

        y_seed = np.ones(len(x_seed), dtype=int)
        x_seed_features = np.concatenate(x_seed_features, axis=0)

        dual_search = self.search_dual(x_seed, subset=expansion_subset, k_sparse=k_sparse, k_dense=k_dense)

        ind_expansion = np.unique(np.concatenate([dual_search['iloc_sparse'], dual_search['iloc_dense']], axis=0))

        if expansion_subset in seed_subsets:
            ind_expansion = np.setdiff1d(ind_expansion, loc_seed)

        expansion_df = pd.DataFrame({'ind': ind_expansion})
        expansion_df['sparse_expansion'] = expansion_df['ind'].isin(ind_sparse)
        expansion_df['dense_expansion'] = expansion_df['ind'].isin(ind_dense)
        expansion_df['label'] = self.y[expansion_subset][ind_expansion]

        x_expansion = [self.x[expansion_subset][k] for k in ind_expansion]
        x_expansion_features = self.features[expansion_subset][ind_expansion]

        y_expansion = np.zeros(len(ind_expansion), dtype=int)
        x = np.concatenate([x_seed_features, x_expansion_features], axis=0)
        y = np.concatenate([y_seed, y_expansion], axis=0)
        ind = np.concatenate([loc_seed, ind_expansion], axis=0)

        return {'x': x, 'y': y, 'ind': ind, 'expansion_df': expansion_df}


        # return {'x_seed': x_seed, 'y_seed': y_seed, 'ind_seed': ind_seed,
        #         'x_expansion': x_expansion, 'y_expansion': y_expansion,
        #         'y_unlabeled_true': y_unlabeled_true, 'ind_expansion': ind_expansion,
        #         'expansion_df': expansion_df}

    def _build_features(self, x, is_train=False, n_workers=None):

        from ..misc.text_features import extract_textstat_features
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

        f = {'validation': self._build_features(self.x['validation'], is_train=True),
             'train': self._build_features(self.x['train'], is_train=False),
             'test': self._build_features(self.x['test'], is_train=False)}
        # the "validation" set which is the second set is the true trained set.
        return f

    def build_features(self):
        _ = self.features

    def build_classification_datasets(self, group_label, k_sparse=None, k_dense=None):

        dataset_train = self.build_expansion_dataset(group_label, seed_subsets='train', expansion_subset='train',
                                                     k_sparse=k_sparse, k_dense=k_dense)

        ind = np.concatenate([dataset_train['ind_unlabeled'], dataset_train['ind_pos']])
        x_train = self.features['train'][ind]
        y_train = np.concatenate([dataset_train['y_unlabeled'], dataset_train['y_pos']])

        dataset_validation = self.build_expansion_dataset(group_label, seed_subsets='train', expansion_subset='validation',
                                                          k_sparse=k_sparse, k_dense=k_dense)

        ind = np.concatenate([dataset_validation['ind_unlabeled'], dataset_validation['ind_pos']])
        x_val = self.features['validation'][ind]
        y_val = np.concatenate([dataset_validation['y_unlabeled'], dataset_validation['y_pos']])


        res_test = self.build_expansion_dataset(group_label, seed_subsets=['train', 'validation'], expansion_subset='test',
                                                k_sparse=k_sparse, k_dense=k_dense)

        x_test = self.features[test_subset][res_test['ind_unlabeled']]
        y_test = (res_test['y_unlabeled_true'] == group_label).astype(int)

        return {'x_train': x_train, 'y_train': y_train, 'ind_train': ind,
                'y_train_true': dataset_train['y_unlabeled_true'],
                'x_test': x_test, 'y_test': y_test, 'ind_test': res_test['ind_unlabeled'],
                'y_test_true': res_test['y_unlabeled_true']}

    def calculate_evaluation_metrics(self, group_label, datasets, y_pred_train, y_pred_test,
                                     expansion_subset='validation', test_subset='test'):

        y_pred = {'train': y_pred_train, 'test': y_pred_test}
        # calculate metrics:
        results = defaultdict(dict)
        for part, org_set in (('train', expansion_subset), ('test', test_subset)):

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

    def evaluate(self, group_label, seed_subset='train', expansion_subset='validation',
                 test_subset='test', k_sparse=None, k_dense=None, threshold=0.5):

        datasets = self.build_classification_datasets(group_label, seed_subset=seed_subset,
                                                      expansion_subset=expansion_subset, test_subset=test_subset,
                                                      k_sparse=k_sparse, k_dense=k_dense)

        self.pu_classifier.fit(datasets['x_train'], datasets['y_train'])

        x_train_unlabeled = datasets['x_train'][datasets['y_train'] == 0]
        y_pred = {'train': self.pu_classifier.predict_proba(x_train_unlabeled)[:, 1],
                  'test': self.pu_classifier.predict_proba(datasets['x_test'])[:, 1]}

        metrics = self.calculate_evaluation_metrics(group_label, datasets, y_pred['train'] > threshold,
                                                    y_pred['test'] > threshold,
                                                    expansion_subset=expansion_subset,
                                                    test_subset=test_subset)

        results = {'metrics': metrics, 'datasets': datasets, 'y_pred': y_pred}

        return results

    def explainability(self, x, explain_with_subset='train', k_sparse=None, k_dense=None):
        if not self.tfidf_sim[explain_with_subset].is_trained:
            logger.warning(f"TFIDF model not fitted for {explain_with_subset}. Fitting now")
            self.fit_tfidf(subset=explain_with_subset)

        if not self.dense_sim[explain_with_subset].is_trained:
            logger.warning(f"Dense model not fitted for {explain_with_subset}. Fitting now")
            self.fit_dense(subset=explain_with_subset)

        res = self.search_dual(x, subset=explain_with_subset, k_sparse=k_sparse, k_dense=k_dense)
        return res


class InvMap:
    def __init__(self, invmap):
        self._invmap = invmap

    def __getitem__(self, ind):
        return self._invmap[ind].values


class Tokenizer:

    def __init__(self, hparams):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.get('tokenizer'))

    def tokenize(self, x):
        return self.tokenizer(x)['input_ids']