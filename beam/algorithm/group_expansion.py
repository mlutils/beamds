from collections import defaultdict
from typing import List, Union

import numpy as np
import pandas as pd

from ..resources import resource
from ..utils import cached_property, Timer, as_numpy
from ..data import BeamData
from ..logging import beam_logger as logger
from .core_algorithm import Algorithm
from ..misc import svd_preprocess
from dataclasses import dataclass


@dataclass
class ExpansionDataset:
    x: List[str]
    y: np.ndarray
    expansion_df: pd.DataFrame
    seed_subsets: List[str]
    expansion_subset: str
    group: int


@dataclass
class EvaluationMetrics:
    original_pool: int
    prevalence_count: int
    prevalence: float
    expansion_recall_count: int
    expansion_recall: float
    expansion_pool: int
    expansion_precision: float
    expansion_gain: float
    final_recall_count: int
    final_recall: float
    final_pool: int
    final_precision: float


@dataclass
class GroupExpansionResults:
    metrics: EvaluationMetrics
    dataset: ExpansionDataset
    y_pred: np.ndarray
    group_label: int
    k_sparse: int
    k_dense: int
    threshold: float
    pu_classifier: object


class GroupExpansionAlgorithm(Algorithm):

    @cached_property
    def base_classifier(self):
        alg = None
        if self.get_hparam('classifier') == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            alg = RandomForestClassifier(n_estimators=100)
        elif self.get_hparam('classifier') == 'catboost':
            from .catboost_algorithm import CBAlgorithm
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

    def svd_fit_transform(self, x):
        x = svd_preprocess(x)
        return self.svd_transformer.fit_transform(x)

    def svd_transform(self, x):
        x = svd_preprocess(x)
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
        from ..similarity import TFIDF
        # for now fix the metric as it is the only supported metric in tfidf sim
        sim = {}
        for k in ['train', 'validation', 'test']:
            tokenizer = Tokenizer(self.hparams)
            sim[k] = TFIDF(preprocessor=tokenizer.tokenize, metric='bm25',
                        chunksize=self.get_hparam('tokenizer-chunksize'),
                        hparams=self.hparams)
        return sim

    def build_dense_model(self):
        from ..similarity import TextSimilarity
        return TextSimilarity.load_dense_model(dense_model=self.get_hparam('dense-model'),
                                               dense_model_device=self.get_hparam('dense_model_device'),
                                               **self.get_hparam('st_kwargs', {}))

    @cached_property
    def dense_sim(self):
        from ..similarity import TextSimilarity
        dense_model = self.build_dense_model()
        sim = {}
        for k in ['train', 'validation', 'test']:
            sim[k] = TextSimilarity(dense_model=dense_model, hparams=self.hparams, metric='l2')
        return sim

    @cached_property
    def _invmap(self):
        im = {}
        for k, v in self.ind.items():
            s = pd.Series(np.arange(len(v)), index=v)
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

    def search_tfidf(self, query, subset='validation', k=5, tfidf_sim=None):
        if tfidf_sim is None:
            tfidf_sim = self.tfidf_sim
        return tfidf_sim[subset].search(query, k=k)

    def search_dense(self, query, subset='validation', k=5, dense_sim=None):
        if dense_sim is None:
            dense_sim = self.dense_sim
        return dense_sim[subset].search(query, k=k)

    @classmethod
    @property
    def special_state_attributes(cls):
        return super(TextGroupExpansionAlgorithm, cls).special_state_attributes.union(['tfidf_sim', 'dense_sim',
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

    def search_dual(self, query: Union[List, str], index=None, subset='validation', k_sparse=5, k_dense=5,
                    tfidf_sim=None, dense_sim=None):
        if isinstance(query, str):
            query = [query]

        res_sparse = self.search_tfidf(query, subset=subset, k=k_sparse, tfidf_sim=tfidf_sim)
        res_dense = self.search_dense(query, subset=subset, k=k_dense, dense_sim=dense_sim)

        if index is None:
            index = np.arange(len(query))

        loc_sparse = res_sparse.index.flatten()
        loc_dense = res_dense.index.flatten()
        iloc_sparse = self.invmap[subset][loc_sparse]
        iloc_dense = self.invmap[subset][loc_dense]
        source_sparse = np.repeat(index[:, None], k_sparse, axis=1).flatten()
        source_dense = np.repeat(index[:, None], k_dense, axis=1).flatten()
        val_sparse = as_numpy(res_sparse.distance.flatten())
        val_dense = as_numpy(res_dense.distance.flatten())

        df_sparse = pd.DataFrame({'val': val_sparse, 'source': source_sparse, 'loc': loc_sparse, 'iloc': iloc_sparse})

        df_sparse = df_sparse.groupby('iloc').agg({'val': 'max', 'source': list, 'loc': 'first'})

        df_dense = pd.DataFrame({'val': val_dense, 'source': source_dense, 'loc': loc_dense, 'iloc': iloc_dense})

        df_dense = df_dense.groupby('iloc').agg({'val': 'max', 'source': list, 'loc': 'first'})

        df = pd.merge(df_sparse, df_dense, how='outer', left_index=True, right_index=True,
                      suffixes=('_sparse', '_dense'), indicator=True)

        df['target'] = df.loc_sparse.fillna(df.loc_dense)

        # Replace values in the _merge column
        df['source_type'] = df['_merge'].replace({'left_only': 'sparse', 'right_only': 'dense', 'both': 'both'})

        df = df.drop(columns=['loc_sparse', 'loc_dense', '_merge'])

        df['label'] = self.y[subset][df.index]

        return df

    def build_expansion_dataset(self, group, seed_subsets: Union[str, List[str]] = 'train',
                                expansion_subset='validation', k_sparse=None, k_dense=None, features=True,
                                tfidf_sim=None, dense_sim=None):

        if isinstance(seed_subsets, str):
            seed_subsets = [seed_subsets]

        k_sparse = k_sparse or self.get_hparam('k-sparse')
        k_dense = k_dense or self.get_hparam('k-dense')

        if tfidf_sim is None and k_sparse and not self.tfidf_sim[expansion_subset].is_trained:
            logger.warning(f"TFIDF model not fitted for {expansion_subset}. Fitting now")
            self.fit_tfidf(subset=expansion_subset)

        if dense_sim is None and k_dense and not self.dense_sim[expansion_subset].is_trained:
            logger.warning(f"Dense model not fitted for {expansion_subset}. Fitting now")
            self.fit_dense(subset=expansion_subset)

        x_seed = []
        x_seed_features = []
        iloc_seed = {}
        loc_seed = []

        for seed_subset in seed_subsets:

            iloc_seed_j = np.where(self.y[seed_subset] == group)[0]
            x_seed.extend([self.x[seed_subset][i] for i in iloc_seed_j])
            if features:
                x_seed_features.append(self.features[seed_subset][iloc_seed_j])
            iloc_seed[seed_subset] = iloc_seed_j
            loc_seed.append(self.ind[seed_subset][iloc_seed_j])

        y_seed = np.ones(len(x_seed), dtype=int)
        if features:
            x_seed_features = np.concatenate(x_seed_features, axis=0)
        loc_seed = np.concatenate(loc_seed, axis=0)

        expansion_df = self.search_dual(x_seed, index=loc_seed, subset=expansion_subset, k_sparse=k_sparse,
                                        k_dense=k_dense, tfidf_sim=tfidf_sim, dense_sim=dense_sim)

        if expansion_subset in seed_subsets:
            expansion_df = expansion_df.loc[~expansion_df.index.isin(iloc_seed[expansion_subset])]

        x_expansion = [self.x[expansion_subset][k] for k in expansion_df.index]
        y_expansion = np.zeros(len(x_expansion), dtype=int)

        if features:
            x_expansion_features = self.features[expansion_subset][expansion_df.index]
            x = np.concatenate([x_seed_features, x_expansion_features], axis=0)
        else:
            x = x_seed
            x.extend(x_expansion)

        y = np.concatenate([y_seed, y_expansion], axis=0)

        return ExpansionDataset(x=x, y=y, expansion_df=expansion_df, seed_subsets=seed_subsets,
                                expansion_subset=expansion_subset, group=group)

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

    def calculate_evaluation_metrics(self, group_label, dataset, y_pred, _eps=1e-8):

        # use the dataclasses to perform the calculations
        original_pool = len(self.y[dataset.expansion_subset])
        prevalence_count = (self.y[dataset.expansion_subset] == group_label).sum()
        prevalence = prevalence_count / original_pool
        expansion_recall_count = (dataset.expansion_df.label == group_label).sum()
        expansion_recall = expansion_recall_count / (prevalence_count + _eps)
        expansion_pool = (dataset.y == 0).sum()
        expansion_precision = expansion_recall_count / (expansion_pool + _eps)
        expansion_gain = expansion_precision / (prevalence + _eps)
        y_train_true = dataset.expansion_df.label == group_label
        final_recall_count = (y_pred * y_train_true == 1).sum()
        final_recall = final_recall_count / (prevalence_count + _eps)
        final_pool = (y_pred == 1).sum()
        final_precision = final_recall_count / (final_pool + _eps)

        return EvaluationMetrics(original_pool, prevalence_count, prevalence, expansion_recall_count, expansion_recall,
                                 expansion_pool, expansion_precision, expansion_gain, final_recall_count, final_recall,
                                 final_pool, final_precision)

    def fit_group(self, group_label, k_sparse=None, k_dense=None, threshold=None, pu_classifier=None,
                  build_model=True, tfidf_sim=None, dense_sim=None):

        if pu_classifier is None:
            pu_classifier = self.pu_classifier
        if k_sparse is None:
            k_sparse = self.get_hparam('k-sparse')
        if k_dense is None:
            k_dense = self.get_hparam('k-dense')

        dataset_train = self.build_expansion_dataset(group_label, seed_subsets='train', expansion_subset='train',
                                                     k_sparse=k_sparse, k_dense=k_dense, features=build_model,
                                                     tfidf_sim=tfidf_sim, dense_sim=dense_sim)

        dataset_validation = self.build_expansion_dataset(group_label, seed_subsets='train',
                                                          expansion_subset='validation', k_sparse=k_sparse,
                                                          k_dense=k_dense)

        if build_model:
            pu_classifier.fit(dataset_train.x, dataset_train.y)
            x_validation_unlabeled = dataset_validation.x[dataset_validation.y == 0]
            y_pred = pu_classifier.predict_proba(x_validation_unlabeled)[:, 1]
        else:
            y_pred = np.ones(len(dataset_validation.y))

        metrics = self.calculate_evaluation_metrics(group_label, dataset_validation, y_pred > threshold)

        return GroupExpansionResults(metrics=metrics, dataset=dataset_validation, y_pred=y_pred,
                                     group_label=group_label, k_sparse=k_sparse, k_dense=k_sparse, threshold=threshold,
                                     pu_classifier=pu_classifier)

    def evaluate_group(self, group_label, k_sparse=None, k_dense=None, threshold=None, pu_classifier=None):

        if pu_classifier is None:
            pu_classifier = self.pu_classifier
        if k_sparse is None:
            k_sparse = self.get_hparam('k-sparse')
        if k_dense is None:
            k_dense = self.get_hparam('k-dense')

        dataset = self.build_expansion_dataset(group_label, seed_subsets=['train', 'validation'],
                                                expansion_subset='test', k_sparse=k_sparse, k_dense=k_dense)

        x_test_unlabeled = dataset.x[dataset.y == 0]
        y_pred = pu_classifier.predict_proba(x_test_unlabeled)[:, 1]

        metrics = self.calculate_evaluation_metrics(group_label, dataset, y_pred > threshold)

        return GroupExpansionResults(metrics=metrics, dataset=dataset, y_pred=y_pred, group_label=group_label,
                                     k_sparse=k_sparse, k_dense=k_sparse, threshold=threshold,
                                     pu_classifier=pu_classifier)

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