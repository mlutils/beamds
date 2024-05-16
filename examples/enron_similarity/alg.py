from functools import cached_property

import numpy as np
import pandas as pd
import spacy

from src.beam import resource, BeamData, Timer, as_numpy
from src.beam.transformer import Transformer
from src.beam import beam_logger as logger
from src.beam.algorithm import Algorithm

from .utils import replace_entity_over_series, build_dataset, split_dataset, extract_textstat_features


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
    def _invmap(self):
        im = {}
        for k, v in self.subsets.items():
            s = pd.Series(np.arange(len(v.values)), index=v.values.index)
            im[k] = s.sort_index()
        return im

    @cached_property
    def invmap(self):
        class InvMap:
            def __init__(self, invmap):
                self._invmap = invmap

            def __getitem__(self, ind):
                return self._invmap[ind].values

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

    def robust_scale(self, x):
        return self.robust_scaler.fit_transform(as_numpy(x))

    def reset(self):
        self.tfidf_sim.reset()
        self.dense_sim.reset()
        self.fitted_subset_tfidf = None
        self.fitted_subset_dense = None

    def fit_tfidf(self, subset='validation'):
        # we need to fit the tfidf model and also apply the transformation in order to
        # calculate the doc_len_sparse attribute
        self.tfidf_sim.fit_transform(self.x[subset][:1000], index=self.ind[subset][:1000])
        self.fitted_subset_tfidf = subset

    def fit_dense(self, subset='validation'):
        self.dense_sim.add(self.x[subset][:1000], index=self.ind[subset][:1000])
        self.fitted_subset_dense = subset

    def search_tfidf(self, query, k=5):
        return self.tfidf_sim.search(query, k=k)

    def search_dense(self, query, k=5):
        return self.dense_sim.search(query, k=k)

    # def save_state(self, path, ext=None):
    #
    #     path = resource(path)
    #     super().save_state(path.joinpath('internal_state'), ext=ext)
    #     self.tfidf_sim.save_state(path.joinpath('tfidf_sim'))
    #     self.dense_sim.save_state(path.joinpath('dense_sim'))
    #
    # def load_state(self, path):
    #
    #     path = resource(path)
    #     super().load_state(path.joinpath('internal_state'))
    #     try:
    #         self.tfidf_sim.load_state(path.joinpath('tfidf_sim'))
    #     except Exception as e:
    #         logger.warning(f"TFIDF model not found: {e}")
    #         raise e
    #     try:
    #         self.dense_sim.load_state(path.joinpath('dense_sim'))
    #     except Exception as e:
    #         logger.warning(f"Dense model not found: {e}")

    @property
    def special_state_attributes(self):
        return ['tfidf_sim', 'dense_sim'] + super().special_state_attributes

    @property
    def excluded_attributes(self):
        return (['dataset', 'metadata', 'nlp_model', 'tokenizer', 'subsets', 'x', 'y', 'ind'] +
                super().excluded_attributes)

    def build_group_dataset(self, group_label,
                            known_subset='train',
                            unknown_subset='validation', k_sparse=None, k_dense=None):

        k_sparse = k_sparse or self.get_hparam('k-sparse')
        k_dense = k_dense or self.get_hparam('k-dense')

        if self.fitted_subset_tfidf != unknown_subset:
            if self.fitted_subset_tfidf is not None:
                logger.warning(f"TFIDF model not fitted for {unknown_subset}. Fitting now and overriding existing fit")
            else:
                logger.info(f"TFIDF model not fitted for {unknown_subset}. Fitting now")
            self.fit_tfidf(unknown_subset)

        if self.fitted_subset_dense != unknown_subset:
            if self.fitted_subset_dense is not None:
                logger.warning(f"Dense model not fitted for {unknown_subset}. Fitting now and overriding existing fit")
            else:
                logger.info(f"Dense model not fitted for {unknown_subset}. Fitting now")
            self.fit_dense(unknown_subset)

        ind_pos = np.where(self.y[known_subset] == group_label)[0]
        v = self.x[known_subset]
        x_pos = [v[i] for i in ind_pos]
        y_pos = np.ones(len(ind_pos), dtype=int)

        res_sparse = self.search_tfidf(x_pos, k=k_sparse)
        res_dense = self.search_dense(x_pos, k=k_dense)

        ind_sparse = self.invmap[unknown_subset][res_sparse.index.flatten()]
        ind_dense = self.invmap[unknown_subset][res_dense.index.flatten()]

        ind_unlabeled = np.unique(np.concatenate([ind_sparse, ind_dense], axis=0))

        x_unlabeled = [self.x[unknown_subset][k] for k in ind_unlabeled]

        y_unlabeled = np.zeros(len(ind_unlabeled), dtype=int)

        y_unlabeled_true = self.y[unknown_subset][ind_unlabeled]

        return {'x_pos': x_pos, 'y_pos': y_pos,
                'x_unlabeled': x_unlabeled,
                'y_unlabeled': y_unlabeled, 'y_unlabeled_true': y_unlabeled_true, 'ind_unlabeled': ind_unlabeled}

    def build_features(self, x):
        x_tfidf = self.tfidf_sim.transform(x)
        x_dense = self.dense_sim.encode(x)
        with Timer(name='svd_transform', logger=logger):
            x_svd = self.svd_fit_transform(x_tfidf)
        with Timer(name='pca_transform', logger=logger):
            x_pca = self.pca_fit_transform(x_dense)
        with Timer(name='extract_textstat_features', logger=logger):
            x_textstat = extract_textstat_features(x, n_workers=self.get_hparam('n_workers'))
            x_textstat = self.robust_scale(x_textstat)
        v = np.concatenate([x_pca, x_svd, x_textstat], axis=1)
        return v

