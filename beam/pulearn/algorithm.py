import re

import pandas as pd

from ..path import beam_path
from ..utils import cached_property
from .config import PULearnConfig
from ..dataset import TabularDataset, UniversalDataset

from ..algorithm import CBAlgorithm, Algorithm
from .model import BeamPUClassifier, BeamCatboostClassifier
from ..logging import beam_logger as logger


class PUCBAlgorithm(CBAlgorithm):

    def __init__(self, hparams=None, name=None, **kwargs):

        super().__init__(hparams=hparams, name=name, _config_scheme=PULearnConfig,  **kwargs)
        self._t0 = None
        self._batch_size = None
        self.pu = self.set_pu()

    @property
    def task_type(self):
        return 'classification'

    @cached_property
    def model(self):
        return BeamCatboostClassifier(**self.catboost_kwargs)

    def set_pu(self):
        conf = {k.removeprefix('pu_'): v for k, v in self.hparams.items() if k in self.hparams.tags.PULearnConfig}
        estimator = conf.pop('estimator', 'catboost')
        if estimator == 'catboost':
            estimator = self.model

        return BeamPUClassifier(estimator=estimator, **conf)

    def _fit(self, x=None, y=None, dataset=None, eval_set=None, cat_features=None, text_features=None,
             embedding_features=None, sample_weight=None, **kwargs):

        self.set_objective()

        if self.experiment:
            self.experiment.prepare_experiment_for_run()

        if dataset is None:
            dataset = TabularDataset(x_train=x, y_train=y, x_test=eval_set[0], y_test=eval_set[1],
                                     cat_features=cat_features, text_features=text_features,
                                     embedding_features=embedding_features, sample_weight=sample_weight)

        self.set_train_reporter(first_epoch=0, n_epochs=self.n_epochs)

        X, y = dataset.get_subset_data('train')
        self.epoch = 0
        self.reporter_pre_epoch(0, batch_size=len(y))

        snapshot_file = None
        if self.experiment:
            snapshot_file = self.experiment.snapshot_file

        self.pu.set_features(cat_features=dataset.cat_columns, embedding_features=dataset.embedding_columns,
                             text_features=dataset.text_columns)

        self.pu.fit(X, y, estimator_fit_kwargs=dict(snapshot_interval=self.get_hparam('snapshot_interval'),
                                                    save_snapshot=self.get_hparam('save_snapshot')), **kwargs)

        if self.experiment:
            self.experiment.save_state(self)

    def predict(self, x: UniversalDataset | pd.DataFrame, **kwargs):

        if isinstance(x, UniversalDataset):
            x = x.data

        return self.pu.predict(x)

    @classmethod
    @property
    def excluded_attributes(cls):
        return super(CBAlgorithm, cls).excluded_attributes.union(['model', 'pu'])

    def load_state_dict(self, path, ext=None, exclude: set | list = None, **kwargs):

        path = beam_path(path)
        # with local_copy(path.joinpath('model.cb'), as_beam_path=False) as p:
        #     self.model.load_model(p)

        self.pu = path.joinpath('pu.pkl').read()

        return Algorithm.load_state_dict(self, path, ext, exclude, **kwargs)

    def save_state_dict(self, state, path, ext=None,  exclude: set | list = None, **kwargs):

        Algorithm.save_state_dict(self, state, path, ext, exclude, **kwargs)

        path = beam_path(path)
        # with local_copy(path.joinpath('model.cb'), as_beam_path=False) as p:
        #     self.model.save_model(p)
        path.joinpath('pu.pkl').write(self.pu)
