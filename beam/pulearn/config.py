from ..config import BeamConfig, BeamParam, ExperimentConfig
from ..algorithm.catboost_algorithm import CatboostConfig


class PULearnConfig(BeamConfig):

    parameters = [
        BeamParam('pu-estimator', str, 'catboost', 'The base estimator to fit on random subsets of the dataset'),
        BeamParam('pu-n_estimators', int, 10, 'The number of base estimators in the ensemble'),
        BeamParam('pu-max_samples', int, 1.0, 'The number of unlabeled samples to draw to train each base estimator'),
        BeamParam('pu-max_features', int, 1.0, 'The number of features to draw from X to train each base estimator'),
        BeamParam('pu-bootstrap', bool, True, 'Whether samples are drawn with replacement'),
        BeamParam('pu-bootstrap_features', bool, False, 'Whether features are drawn with replacement'),
        BeamParam('pu-oob_score', bool, True, 'Whether to use out-of-bag samples to estimate the generalization error'),
        BeamParam('pu-warm_start', bool, False,
                  'When set to True, reuse the solution of the previous call to fit and add more estimators to the '
                  'ensemble, otherwise, just fit a whole new ensemble'),
        BeamParam('pu-n_jobs', int, 1, 'The number of jobs to run in parallel for both `fit` and `predict`'),
        BeamParam('pu-random_state', int, None,
                  'If int, random_state is the seed used by the random number generator; '
                  'If RandomState instance, random_state is the random number generator; '
                  'If None, the random number generator is the RandomState instance used by `np.random`'),
        BeamParam('pu-verbose', int, 0, 'Controls the verbosity of the building process'),
    ]


class PULearnExperimentConfig(PULearnConfig, ExperimentConfig):
    defaults = {'project': 'pu_learn_beam', 'algorithm': 'BeamPUClassifier'}


class PULearnCBExperimentConfig(PULearnExperimentConfig, CatboostConfig):
    pass