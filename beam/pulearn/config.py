from ..config import BeamConfig, BeamParam, ExperimentConfig
from ..algorithm.catboost_algorithm import CatboostConfig


class PULearnConfig(BeamConfig):

    parameters = [
        BeamParam('estimator', str, 'catboost', 'The base estimator to fit on random subsets of the dataset'),
        BeamParam('n_estimators', int, 10, 'The number of base estimators in the ensemble'),
        BeamParam('max_samples', int, 1.0, 'The number of unlabeled samples to draw to train each base estimator'),
        BeamParam('max_features', int, 1.0, 'The number of features to draw from X to train each base estimator'),
        BeamParam('bootstrap', bool, True, 'Whether samples are drawn with replacement'),
        BeamParam('bootstrap_features', bool, False, 'Whether features are drawn with replacement'),
        BeamParam('oob_score', bool, True, 'Whether to use out-of-bag samples to estimate the generalization error'),
        BeamParam('warm_start', bool, False,
                  'When set to True, reuse the solution of the previous call to fit and add more estimators to the '
                  'ensemble, otherwise, just fit a whole new ensemble'),
        BeamParam('n_jobs', int, 1, 'The number of jobs to run in parallel for both `fit` and `predict`'),
        BeamParam('random_state', int, None,
                  'If int, random_state is the seed used by the random number generator; '
                  'If RandomState instance, random_state is the random number generator; '
                  'If None, the random number generator is the RandomState instance used by `np.random`'),
        BeamParam('verbose', int, 0, 'Controls the verbosity of the building process'),
    ]


class PULearnExperimentConfig(PULearnConfig, ExperimentConfig):
    defaults = {'project': 'pu_learn_beam', 'algorithm': 'BeamPUClassifier'}


class PULearnCBExperimentConfig(PULearnExperimentConfig, CatboostConfig):
    pass