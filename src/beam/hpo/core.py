import copy
import time

from .params import HPOConfig
from ..utils import find_port, check_type, is_notebook, beam_device
from ..config import print_beam_hyperparameters
from ..logger import beam_logger as logger
from ..path import beam_path, BeamPath
from ..core import Processor
from ..algorithm import NeuralAlgorithm
from functools import partial
from ..experiment import beam_algorithm_generator

from .._version import __version__


class BeamHPO(Processor):

    def __init__(self, hparams, *args, hpo_config=None,
                 alg=None, dataset=None, algorithm_generator=None, alg_args=None,
                 alg_kwargs=None, dataset_args=None, dataset_kwargs=None, post_train_hook=None, **kwargs):

        if hpo_config is None:
            hpo_config = HPOConfig(**kwargs)

        super().__init__(*args, hparams=hpo_config)
        logger.info(f"Creating new study (Beam version: {__version__})")

        self.experiment_hparams = hparams

        self.experiment_hparams.set('reload', False)
        self.experiment_hparams.set('override', False)
        self.experiment_hparams.set('print_results', self.hparams.get('print_results'))
        self.experiment_hparams.set('visualize_weights', False)
        self.experiment_hparams.set('enable_tqdm', self.hparams.get('enable_tqdm', False))
        self.experiment_hparams.set('n_gpus', 0)

        exptime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.identifier = f'{self.experiment_hparams.identifier}_hp_optimization_{exptime}'
        self.experiment_hparams.set('identifier', self.identifier)

        self.alg = alg
        if algorithm_generator is None:
            self.ag = partial(beam_algorithm_generator, dataset=dataset,
                              alg_args=alg_args, alg_kwargs=alg_kwargs, dataset_args=dataset_args,
                              dataset_kwargs=dataset_kwargs)
        else:
            self.ag = algorithm_generator
        self.device = beam_device(self.experiment_hparams.device)

        if self.hparams.get('print_hyperparameters'):
            print_beam_hyperparameters(self.experiment_hparams, default_params=HPOConfig(return_defaults=True))

        hpo_path = self.hparams.get('hpo_path')
        if hpo_path is not None:

            root_path = beam_path(hpo_path)
            hpo_path = str(root_path.joinpath(self.experiment_hparams.project_name,
                                              self.experiment_hparams.algorithm,
                                              self.experiment_hparams.identifier))

        else:
            logger.warning("No hpo_path specified. HPO results will be saved only to each experiment directory.")
            root_path = beam_path(self.experiment_hparams.get('logs_path'))
            if type(root_path) is BeamPath:
                hpo_path = str(root_path.joinpath('hpo', self.experiment_hparams.project_name,
                                                  self.experiment_hparams.algorithm,
                                                  self.experiment_hparams.identifier))

        self.hpo_path = hpo_path
        self.experiments_tracker = []
        self.suggestions = {}
        self.post_train_hook = post_train_hook

    @staticmethod
    def get_optimization_mode(mode, objective_name):
        return NeuralAlgorithm.get_optimization_mode(mode, objective_name)

    def add_suggestion(self, param, func, *args, **kwargs):
        self.suggestions[param] = {'func': func, 'args': args, 'kwargs': kwargs}

    def linspace(self, param, start, end, n_steps, endpoint=True,  dtype=None):
        param = param.replace('-', '_').strip()
        self.suggestions[param] = partial(self._linspace, param=param, start=start, end=end, n_steps=n_steps,
                       endpoint=endpoint, dtype=dtype)

    def logspace(self, param, start, end, n_steps, base=None, dtype=None):
        param = param.replace('-', '_').strip()
        self.suggestions[param] = partial(self._logspace, param=param, start=start, end=end,
                                          n_steps=n_steps, base=base, dtype=dtype)

    def uniform(self, param, start, end):
        param = param.replace('-', '_').strip()
        self.suggestions[param] = partial(self._uniform, param=param, start=start, end=end)

    def loguniform(self, param, start, end):
        param = param.replace('-', '_').strip()
        self.suggestions[param] = partial(self._loguniform, param=param, start=start, end=end)

    def categorical(self, param, choices):
        param = param.replace('-', '_').strip()
        self.suggestions[param] = partial(self._categorical, param=param, choices=choices)

    def randn(self, param, mu, sigma):
        param = param.replace('-', '_').strip()
        self.suggestions[param] = partial(self._randn, param=param, mu=mu, sigma=sigma)

    def get_suggestions(self, *args, **kwargs):

        config = {}
        for k, v in self.suggestions.items():
            config[k] = v(*args, **kwargs)

        return config

    @staticmethod
    def _linspace(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _logspace(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _uniform(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _loguniform(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _categorical(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _randn(*args, **kwargs):
        raise NotImplementedError

    def choice(self, param, *args, **kwargs):
        raise NotImplementedError

    def quniform(self, param, *args, **kwargs):
        raise NotImplementedError

    def qloguniform(self, param, *args, **kwargs):
        raise NotImplementedError

    def qrandn(self, param, *args, **kwargs):
        raise NotImplementedError

    def randint(self, param, *args, **kwargs):
        raise NotImplementedError

    def qrandint(self, param, *args, **kwargs):
        raise NotImplementedError

    def lograndint(self, param, *args, **kwargs):
        raise NotImplementedError

    def qlograndint(self, param, *args, **kwargs):
        raise NotImplementedError

    def grid_search(self, param, *args, **kwargs):
        raise NotImplementedError

    def sample_from(self, param, *args, **kwargs):
        raise NotImplementedError

    def discrete_uniform(self, param, *args, **kwargs):
        raise NotImplementedError

    def float(self, param, *args, **kwargs):
        raise NotImplementedError

    def int(self, param, *args, **kwargs):
        raise NotImplementedError

    def tracker(self, algorithm=None, results=None, hparams=None, suggestion=None):

        tracker = {}

        if algorithm is not None and self.hparams.get('track_algorithms'):
            tracker['algorithm'] = algorithm

        if results is not None and self.hparams.get('track_results'):
            tracker['results'] = results

        if hparams is not None and self.hparams.get('track_hparams'):
            tracker['hparams'] = hparams

        if suggestion is not None and self.hparams.get('track_suggestion'):
            tracker['suggestion'] = suggestion

        if len(tracker):
            self.experiments_tracker.append(tracker)

        if self.hpo_path is not None:
            path = beam_path(self.hpo_path).joinpath('tracker')
            path.mkdir(parents=True, exist_ok=True)
            path.joinpath('tracker.pkl').write(tracker)

    def generate_hparams(self, config):

        hparams = copy.deepcopy(self.experiment_hparams)
        if self.experiment_hparams.get('print_hyperparameters'):
            logger.info('Next Hyperparameter suggestion:')
            for k, v in config.items():
                logger.info(k + ': ' + str(v))

        hparams.update(config)
        return hparams

    def runner(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError
