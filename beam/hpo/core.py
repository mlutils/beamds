import copy
import time
from functools import partial

from .params import HPOConfig
from .._version import __version__
from ..algorithm import Algorithm, NeuralAlgorithm
from ..config import print_beam_hyperparameters
from ..experiment import nn_algorithm_generator, simple_algorithm_generator
from ..logging import beam_logger as logger
from ..path import beam_path, BeamPath
from ..processor import Processor
from ..utils import beam_device


class BeamHPO(Processor):

    def __init__(self, hparams, *args, _reload_study=False, hpo_config=None,
                 alg=None, dataset=None, algorithm_generator=None, alg_args=None,
                 alg_kwargs=None, dataset_args=None, dataset_kwargs=None, post_train_hook=None,
                 **kwargs):

        if hpo_config is None:
            hpo_config = HPOConfig(**kwargs)

        super().__init__(*args, hparams=hpo_config, _config_scheme=HPOConfig,  **kwargs)

        self.is_reloaded = _reload_study
        if _reload_study:
            logger.info(f"Reloading existing study from {self.hpo_path} (Beam version: {__version__})")
        else:
            hpo_path = self.hparams.get('hpo_path')
            if hpo_path is not None:

                root_path = beam_path(hpo_path)
                hpo_path = str(root_path.joinpath(hparams.project_name,hparams.algorithm,
                                                  hparams.identifier))

            else:
                logger.warning("No hpo_path specified. HPO results will be saved only to each experiment directory.")
                root_path = beam_path(self.experiment_hparams.get('logs_path'))
                if type(root_path) is BeamPath:
                    hpo_path = str(root_path.joinpath('hpo', hparams.project_name, hparams.algorithm,
                                                      hparams.identifier))

            self.hpo_path = beam_path(hpo_path)
            logger.info(f"Creating new study at {self.hpo_path} (Beam version: {__version__})")
            self.hpo_path.mkdir()
            self.hpo_path.joinpath('hpo_config.yaml').write(self.hparams)

        self.experiment_hparams = hparams
        self.experiment_hparams.set('reload', False)
        self.experiment_hparams.set('override', False)
        self.experiment_hparams.set('print_results', self.hparams.get('print_results'))
        self.experiment_hparams.set('visualize_weights', False)
        self.experiment_hparams.set('enable_tqdm', self.hparams.get('enable_tqdm', False))
        # self.experiment_hparams.set('n_gpus', 0)

        exptime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.identifier = f'{self.experiment_hparams.identifier}_hp_optimization_{exptime}'
        self.experiment_hparams.set('identifier', self.identifier)
        if not self.is_reloaded:
            self.hpo_path.joinpath('base_config.yaml').write(self.experiment_hparams)

        self.alg = alg
        if algorithm_generator is None:

            algorithm_generator = nn_algorithm_generator if isinstance(self.alg, NeuralAlgorithm) \
                else simple_algorithm_generator

            self.ag = partial(algorithm_generator, dataset=dataset,
                              alg_args=alg_args, alg_kwargs=alg_kwargs, dataset_args=dataset_args,
                              dataset_kwargs=dataset_kwargs)
        else:
            self.ag = algorithm_generator
        self.device = beam_device(self.experiment_hparams.device)

        if self.hparams.get('print_hyperparameters'):
            print_beam_hyperparameters(self.experiment_hparams, default_params=HPOConfig(return_defaults=True))

        self.experiments_tracker = []
        self.suggestions = {}
        self.post_train_hook = post_train_hook

    def fit_algorithm(self, experiment):

        if isinstance(self.alg, NeuralAlgorithm) or issubclass(self.alg, NeuralAlgorithm):
            alg, report = experiment.fit(alg=self.alg, algorithm_generator=self.ag, return_results=True,
                                         runner='default')
        else:
            alg, report = experiment.fit(alg=self.alg, algorithm_generator=self.ag, return_results=True,
                                         runner='simple')

        return alg, report

    def get_optimization_mode(self, mode, objective_name):
        return self.alg.get_optimization_mode(mode, objective_name)

    def add_suggestion(self, param, func, *args, **kwargs):
        self.suggestions[param] = {'func': func, 'args': args, 'kwargs': kwargs}

    def linspace(self, param, start, end, n_steps=None, endpoint=None,  dtype=None):
        param = param.replace('-', '_').strip()
        if endpoint is None:
            endpoint = True

        if n_steps is None:
            assert int(end) == end and int(start) == start, \
                "n_steps must be specified if start or end is not an integer"
            n_steps = int(end - start + 1)

        self.suggestions[param] = partial(self._linspace, param=param, start=start, end=end, n_steps=n_steps,
                                          endpoint=endpoint, dtype=dtype)

    def add_parameter(self, param, kind, *args, **kwargs):
        func = getattr(self, kind)
        func(param, *args, **kwargs)

    def logspace(self, param, start, end, n_steps=None, base=None, dtype=None):
        param = param.replace('-', '_').strip()

        if n_steps is None:
            n_steps = int(end - start + 1)

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
            path.mkdir()
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

    def score_table(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def best(self):
        raise NotImplementedError

    def _best(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def retrieve_algorithm(self, path):
        path = beam_path(path)
        conf = path.joinpath('beam_configuration.pkl').read()
        alg = beam_path(conf['experiment_dir']).joinpath('best_model.bmpr').read()
        return alg

    # def run(self, *args, runtime_env=None, tune_config_kwargs=None, run_config_kwargs=None,
    #             init_config_kwargs=None, restore_path=None, restore_config=None, **kwargs):

    # def run(self, suggest=None, load_study=False, storage=None, sampler=None, pruner=None, study_name=None,
    #         direction=None, load_if_exists=False, directions=None, *args, **kwargs):