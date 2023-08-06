import time
import os
import copy

from .utils import find_port, print_beam_hyperparameters, check_type, is_notebook, beam_device
from .logger import beam_logger as logger
from .path import beam_path, BeamPath
import pandas as pd
import ray
from ray.tune import JupyterNotebookReporter
from ray import tune
import optuna
from functools import partial
from .experiment import Experiment, beam_algorithm_generator
from ray.tune.stopper import Stopper
from typing import Union
import datetime
from ._version import __version__
import numpy as np
from scipy.special import erfinv


class TimeoutStopper(Stopper):
    """Stops all trials after a certain timeout.

    This stopper is automatically created when the `time_budget_s`
    argument is passed to `tune.run()`.

    Args:
        timeout: Either a number specifying the timeout in seconds, or
            a `datetime.timedelta` object.
    """

    def __init__(self, timeout: Union[int, float, datetime.timedelta]):
        from datetime import timedelta

        if isinstance(timeout, timedelta):
            self._timeout_seconds = timeout.total_seconds()
        elif isinstance(timeout, (int, float)):
            self._timeout_seconds = timeout
        else:
            raise ValueError(
                "`timeout` parameter has to be either a number or a "
                "`datetime.timedelta` object. Found: {}".format(type(timeout))
            )

        self._budget = self._timeout_seconds

        self.start_time = {}

    def stop_all(self):
        return False

    def __call__(self, trial_id, result):
        now = time.time()

        if trial_id in self.start_time:
            if now - self.start_time[trial_id] >= self._budget:
                logger.info(
                    f"Reached timeout of {self._timeout_seconds} seconds. "
                    f"Stopping this trials."
                )
                return True
        else:
            self.start_time[trial_id] = now

        return False


class Study(object):

    def __init__(self, hparams, Alg=None, Dataset=None, algorithm_generator=None, print_results=False,
                 alg_args=None, alg_kwargs=None, dataset_args=None, dataset_kwargs=None, enable_tqdm=False,
                 print_hyperparameters=True, track_results=False, track_algorithms=False,
                 track_hparams=True, track_suggestion=True, hpo_dir=None):

        logger.info(f"Creating new study (Beam version: {__version__})")
        hparams.reload = False
        hparams.override = False
        hparams.print_results = print_results
        hparams.visualize_weights = False
        hparams.enable_tqdm = enable_tqdm
        hparams.parallel = 0

        exptime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        hparams.identifier = f'{hparams.identifier}_hp_optimization_{exptime}'

        if algorithm_generator is None:
            self.ag = partial(beam_algorithm_generator, Alg=Alg, Dataset=Dataset,
                              alg_args=alg_args, alg_kwargs=alg_kwargs, dataset_args=dataset_args,
                              dataset_kwargs=dataset_kwargs)
        else:
            self.ag = algorithm_generator
        self.hparams = hparams
        self.device = beam_device(hparams.device)
        self.conf = {}

        if print_hyperparameters:
            print_beam_hyperparameters(hparams)

        if hpo_dir is None:
            if self.hparams.hpo_dir is not None:

                root_path = beam_path(self.hparams.hpo_dir)
                hpo_dir = str(root_path.joinpath('hpo_results', self.hparams.project_name, self.hparams.algorithm,
                                                 self.hparams.identifier))

            else:
                root_path = beam_path(self.hparams.root_dir)
                if type(root_path) is BeamPath:
                    hpo_dir = str(root_path.joinpath('hpo_results', self.hparams.project_name, self.hparams.algorithm,
                                                          self.hparams.identifier))

        self.hpo_dir = hpo_dir

        if hpo_dir is None:
            logger.warning("No hpo_dir specified. HPO results will be saved only to each experiment directory.")

        self.experiments_tracker = []
        self.track_results = track_results
        self.track_algorithms = track_algorithms
        self.track_hparams = track_hparams
        self.track_suggestion = track_suggestion

    def optuna_linspace(self, trial, param, start, end, n_steps, endpoint=True,  dtype=None):
        x = np.linspace(start, end, n_steps, endpoint=endpoint)
        if np.sum(np.abs(x - np.round(x))) < 1e-8 or dtype in [int, np.int, np.int64, 'int', 'int64']:
            x = np.round(x).astype(int)
        i = trial.suggest_int(param, 0, len(x) - 1)
        return x[i]

    def optuna_logspace(self, trial, param, start, end, n_steps,base=None, dtype=None):
        x = np.logspace(start, end, n_steps, base=base)
        if np.sum(np.abs(x - np.round(x))) < 1e-8 or dtype in [int, np.int, np.int64, 'int', 'int64']:
            x = np.round(x).astype(int)
        i = trial.suggest_int(param, 0, len(x) - 1)
        return x[i]

    def optuna_uniform(self, trial, param, start, end):
        return trial.suggest_uniform(param, start, end)

    def optuna_loguniform(self, trial, param, start, end):
        return trial.suggest_loguniform(param, start, end)

    def optuna_categorical(self, trial, param, choices):
        return trial.suggest_categorical(param, choices)

    def optuna_randn(self, trial, param, mu, sigma):
        x = trial.suggest_uniform(param, 0, 1)
        return mu + sigma * np.sqrt(2) * erfinv(2 * x - 1)

    def tune_categorical(self, trial, param, choices):
        return tune.choice(choices)

    def tune_uniform(self, trial, param, start, end):
        return tune.uniform(start, end)

    def tune_loguniform(self, trial, param, start, end):
        return tune.loguniform(start, end)

    def tune_linspace(self, trial, param, start, end, n_steps, endpoint=True, dtype=None):
        x = np.linspace(start, end, n_steps, endpoint=endpoint)
        step_size = (end - start) / n_steps
        end = end - step_size * (1 - endpoint)

        if np.sum(np.abs(x - np.round(x))) < 1e-8 or dtype in [int, np.int, np.int64, 'int', 'int64']:

            start = int(np.round(start))
            step_size = int(np.round(step_size))
            end = int(np.round(end))

            return tune.qrandint(start, end, step_size)

        return tune.quniform(start, end, (end - start) / n_steps)

    def tune_logspace(self, trial, param, start, end, n_steps, base=None, dtype=None):

        if base is None:
            base = 10

        emin = base ** start
        emax = base ** end

        x = np.logspace(start, end, n_steps, base=base)

        if np.sum(np.abs(x - np.round(x))) < 1e-8 or dtype in [int, np.int, np.int64, 'int', 'int64']:
            base = int(x[1] / x[0])
            return tune.lograndint(int(emin), int(emax), base=base)

        step_size = (x[1] / x[0]) ** ( (end - start) / n_steps )
        return tune.qloguniform(emin, emax, step_size, base=base)

    def tune_randn(self, trial, param, mu, sigma):
        return tune.qrandn(mu, sigma)

    def uniform(self, param, *args, **kwargs):
        self.conf[param.strip('-').replace('-', '_')] = {'func': 'uniform', 'args': args, 'kwargs': kwargs}

    def loguniform(self, param, *args, **kwargs):
        self.conf[param.strip('-').replace('-', '_')] = {'func': 'loguniform', 'args': args, 'kwargs': kwargs}

    def choice(self, param, *args, **kwargs):
        self.conf[param.strip('-').replace('-', '_')] = {'func': 'choice', 'args': args, 'kwargs': kwargs}

    def quniform(self, param, *args, **kwargs):
        self.conf[param.strip('-').replace('-', '_')] = {'func': 'quniform', 'args': args, 'kwargs': kwargs}

    def qloguniform(self, param, *args, **kwargs):
        self.conf[param.strip('-').replace('-', '_')] = {'func': 'qloguniform', 'args': args, 'kwargs': kwargs}

    def randn(self, param, *args, **kwargs):
        self.conf[param.strip('-').replace('-', '_')] = {'func': 'randn', 'args': args, 'kwargs': kwargs}

    def qrandn(self, param, *args, **kwargs):
        self.conf[param.strip('-').replace('-', '_')] = {'func': 'qrandn', 'args': args, 'kwargs': kwargs}

    def randint(self, param, *args, **kwargs):
        self.conf[param.strip('-').replace('-', '_')] = {'func': 'randint', 'args': args, 'kwargs': kwargs}

    def qrandint(self, param, *args, **kwargs):
        self.conf[param.strip('-').replace('-', '_')] = {'func': 'qrandint', 'args': args, 'kwargs': kwargs}

    def lograndint(self, param, *args, **kwargs):
        self.conf[param.strip('-').replace('-', '_')] = {'func': 'lograndint', 'args': args, 'kwargs': kwargs}

    def qlograndint(self, param, *args, **kwargs):
        self.conf[param.strip('-').replace('-', '_')] = {'func': 'qlograndint', 'args': args, 'kwargs': kwargs}

    def grid_search(self, param, *args, **kwargs):
        self.conf[param.strip('-').replace('-', '_')] = {'func': 'grid_search', 'args': args, 'kwargs': kwargs}

    def sample_from(self, param, *args, **kwargs):
        self.conf[param.strip('-').replace('-', '_')] = {'func': 'sample_from', 'args': args, 'kwargs': kwargs}

    def categorical(self, param, *args, **kwargs):
        self.conf[param.strip('-').replace('-', '_')] = {'func': 'categorical', 'args': args, 'kwargs': kwargs}

    def discrete_uniform(self, param, *args, **kwargs):
        self.conf[param.strip('-').replace('-', '_')] = {'func': 'discrete_uniform', 'args': args, 'kwargs': kwargs}

    def float(self, param, *args, **kwargs):
        self.conf[param.strip('-').replace('-', '_')] = {'func': 'float', 'args': args, 'kwargs': kwargs}

    def int(self, param, *args, **kwargs):
        self.conf[param.strip('-').replace('-', '_')] = {'func': 'int', 'args': args, 'kwargs': kwargs}

    def tracker(self, algorithm=None, results=None, hparams=None, suggestion=None):

        tracker = {}

        if algorithm is not None and self.track_algorithms:
            tracker['algorithm'] = algorithm

        if results is not None and self.track_results:
            tracker['results'] = results

        if hparams is not None and self.track_hparams:
            tracker['hparams'] = hparams

        if suggestion is not None and self.track_suggestion:
            tracker['suggestion'] = suggestion

        if len(tracker):
            self.experiments_tracker.append(tracker)

        if self.hpo_dir is not None:
            path = beam_path(self.hpo_dir).joinpath('tracker')
            path.mkdir(parents=True, exist_ok=True)
            path.joinpath('tracker.pkl').write(tracker)

    @staticmethod
    def init_ray(runtime_env=None, dashboard_port=None, include_dashboard=True):

        ray.init(runtime_env=runtime_env, dashboard_port=dashboard_port,
                 include_dashboard=include_dashboard, dashboard_host="0.0.0.0")

    def runner_tune(self, config, parallel=None):

        hparams = copy.deepcopy(self.hparams)

        for k, v in config.items():
            setattr(hparams.replace('-', '_'), k, v)

        # set device to 0 (ray exposes only a single device
        hparams.device = '0'
        if parallel is not None:
            hparams.parallel = parallel

        experiment = Experiment(hparams, hpo='tune', print_hyperparameters=False)
        alg, results = experiment(self.ag, return_results=True)

        self.tracker(algorithm=alg, results=results, hparams=hparams, suggestion=config)

        if 'objective' in results:
            if type('objective') is tuple:
                return results['objective']
            elif isinstance(results['objective'], dict):
                tune.report(**results['objective'])
            else:
                return results['objective']

    def runner_optuna(self, trial, suggest):

        config = suggest(trial)

        logger.info('Next Hyperparameter suggestion:')
        for k, v in config.items():
            logger.info(k + ': ' + str(v))

        hparams = copy.deepcopy(self.hparams)

        for k, v in config.items():
            setattr(hparams, k.replace('-', '_'), v)

        experiment = Experiment(hparams, hpo='optuna', trial=trial, print_hyperparameters=False)
        alg, results = experiment(self.ag, return_results=True)

        self.tracker(algorithm=alg, results=results, hparams=hparams, suggestion=config)

        if 'objective' in results:
            if type('objective') is tuple:
                return results['objective']
            elif isinstance(results['objective'], dict):
                tune.report(**results['objective'])
            else:
                return results['objective']

    def tune(self, *args, config=None, timeout=0, runtime_env=None, dashboard_port=None,
             get_port_from_beam_port_range=True, include_dashboard=True, local_dir=None, **kwargs):

        # TODO: move to tune.Tuner and tuner.run()

        if config is None:
            config = {}

        if local_dir is None and self.hpo_dir is not None:
            path = beam_path(self.hpo_dir)
            local_dir = path.joinpath('tune')
            local_dir.mkdir(parents=True, exist_ok=True)

        base_conf = {k: getattr(tune, v['func'])(*v['args'], **v['kwargs']) for k, v in self.conf.items()}
        config.update(base_conf)

        ray.shutdown()

        dashboard_port = find_port(port=dashboard_port, get_port_from_beam_port_range=get_port_from_beam_port_range)
        if dashboard_port is None:
            return

        logger.info(f"Opening ray-dashboard on port: {dashboard_port}")
        self.init_ray(runtime_env=runtime_env, dashboard_port=int(dashboard_port), include_dashboard=include_dashboard)

        if 'stop' in kwargs:
            stop = kwargs['stop']
        else:
            stop = None
            if timeout > 0:
                stop = TimeoutStopper(timeout)

        parallel = None
        if 'resources_per_trial' in kwargs and 'gpu' in kwargs['resources_per_trial']:
            gpus = kwargs['resources_per_trial']['gpu']
            if 'cpu' not in self.device.type:
                parallel = gpus

        runner_tune = partial(self.runner_tune, parallel=parallel)

        logger.info(f"Starting ray-tune hyperparameter optimization process. Results and logs will be stored at {local_dir}")

        if 'metric' not in kwargs.keys():
            if 'objective' in self.hparams and self.hparams.objective is not None:
                kwargs['metric'] = self.hparams.objective
            else:
                kwargs['metric'] = 'objective'
        if 'mode' not in kwargs.keys():
            kwargs['mode'] = 'max'

        if 'progress_reporter' not in kwargs.keys() and is_notebook():
            kwargs['progress_reporter'] = JupyterNotebookReporter(overwrite=True)

        analysis = tune.run(runner_tune, config=config, local_dir=local_dir, *args, stop=stop, **kwargs)

        return analysis

    def grid_search(self, load_study=False, storage=None, sampler=None, pruner=None, study_name=None, direction=None,
                    load_if_exists=False, directions=None, sync_parameters=None, explode_parameters=None, **kwargs):

        df_sync = pd.DataFrame(sync_parameters)
        df_explode = pd.DataFrame([explode_parameters])
        for c in list(df_explode.columns):
            df_explode = df_explode.explode(c)

        if sync_parameters is None:
            df = df_explode
        elif explode_parameters is None:
            df = df_sync
        else:
            df = df_sync.merge(df_explode, how='cross')

        df = df.reset_index(drop=True)
        n_trials = len(df)

        if not 'cpu' in self.device.type:
            if 'n_jobs' not in kwargs or kwargs['n_jobs'] != 1:
                logger.warning("Optuna does not support multi-GPU jobs. Setting number of parallel jobs to 1")
            kwargs['n_jobs'] = 1

        if study_name is None:
            study_name = f'{self.hparams.project_name}/{self.hparams.algorithm}/{self.hparams.identifier}'

        if direction is None:
            direction = 'maximize'

        if storage is None:
            if self.hpo_dir is not None:

                path = beam_path(self.hpo_dir)
                path.joinpath('optuna').mkdir(parents=True, exist_ok=True)

                storage = f'sqlite:///{self.hpo_dir}/{study_name}.db'

        if load_study:
            study = optuna.create_study(storage=storage, sampler=sampler, pruner=pruner, study_name=study_name)
        else:
            study = optuna.create_study(storage=storage, sampler=sampler, pruner=pruner, study_name=study_name,
                                        direction=direction, load_if_exists=load_if_exists, directions=directions)

        for it in df.iterrows():
            study.enqueue_trial(it[1].to_dict())

        def dummy_suggest(trial):
            config = {}
            for k, v in it[1].items():
                v_type = check_type(v)
                if v_type.element == 'int':
                    config[k] = trial.suggest_int(k, 0, 1)
                elif v_type.element == 'str':
                    config[k] = trial.suggest_categorical(k, ['a', 'b'])
                else:
                    config[k] = trial.suggest_float(k, 0, 1)

            return config

        runner = partial(self.runner_optuna, suggest=dummy_suggest)
        study.optimize(runner, n_trials=n_trials, **kwargs)

        return study

    def optuna(self, suggest=None, load_study=False, storage=None, sampler=None, pruner=None, study_name=None, direction=None,
               load_if_exists=False, directions=None, *args, **kwargs):

        if suggest is None:
            suggest = lambda trial: {k: getattr(trial, f'suggest_{v["func"]}')(k, *v['args'], **v['kwargs'])
                        for k, v in self.conf.items()}

        if not 'cpu' in self.device.type:
            if 'n_jobs' not in kwargs or kwargs['n_jobs'] != 1:
                logger.warning("Optuna does not support multi-GPU jobs. Setting number of parallel jobs to 1")
            kwargs['n_jobs'] = 1

        if direction is None:
            direction = 'maximize'

        if study_name is None:
            study_name = f'{self.hparams.project_name}/{self.hparams.algorithm}/{self.hparams.identifier}'

        if storage is None:
            if self.hpo_dir is not None:

                path = beam_path(self.hpo_dir)
                path.joinpath('optuna').mkdir(parents=True, exist_ok=True)

                # storage = f'sqlite:///{self.hpo_dir}/{study_name}.db'
                # logger.info(f"Using {storage} as storage to store the trials results")

        runner = partial(self.runner_optuna, suggest=suggest)

        if load_study:
            study = optuna.load_study(storage=storage, sampler=sampler, pruner=pruner, study_name=study_name)
        else:
            study = optuna.create_study(storage=storage, sampler=sampler, pruner=pruner, study_name=study_name,
                                        direction=direction, load_if_exists=load_if_exists, directions=directions)

        study.optimize(runner, *args, gc_after_trial=True, **kwargs)

        return study