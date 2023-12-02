import time
import copy

from .utils import TimeoutStopper
from ..utils import find_port, check_type, is_notebook, beam_device
from ..config import print_beam_hyperparameters
from ..logger import beam_logger as logger
from ..path import beam_path, BeamPath
from ..core import Processor
import pandas as pd
import ray
from ray.tune import JupyterNotebookReporter
from ray import tune
import optuna
from functools import partial
from ..experiment import Experiment, beam_algorithm_generator

from .._version import __version__
import numpy as np
from scipy.special import erfinv
from .core import BeamHPO


class OptunaHPO(BeamHPO):

    def linspace(self, trial, param, start, end, n_steps, endpoint=True,  dtype=None):
        x = np.linspace(start, end, n_steps, endpoint=endpoint)
        if np.sum(np.abs(x - np.round(x))) < 1e-8 or dtype in [int, np.int, np.int64, 'int', 'int64']:
            x = np.round(x).astype(int)
        i = trial.suggest_int(param, 0, len(x) - 1)
        return x[i]

    def logspace(self, trial, param, start, end, n_steps,base=None, dtype=None):
        x = np.logspace(start, end, n_steps, base=base)
        if np.sum(np.abs(x - np.round(x))) < 1e-8 or dtype in [int, np.int, np.int64, 'int', 'int64']:
            x = np.round(x).astype(int)
        i = trial.suggest_int(param, 0, len(x) - 1)
        return x[i]

    def uniform(self, trial, param, start, end):
        return trial.suggest_uniform(param, start, end)

    def loguniform(self, trial, param, start, end):
        return trial.suggest_loguniform(param, start, end)

    def categorical(self, trial, param, choices):
        return trial.suggest_categorical(param, choices)

    def randn(self, trial, param, mu, sigma):
        x = trial.suggest_uniform(param, 0, 1)
        return mu + sigma * np.sqrt(2) * erfinv(2 * x - 1)

    def runner(self, trial, suggest):

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

    def run(self, suggest=None, load_study=False, storage=None, sampler=None, pruner=None, study_name=None, direction=None,
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


class RayHPO(BeamHPO):

    def categorical(self, trial, param, choices):
        return tune.choice(choices)

    def uniform(self, trial, param, start, end):
        return tune.uniform(start, end)

    def loguniform(self, trial, param, start, end):
        return tune.loguniform(start, end)

    def linspace(self, trial, param, start, end, n_steps, endpoint=True, dtype=None):
        x = np.linspace(start, end, n_steps, endpoint=endpoint)
        step_size = (end - start) / n_steps
        end = end - step_size * (1 - endpoint)

        if np.sum(np.abs(x - np.round(x))) < 1e-8 or dtype in [int, np.int, np.int64, 'int', 'int64']:

            start = int(np.round(start))
            step_size = int(np.round(step_size))
            end = int(np.round(end))

            return tune.qrandint(start, end, step_size)

        return tune.quniform(start, end, (end - start) / n_steps)

    def logspace(self, trial, param, start, end, n_steps, base=None, dtype=None):

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

    def randn(self, trial, param, mu, sigma):
        return tune.qrandn(mu, sigma)

    @staticmethod
    def init_ray(runtime_env=None, dashboard_port=None, include_dashboard=True):

        ray.init(runtime_env=runtime_env, dashboard_port=dashboard_port,
                 include_dashboard=include_dashboard, dashboard_host="0.0.0.0")

    def runner(self, config, parallel=None):

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

    def run(self, *args, config=None, timeout=0, runtime_env=None, dashboard_port=None,
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
