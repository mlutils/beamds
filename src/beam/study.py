import re
import sys
import time
import numpy as np
import os
import warnings

from torch.utils.tensorboard import SummaryWriter
from shutil import copytree
import torch
import copy
import shutil
from collections import defaultdict
from .utils import include_patterns, logger, find_port, print_beam_hyperparameters, check_type
import pandas as pd
import torch.multiprocessing as mp
from .utils import setup, cleanup, set_seed, find_free_port, check_if_port_is_available, is_notebook
import torch.distributed as dist
import ray
from ray.tune import JupyterNotebookReporter
from ray import tune
import optuna
from functools import partial
from .experiment import Experiment, beam_algorithm_generator
from ray.tune.stopper import Stopper
from typing import Union
import datetime


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
                 track_hparams=True, track_suggestion=True):

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

        if print_hyperparameters:
            print_beam_hyperparameters(hparams)

        self.ray_logs = os.path.join(self.hparams.root_dir, 'ray_results', self.hparams.project_name,
                                     self.hparams.algorithm, self.hparams.identifier)

        self.experiments_tracker = []
        self.track_results = track_results
        self.track_algorithms = track_algorithms
        self.track_hparams = track_hparams
        self.track_suggestion = track_suggestion

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

    @staticmethod
    def init_ray(runtime_env=None, dashboard_port=None, include_dashboard=True):

        ray.init(runtime_env=runtime_env, dashboard_port=dashboard_port,
                 include_dashboard=include_dashboard, dashboard_host="0.0.0.0")

    def runner_tune(self, config, parallel=None):

        hparams = copy.deepcopy(self.hparams)

        for k, v in config.items():
            setattr(hparams, k, v)

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
            setattr(hparams, k, v)

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

    def tune(self, config, *args, timeout=0, runtime_env=None, dashboard_port=None,
             get_port_from_beam_port_range=True, include_dashboard=True, **kwargs):

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
            if 'cpu' not in self.hparams.device:
                parallel = gpus

        runner_tune = partial(self.runner_tune, parallel=parallel)

        logger.info(f"Starting ray-tune hyperparameter optimization process. Results and logs will be stored at {self.ray_logs}")

        if 'metric' not in kwargs.keys():
            if 'objective' in self.hparams:
                kwargs['metric'] = self.hparams['objective']
            else:
                kwargs['metric'] = 'objective'
        if 'mode' not in kwargs.keys():
            kwargs['mode'] = 'max'

        if 'progress_reporter' not in kwargs.keys() and is_notebook():
            kwargs['progress_reporter'] = JupyterNotebookReporter(overwrite=True)

        analysis = tune.run(runner_tune, config=config, local_dir=self.ray_logs, *args, stop=stop, **kwargs)

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

        if not 'cpu' in self.hparams.device:
            if 'n_jobs' not in kwargs or kwargs['n_jobs'] != 1:
                logger.warning("Optuna does not support multi-GPU jobs. Setting number of parallel jobs to 1")
            kwargs['n_jobs'] = 1

        if study_name is None:
            study_name = f'{self.hparams.project_name}/{self.hparams.algorithm}/{self.hparams.identifier}'

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

    def optuna(self, suggest, load_study=False, storage=None, sampler=None, pruner=None, study_name=None, direction=None,
               load_if_exists=False, directions=None, *args, **kwargs):

        if not 'cpu' in self.hparams.device:
            if 'n_jobs' not in kwargs or kwargs['n_jobs'] != 1:
                logger.warning("Optuna does not support multi-GPU jobs. Setting number of parallel jobs to 1")
            kwargs['n_jobs'] = 1

        if study_name is None:
            study_name = f'{self.hparams.project_name}/{self.hparams.algorithm}/{self.hparams.identifier}'

        runner = partial(self.runner_optuna, suggest=suggest)

        if load_study:
            study = optuna.load_study(storage=storage, sampler=sampler, pruner=pruner, study_name=study_name)
        else:
            study = optuna.create_study(storage=storage, sampler=sampler, pruner=pruner, study_name=study_name,
                                        direction=direction, load_if_exists=load_if_exists, directions=directions)

        study.optimize(runner, *args, **kwargs)

        return study