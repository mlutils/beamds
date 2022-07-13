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
from .utils import include_patterns, logger
import pandas as pd
import torch.multiprocessing as mp
from .utils import setup, cleanup, set_seed, find_free_port, check_if_port_is_available, is_notebook
import torch.distributed as dist
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
            taken = now - self.start_time[trial_id]
            self._budget -= taken
        else:
            self.start_time[trial_id] = now

        if self._budget <= 0:
            logger.info(
                f"Reached timeout of {self._timeout_seconds} seconds. "
                f"Stopping this trials."
            )
            return True
        return False


class Study(object):

    def __init__(self, args, Alg=None, Dataset=None, algorithm_generator=None, print_results=False, enable_tqdm=False):

        args.reload = False
        args.override = False
        args.print_results = print_results
        args.visualize_weights = False
        args.enable_tqdm = enable_tqdm
        args.parallel = 0

        exptime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        args.identifier = f'{args.identifier}_hp_optimization_{exptime}'

        if algorithm_generator is None:
            self.ag = partial(beam_algorithm_generator, Alg=Alg, Dataset=Dataset)
        else:
            self.ag = algorithm_generator
        self.args = args

        logger.info('Hyperparameter Optimization')
        logger.info(f"beam project: {args.project_name}")
        logger.info('Experiment Hyperparameters')

        for k, v in vars(args).items():
            logger.info(k + ': ' + str(v))

    def runner_tune(self, config):

        args = copy.deepcopy(self.args)

        for k, v in config.items():
            setattr(args, k, v)

        experiment = Experiment(args, results_names='objective', hpo='tune', print_hyperparameters=False)
        alg, results = experiment(self.ag, return_results=True)

        if 'objective' in results:
            if type('objective') is tuple:
                return results['objective']
            elif issubclass(type(results['objective']), dict):
                tune.report(**results['objective'])
            else:
                return results['objective']

    def runner_optuna(self, trial, suggest):

        config = suggest(trial)

        logger.info('Next Hyperparameter suggestion:')
        for k, v in config.items():
            logger.info(k + ': ' + str(v))

        args = copy.deepcopy(self.args)

        for k, v in config.items():
            setattr(args, k, v)

        experiment = Experiment(args, hpo='optuna', results_names='objective',
                                trial=trial, print_hyperparameters=False)
        alg, results = experiment(self.ag, return_results=True)

        if 'objective' in results:
            if type('objective') is tuple:
                return results['objective']
            elif issubclass(type(results['objective']), dict):
                tune.report(**results['objective'])
            else:
                return results['objective']

    def tune(self, config, *args, timeout=0, **kwargs):

        if 'stop' in kwargs:
            stop = kwargs['stop']
        else:
            stop = None
            if timeout > 0:
                stop = TimeoutStopper(timeout)

        analysis = tune.run(self.runner_tune, config=config, *args, stop=stop, **kwargs)

        return analysis

    def optuna(self, suggest, storage=None, sampler=None, pruner=None, study_name=None, direction=None,
               load_if_exists=False, directions=None, *args, **kwargs):

        if study_name is None:
            study_name = f'{self.args.project_name}/{self.args.algorithm}/{self.args.identifier}'

        runner = partial(self.runner_optuna, suggest=suggest)
        study = optuna.create_study(storage=storage, sampler=sampler, pruner=pruner, study_name=study_name,
                                    direction=direction, load_if_exists=load_if_exists, directions=directions)
        study.optimize(runner, *args, **kwargs)

        return study