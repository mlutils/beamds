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


class BeamHPO(Processor):

    def __init__(self, hparams, *args,
                 alg=None, dataset=None, algorithm_generator=None, print_results=False, alg_args=None,
                 alg_kwargs=None, dataset_args=None, dataset_kwargs=None,
                 enable_tqdm=False, print_hyperparameters=True,
                 track_results=False, track_algorithms=False, track_hparams=True, track_suggestion=True, hpo_dir=None,
                 **kwargs):

        super().__init__(*args, hparams=hparams, **kwargs)
        logger.info(f"Creating new study (Beam version: {__version__})")

        self.hparams.set('reload', False, model=False)
        self.hparams.set('override', False, model=False)
        self.hparams.set('print_results', print_results, model=False)
        self.hparams.set('visualize_weights', False, model=False)
        self.hparams.set('enable_tqdm', enable_tqdm, model=False)
        self.hparams.set('parallel', 0, model=False)

        exptime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.hparams.set('identifier', f'{self.hparams.identifier}_hp_optimization_{exptime}', model=False)

        if algorithm_generator is None:
            self.ag = partial(beam_algorithm_generator, alg=alg, dataset=dataset,
                              alg_args=alg_args, alg_kwargs=alg_kwargs, dataset_args=dataset_args,
                              dataset_kwargs=dataset_kwargs)
        else:
            self.ag = algorithm_generator
        self.device = beam_device(hparams.device)

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

    def uniform(self, param, *args, **kwargs):
        raise NotImplementedError
    def loguniform(self, param, *args, **kwargs):
        raise NotImplementedError
    def choice(self, param, *args, **kwargs):
        raise NotImplementedError
    def quniform(self, param, *args, **kwargs):
        raise NotImplementedError
    def qloguniform(self, param, *args, **kwargs):
        raise NotImplementedError
    def randn(self, param, *args, **kwargs):
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
    def categorical(self, param, *args, **kwargs):
        raise NotImplementedError
    def discrete_uniform(self, param, *args, **kwargs):
        raise NotImplementedError
    def float(self, param, *args, **kwargs):
        raise NotImplementedError
    def int(self, param, *args, **kwargs):
        raise NotImplementedError
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

    def runner(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError
