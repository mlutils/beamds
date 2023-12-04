import copy

from .utils import TimeoutStopper
from ..utils import find_port, check_type, is_notebook, beam_device
from ..logger import beam_logger as logger
from ..path import beam_path, BeamPath

import ray
from ray.tune import JupyterNotebookReporter
from ray import tune
from functools import partial
from ..experiment import Experiment

import numpy as np
from .core import BeamHPO


class RayHPO(BeamHPO):

    @staticmethod
    def _categorical(param, choices):
        return tune.choice(choices)

    @staticmethod
    def _uniform(param, start, end):
        return tune.uniform(start, end)

    @staticmethod
    def _loguniform(param, start, end):
        return tune.loguniform(start, end)

    @staticmethod
    def _linspace(param, start, end, n_steps, endpoint=True, dtype=None):
        x = np.linspace(start, end, n_steps, endpoint=endpoint)
        step_size = (end - start) / n_steps
        end = end - step_size * (1 - endpoint)

        if np.sum(np.abs(x - np.round(x))) < 1e-8 or dtype in [int, np.int, np.int64, 'int', 'int64']:

            start = int(np.round(start))
            step_size = int(np.round(step_size))
            end = int(np.round(end))

            return tune.qrandint(start, end, step_size)

        return tune.quniform(start, end, (end - start) / n_steps)

    @staticmethod
    def _logspace(param, start, end, n_steps, base=None, dtype=None):

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

    @staticmethod
    def _randn(param, mu, sigma):
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

        search_space = self.get_suggestions()

        if local_dir is None and self.hpo_dir is not None:
            path = beam_path(self.hpo_dir)
            local_dir = path.joinpath('tune')
            local_dir.mkdir(parents=True, exist_ok=True)

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

        runner_tune = partial(self.runner, parallel=parallel)

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

        tuner = tune.Tuner(runner_tune, param_space=search_space,)
        analysis = tuner.fit()

        # analysis = tune.run(runner_tune, config=config, local_dir=local_dir, *args, stop=stop, **kwargs)

        return analysis
