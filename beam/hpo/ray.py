import copy
from functools import partial

import numpy as np
from ray import tune, train
from ray.air import RunConfig
from ray.tune import JupyterNotebookReporter, TuneConfig
from ray.tune.search.optuna import OptunaSearch

from .core import BeamHPO
from .utils import TimeoutStopper
from ..distributed.ray_dispatcher import RayClient
from ..experiment import Experiment
from ..logging import beam_logger as logger
from ..path import beam_path
from ..utils import find_port, is_notebook

from ray.tune.search.sample import Uniform


class LogSpace(Uniform):

    def __init__(self, sampler, x):
        self.sampler = sampler
        self.x = x

    def sample(self, *args, **kwargs):
        i = self.sampler.sample(*args, **kwargs)
        return self.x[i]


class RayHPO(BeamHPO, RayClient):

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

        if np.sum(np.abs(x - np.round(x))) < 1e-8 or dtype in [int, np.int64, 'int', 'int64']:

            start = int(np.round(start))
            step_size = int(np.round(step_size))
            end = int(np.round(end))

            return tune.qrandint(start, end, step_size)

        return tune.quniform(start, end, (end - start) / n_steps)

    @staticmethod
    def _logspace(param, start, end, n_steps, base=None, dtype=None):

        if base is None:
            base = 10

        x = np.logspace(start, end, n_steps, base=base)
        if dtype is not None:
            x = x.astype(dtype)

        return tune.choice(x.tolist())

    def trainable(self, hparams):
        runner_tune = tune.with_resources(
            tune.with_parameters(partial(self.runner)),
            resources={"cpu": hparams.get('cpus-per-trial'),
                       "gpu": hparams.get('gpus-per-trial')}
        )
        return runner_tune

    @staticmethod
    def _randn(param, mu, sigma):
        return tune.qrandn(mu, sigma)

    def runner(self, config):

        hparams = self.generate_hparams(config)

        experiment = Experiment(hparams, hpo='tune', print_hyperparameters=False)

        trial_dir = beam_path(train.get_context().get_trial_dir())
        logger.info(f"Experiment directory is: {experiment.experiment_dir}, see experiment_dir at beam_configuration.pkl"
                    f" in trial directory ({trial_dir})")
        trial_dir.joinpath('beam_configuration.pkl').write({'experiment_dir': experiment.experiment_dir})

        alg, report = self.fit_algorithm(experiment=experiment)

        train.report({report.objective_name: report.best_objective})
        if self.post_train_hook is not None:
            self.post_train_hook(alg=alg, experiment=experiment, hparams=hparams, suggestion=config, results=report)

        self.tracker(algorithm=alg, results=report.results, hparams=hparams, suggestion=config)

    def run(self, *args, runtime_env=None, tune_config_kwargs=None, run_config_kwargs=None,
            init_config_kwargs=None, restore_path=None, restore_config=None, **kwargs):

        hparams = copy.deepcopy(self.hparams)
        hparams.update(kwargs)

        search_space = self.get_suggestions()

        # the ray init configuation

        ray_address = self.hparams.get('ray_address')
        init_config_kwargs = init_config_kwargs or {}

        if ray_address != 'auto':

            dashboard_port = find_port(port=self.hparams.get('dashboard_port'),
                                       get_port_from_beam_port_range=self.hparams.get('get_port_from_beam_port_range'))
            logger.info(f"Opening ray-dashboard on port: {dashboard_port}")
            include_dashboard = self.hparams.get('include_dashboard')

        else:

            dashboard_port = None
            include_dashboard = False

        self.init_ray(address=ray_address, include_dashboard=include_dashboard, dashboard_port=dashboard_port,
                      runtime_env=runtime_env, **init_config_kwargs)

        # the ray tune configuation
        stop = kwargs.get('stop', None)
        train_timeout = hparams.get('train-timeout')
        if train_timeout is not None and train_timeout > 0:
            stop = TimeoutStopper(train_timeout)

        # fix gpu to device 0
        if self.experiment_hparams.get('device') != 'cpu':
            self.experiment_hparams.set('device', 'cuda')

        tune_config_kwargs = tune_config_kwargs or {}
        if 'metric' not in tune_config_kwargs.keys():

            objective_name = self.experiment_hparams.get('objective')
            if objective_name is None:
                if hasattr(self.alg, 'expected_objective_name'):
                    objective_name = self.alg.expected_objective_name(self.experiment_hparams)
            tune_config_kwargs['metric'] = objective_name

        if 'mode' not in tune_config_kwargs.keys():
            mode = self.experiment_hparams.get('optimization-mode')
            tune_config_kwargs['mode'] = self.get_optimization_mode(mode, tune_config_kwargs['metric'])

        if 'progress_reporter' not in tune_config_kwargs.keys() and is_notebook():
            tune_config_kwargs['progress_reporter'] = JupyterNotebookReporter(overwrite=True)

        tune_config_kwargs['num_samples'] = self.hparams.get('n_trials')
        tune_config_kwargs['max_concurrent_trials'] = self.hparams.get('n_jobs', 1)

        # if 'scheduler' not in tune_config_kwargs.keys():
        #     tune_config_kwargs['scheduler'] = ASHAScheduler()

        if 'search_alg' not in tune_config_kwargs.keys():
            metric = tune_config_kwargs['metric']
            mode = tune_config_kwargs['mode']
            tune_config_kwargs['search_alg'] = OptunaSearch(space=None, metric=metric, mode=mode)
            # tune_config_kwargs['search_alg'] = OptunaSearch()

        tune_config = TuneConfig(**tune_config_kwargs)

        run_config_kwargs = run_config_kwargs or {}
        if 'name' not in run_config_kwargs.keys():
            run_config_kwargs['name'] = self.identifier
        if 'stop' not in run_config_kwargs.keys():
            run_config_kwargs['stop'] = stop
        if 'storage_path' not in run_config_kwargs.keys():
            run_config_kwargs['storage_path'] = self.hpo_path
        run_config = RunConfig(**run_config_kwargs)

        logger.info(f"Starting ray-tune hyperparameter optimization process. "
                    f"Results and logs will be stored at {self.hpo_path}")

        tuner = self.tuner(hparams, restore_path=restore_path, restore_config=restore_config,
                           search_space=search_space, tune_config=tune_config, run_config=run_config)
        analysis = tuner.fit()

        return analysis

    def tuner(self, hparams, restore_path=None, restore_config=None, search_space=None, tune_config=None, run_config=None):
        if restore_path:
            restore_config = restore_config or {}
            tuner = tune.Tuner.restore(restore_path, self.trainable(hparams), **restore_config)
        else:
            tuner = tune.Tuner(self.trainable(hparams), param_space=search_space,
                               tune_config=tune_config, run_config=run_config)

        return tuner

    def get_results(self, analysis, *args, **kwargs):
        return self.tuner(*args, **kwargs).get_results()

    def score_table(self, *args, **kwargs):
        res = self.get_results(*args, **kwargs)
        return res.get_dataframe()

    def _best(self, *args, **kwargs):
        best = self.get_results(*args, **kwargs).get_best_result()
        return self.retrive_algorithm_from_path(best.path)

