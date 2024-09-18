from functools import partial

import numpy as np
import optuna
import pandas as pd
from scipy.special import erfinv

from .core import BeamHPO
from ..experiment import Experiment
from ..logging import beam_logger as logger
from ..path import beam_path
from ..utils import check_type
from ..type import Types


class OptunaBase:

    @staticmethod
    def _linspace(trial, param, start, end, n_steps, endpoint=True,  dtype=None):
        x = np.linspace(start, end, n_steps, endpoint=endpoint)
        if np.sum(np.abs(x - np.round(x))) < 1e-8 or dtype in [int, np.int, np.int64, 'int', 'int64']:
            x = np.round(x).astype(int)
        i = trial.suggest_int(param, 0, len(x) - 1)
        return x[i]

    @staticmethod
    def _logspace(trial, param, start, end, n_steps, base=None, dtype=None):
        if base is None:
            base = 10
        x = np.logspace(start, end, n_steps, base=base)
        if np.sum(np.abs(x - np.round(x))) < 1e-8 or dtype in [int, np.int64, 'int', 'int64']:
            x = np.round(x).astype(int)
        i = trial.suggest_int(param, 0, len(x) - 1)
        return x[i]

    @staticmethod
    def _uniform(trial, param, start, end):
        return trial.suggest_uniform(param, start, end)

    @staticmethod
    def _loguniform(trial, param, start, end):
        return trial.suggest_loguniform(param, start, end)

    @staticmethod
    def _categorical(trial, param, choices):
        return trial.suggest_categorical(param, choices)

    @staticmethod
    def _randn(trial, param, mu, sigma):
        x = trial.suggest_uniform(param, 0, 1)
        return mu + sigma * np.sqrt(2) * erfinv(2 * x - 1)


class OptunaHPO(OptunaBase, BeamHPO):

    def runner(self, trial, suggest):

        config = suggest(trial)
        hparams = self.generate_hparams(config)

        experiment = Experiment(hparams, hpo='optuna', trial=trial, print_hyperparameters=False)

        logger.info(f"Experiment directory is: {experiment.experiment_dir}, see experiment_dir attr in trial logs")
        trial.set_user_attr("experiment_dir", experiment.experiment_dir.as_uri())

        alg, report = self.fit_algorithm(experiment=experiment)

        if self.post_train_hook is not None:
            self.post_train_hook(alg=alg, experiment=experiment, hparams=hparams, suggestion=config, results=report)

        self.tracker(algorithm=alg, results=report, hparams=hparams, suggestion=config)

        return report.objective

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
            if self.hpo_path is not None:

                path = beam_path(self.hpo_path)
                path.joinpath('optuna').mkdir(parents=True, exist_ok=True)

                storage = f'sqlite:///{self.hpo_path}/{study_name}.db'

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
                if v_type.element == Types.int:
                    config[k] = trial.suggest_int(k, 0, 1)
                elif v_type.element == Types.str:
                    config[k] = trial.suggest_categorical(k, ['a', 'b'])
                else:
                    config[k] = trial.suggest_float(k, 0, 1)

            return config

        runner = partial(self.runner, suggest=dummy_suggest)
        study.optimize(runner, n_trials=n_trials, **kwargs)

        return study

    def run(self, suggest=None, load_study=False, storage=None, sampler=None, pruner=None, study_name=None,
            direction=None, load_if_exists=False, directions=None, *args, **kwargs):

        if suggest is None:
            suggest = self.get_suggestions

        if not 'cpu' in self.device.type:
            if 'n_jobs' not in kwargs or kwargs['n_jobs'] != 1:
                logger.warning("Optuna does not support multi-GPU jobs. Setting number of parallel jobs to 1")
            kwargs['n_jobs'] = 1

        if direction is None:
            mode = self.experiment_hparams.get('optimization-mode')
            objective = self.experiment_hparams.get('objective')
            direction = self.get_optimization_mode(mode, objective)
            direction = 'maximize' if direction == 'max' else 'minimize'

        if study_name is None:
            study_name = f'{self.hparams.project_name}-{self.hparams.algorithm}-{self.hparams.identifier}'
            # study_name = 'experiments'

        if storage is None:
            if self.hpo_path is not None:

                path = beam_path(self.hpo_path)
                path.joinpath('optuna').mkdir(parents=True, exist_ok=True)

                storage = f'sqlite:///{self.hpo_path}/optuna/{study_name}.db'
                logger.info(f"Using {storage} as storage to store the trials results")

        runner = partial(self.runner, suggest=suggest)

        if load_study:
            study = optuna.load_study(storage=storage, sampler=sampler, pruner=pruner, study_name=study_name)
        else:
            study = optuna.create_study(storage=storage, sampler=sampler, pruner=pruner, study_name=study_name,
                                        direction=direction, load_if_exists=load_if_exists, directions=directions)

        n_trials = self.hparams.n_trials

        study.optimize(runner, n_trials=n_trials, gc_after_trial=True, **kwargs)

        return study
