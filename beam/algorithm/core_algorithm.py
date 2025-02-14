from dataclasses import dataclass
from timeit import default_timer as timer
from functools import wraps

from ..utils import cached_property
from ..processor import Processor
from ..logging import beam_logger as logger
from ..experiment import BeamReport


@dataclass
class FitReport:
    objective: float
    objective_name: str
    optimization_mode: str
    epoch: int
    best_objective: float
    best_epoch: int
    best_state: bool
    results: dict

    def __str__(self):
        return (f"Objective name: {self.objective_name}, Optimization mode: {self.optimization_mode}, "
                f"Objective: {self.objective}, Epoch: {self.epoch}, Best Objective: {self.best_objective}, "
                f"Best Epoch: {self.best_epoch}, Best State: {self.best_state}")


class Algorithm(Processor):
    def __init__(self, hparams=None, name=None, experiment=None, **kwargs):
        super().__init__(hparams=hparams, name=name, **kwargs)

        self._experiment = None
        self.reporter = None

        self.epoch = 0
        self.t0 = timer()
        self._device = None
        self.trial = None

        self.epoch_length = None
        self.eval_subset = None
        self.objective = None
        self.best_objective = None
        self.best_epoch = None
        self.best_state = False
        self.datasets = None
        self._n_epochs = None

        self.clear_experiment_properties()
        if experiment is not None:
            self.experiment = experiment

    def load_datasets(self, datasets):
        self.datasets = datasets

    @classmethod
    @property
    def excluded_attributes(cls) -> set[str]:
        return super(Algorithm, cls).excluded_attributes.union(['_experiment', 'reporter', 'trial',
                                                                'datasets', 'persistent_dataloaders',
                                                                'dataloaders'])

    def calculate_objective_and_report(self, i):

        objective = self.calculate_objective()

        if self.get_hparam('objective_to_report') == 'last':
            report_objective = objective
        elif self.get_hparam('objective_to_report') == 'best':
            report_objective = self.best_objective
        else:
            raise Exception(f"Unknown objective_to_report: {self.get_hparam('objective_to_report')} "
                            f"should be [last|best]")
        self.report(report_objective, i)

        return objective

    @property
    def elapsed_time(self):
        return timer() - self.t0

    @staticmethod
    def no_experiment_message(property):
        logger.warning(f"{property} is not supported without an active experiment. Set self.experiment = experiment")

    @property
    def experiment(self):
        logger.debug(f"Fetching the experiment which is currently associated with the algorithm")
        return self._experiment

    # a setter function
    @experiment.setter
    def experiment(self, experiment):
        logger.debug(f"The algorithm is now linked to an experiment directory: {experiment.experiment_dir}")
        self.trial = experiment.trial
        self.hparams = experiment.hparams
        self.clear_experiment_properties()
        self._experiment = experiment

    @cached_property
    def train_reporter(self):
        return BeamReport(objective=self.objective_name, optimization_mode=self.optimization_mode,
                          aux_objectives=['loss'], aux_objectives_modes=['min'])

    def set_reporter(self, reporter=None):
        self.reporter = reporter
        self.reporter.reset_time(None)
        self.reporter.reset_epoch(0, total_epochs=None)

    def set_train_reporter(self, first_epoch, n_epochs=None):

        if n_epochs is None:
            n_epochs = self.n_epochs

        self.reporter = self.train_reporter
        self.reporter.reset_time(first_epoch=first_epoch, n_epochs=n_epochs)

    def clear_experiment_properties(self):

        self.clear_cache('device', 'distributed_training', 'distributed_training_framework', 'hpo', 'rank', 'world_size', 'enable_tqdm', 'n_epochs',
                            'batch_size_train', 'batch_size_eval', 'pin_memory', 'autocast_device', 'model_dtype', 'amp',
                            'scaler', 'swa_epochs')

    @cached_property
    def hpo(self):
        if self.experiment is None:
            self.no_experiment_message('hpo')
            return False
        return self.experiment.hpo

    @cached_property
    def enable_tqdm(self):
        return self.get_hparam('enable_tqdm') if (self.get_hparam('tqdm_threshold') == 0
                                                              or not self.get_hparam('enable_tqdm')) else None
    @property
    def n_epochs(self):
        if self._n_epochs is None:
            self._n_epochs = self.get_hparam('n_epochs')
        return self._n_epochs

    def training_closure(self, *args, **kwargs):
        pass

    def calculate_objective(self):
        '''
        This function calculates the optimization non-differentiable objective. It is used for hyperparameter optimization
        and for ReduceOnPlateau scheduling. It is also responsible for tracking the best checkpoint
        '''

        self.best_objective = self.reporter.best_objective
        self.best_epoch = self.reporter.best_epoch
        self.objective = self.reporter.objective
        self.best_state = self.reporter.best_state

        return self.objective

    def report(self, objective, epoch=None):
        '''
        Use this function to report results to hyperparameter optimization frameworks
        also you can add key 'objective' to the results dictionary to report the final scores.
        '''

        objective_name = self.get_hparam('objective', 'objective')
        if self.get_hparam('report_best_objective', False):
            objective_value = self.best_objective
        else:
            objective_value = objective

        if self.hpo == 'tune':
            metrics = {objective_name: objective_value}
            from ray import train
            train.report(metrics)

        elif self.hpo == 'optuna':
            import optuna
            self.trial.report(objective_value, epoch)
            self.trial.set_user_attr('best_value', self.best_objective)
            self.trial.set_user_attr('best_epoch', self.best_epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
            train_timeout = self.get_hparam('train-timeout')
            if train_timeout is not None and 0 < train_timeout < self.elapsed_time:
                raise optuna.exceptions.OptunaError(f"Trial timed out after {self.get_hparam('train-timeout')} seconds.")

    @cached_property
    def objective_name(self):
        objective_name = self.expected_objective_name(self.hparams)
        self.set_hparam('objective', objective_name)
        return objective_name

    @staticmethod
    def expected_objective_name(hparams):
        objective_name = hparams.get('objective', 'loss')
        return objective_name

    @cached_property
    def optimization_mode(self):
        optimization_mode = self.get_hparam('optimization_mode', None)
        optimization_mode = self.get_optimization_mode(optimization_mode, self.objective_name)
        return optimization_mode

    @staticmethod
    def get_optimization_mode(mode, objective_name):
        # do not override this method in sub-classes as ray tune and optuna use it before the algorithm is initialized
        if mode is not None:
            return mode
        if objective_name is None or any(n in objective_name.lower() for n in ['loss', 'error', 'mse',
                                                                               'entropy', 'mae', 'mape']):
            mode = 'min'
        else:
            mode = 'max'
        logger.debug(f"Algorithm: Optimization mode: {mode}, objective: {objective_name}")

        return mode

    def early_stopping(self, epoch=None):
        '''
        Use this function to early stop your model based on the results or any other metric in the algorithm class
        '''

        if self.rank > 0:
            return False

        train_timeout = self.get_hparam('train-timeout')
        if train_timeout is not None and 0 < train_timeout < self.elapsed_time:
            logger.info(f"Stopping training at epoch {self.epoch} - timeout {self.get_hparam('train-timeout')}")
            return True

        stop_at = self.get_hparam('stop_at')
        early_stopping_patience = self.get_hparam('early_stopping_patience')
        if self.objective is None and stop_at is not None:
            logger.warning("Early stopping is enabled (stop_at is not None) but no objective is defined. "
                           "set objective in the hparams")
            return False
        if self.objective is None and early_stopping_patience is not None:
            logger.warning("Early stopping is enabled (early_stopping_patience is not None) "
                           "but no objective is defined. set objective in the hparams")
            return False

        if stop_at is not None:
            if self.best_objective is not None:

                if self.optimization_mode == 'max':
                    res = self.best_objective > stop_at
                    if res:
                        logger.info(f"Stopping training at {self.best_objective} > {stop_at}")
                else:
                    res = self.best_objective < stop_at
                    if res:
                        logger.info(f"Stopping training at {self.best_objective} < {stop_at}")
                return res

        if early_stopping_patience is not None and early_stopping_patience > 0:
            res = self.epoch - self.best_epoch > early_stopping_patience
            if res:
                logger.info(f"Stopping training at epoch {self.epoch} - best epoch {self.best_epoch} > {early_stopping_patience}")
            return res

        return False

    def preprocess_inference(self, *args, **kwargs):
        pass

    def postprocess_inference(self, *args, **kwargs):
        pass

    def postprocess_epoch(self, *args, **kwargs):
        pass

    def preprocess_epoch(self, *args, **kwargs):
        pass

    def _fit(self, *args, **kwargs):
        raise NotImplementedError("please implement _fit method in your sub-class")

    def fit(self, *args, **kwargs) -> FitReport:

        if self._experiment is None:
            from ..config import ExperimentConfig
            from ..experiment import Experiment
            conf = ExperimentConfig(self.hparams)
            if conf.log_experiment is None:
                conf.log_experiment = False
            experiment = Experiment(conf)
            self.experiment = experiment

        results = self._fit(*args, **kwargs)
        report = FitReport(objective=self.objective, epoch=self.epoch, best_objective=self.best_objective,
                           best_epoch=self.best_epoch, best_state=self.best_state, results=results,
                           objective_name=self.objective_name, optimization_mode=self.optimization_mode)
        return report

    def predict(self, *args, **kwargs):
        raise NotImplementedError('predict method not implemented')

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError('evaluate method not implemented')

    def report_scalar(self, name, val, subset=None, aggregation=None, append=None, **kwargs):
        self.reporter.report_scalar(name, val, subset=subset, aggregation=aggregation, append=append, **kwargs)

    def report_data(self, name, val, subset=None, data_type=None, **kwargs):

        if '/' in name:
            dt, name = name.split('/')

            if data_type is None:
                data_type = dt
            else:
                data_type = f"{dt}_{data_type}"

        self.reporter.report_data(name, val, subset=subset, data_type=data_type, **kwargs)

    def report_image(self, name, val, subset=None, **kwargs):
        self.reporter.report_image(name, val, subset=subset, **kwargs)

    def report_images(self, name, val, subset=None, **kwargs):
        self.reporter.report_images(name, val, subset=subset, **kwargs)

    def report_scalars(self, name, val, subset=None, **kwargs):
        self.reporter.report_scalars(name, val, subset=subset, **kwargs)

    def report_histogram(self, name, val, subset=None, **kwargs):
        self.reporter.report_histogram(name, val, subset=subset, **kwargs)

    def report_figure(self, name, val, subset=None, **kwargs):
        self.reporter.report_figure(name, val, subset=subset, **kwargs)

    def report_video(self, name, val, subset=None, **kwargs):
        self.reporter.report_video(name, val, subset=subset, **kwargs)

    def report_audio(self, name, val, subset=None, **kwargs):
        self.reporter.report_audio(name, val, subset=subset, **kwargs)

    def report_embedding(self, name, val, subset=None, **kwargs):
        self.reporter.report_embedding(name, val, subset=subset, **kwargs)

    def report_text(self, name, val, subset=None, **kwargs):
        self.reporter.report_text(name, val, subset=subset, **kwargs)

    def report_mesh(self, name, val, subset=None, **kwargs):
        self.reporter.report_mesh(name, val, subset=subset, **kwargs)

    def report_pr_curve(self, name, val, subset=None, **kwargs):
        self.reporter.report_pr_curve(name, val, subset=subset, **kwargs)

    def get_scalar(self, name, subset=None, aggregate=False):
        v = self.reporter.get_scalar(name, subset=subset, aggregate=aggregate)
        return self.reporter.stack_scalar(v)

    def get_scalars(self, name, subset=None, aggregate=False):
        d = self.reporter.get_scalars(name, subset=subset, aggregate=aggregate)
        for k, v in d.items():
            d[k] = self.reporter.stack_scalar(v)
        return d

    def get_data(self, name, subset=None, data_type=None):

        if '/' in name:
            dt, name = name.split('/')

            if data_type is None:
                data_type = dt
            else:
                data_type = f"{dt}_{data_type}"

        return self.reporter.get_data(name, subset=subset, data_type=data_type)

    def get_image(self, name, subset=None):
        return self.reporter.get_image(name, subset=subset)

    def get_images(self, name, subset=None):
        return self.reporter.get_images(name, subset=subset)

    def get_histogram(self, name, subset=None):
        return self.reporter.get_histogram(name, subset=subset)

    def get_figure(self, name, subset=None):
        return self.reporter.get_figure(name, subset=subset)

    def get_video(self, name, subset=None):
        return self.reporter.get_video(name, subset=subset)

    def get_audio(self, name, subset=None):
        return self.reporter.get_audio(name, subset=subset)

    def get_embedding(self, name, subset=None):
        return self.reporter.get_embedding(name, subset=subset)

    def get_text(self, name, subset=None):
        return self.reporter.get_text(name, subset=subset)

    def get_mesh(self, name, subset=None):
        return self.reporter.get_mesh(name, subset=subset)

    def get_pr_curve(self, name, subset=None):
        return self.reporter.get_pr_curve(name, subset=subset)

    @wraps(Processor.load_state)
    def load_checkpoint(self, *args, **kwargs):
        return self.load_state(*args, **kwargs)

    def save_checkpoint(self, *args, **kwargs):
        return self.save_state(*args, **kwargs)
