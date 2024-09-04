import re

from .. import beam_path
from ..path import local_copy
from ..utils import parse_string_number, as_numpy, cached_property, set_seed
from ..experiment.utils import build_device_list
from .config import CatboostConfig

from .core_algorithm import Algorithm
from ..logging import beam_logger as logger


class CBAlgorithm(Algorithm):

    def __init__(self, hparams=None, name=None, **kwargs):

        _config_scheme = kwargs.pop('_config_scheme', CatboostConfig)
        super().__init__(hparams=hparams, name=name, _config_scheme=_config_scheme,  **kwargs)
        self._t0 = None
        self._batch_size = None

    @property
    def log_frequency(self):
        return self.get_hparam('log_frequency', 1)

    @property
    def device_type(self):
        return 'CPU' if self.get_hparam('device', 'cpu') else 'GPU'

    @property
    def task_type(self):
        tp = self.get_hparam('cb_task', 'classification')
        assert tp in ['classification', 'regression', 'ranking'], f"Invalid task type: {tp}"
        return tp

    @property
    def devices(self):
        device_list = build_device_list(self.hparams)
        device_list = [d.index for d in device_list]
        return device_list

    @property
    def eval_metric(self):
        if self.task_type == 'regression':
            em = 'RMSE'
        else:
            em = 'Accuracy'
        return self.get_hparam('eval_metric', em)

    @property
    def custom_metric(self):
        if self.task_type == 'regression':
            cm = []
        else:
            cm = ['Precision', 'Recall']
        return self.get_hparam('custom_metric', cm)

    @cached_property
    def optimization_mode(self):
        objective_mode = self.get_hparam('objective_mode', None)
        objective_name = self.get_hparam('objective', None)
        return self.get_optimization_mode(objective_mode, objective_name)

    def set_objective(self):
        objective_name = self.get_hparam('objective', self.eval_metric)
        if type(objective_name) is list:
            objective_name = objective_name[0]
        self.set_hparam('objective', objective_name)

    @cached_property
    def model(self):

        seed = self.get_hparam('seed')
        if seed == 0:
            seed = None

        cb_kwargs = {
            'random_seed': seed,
            'task_type': self.device_type,
            'devices': self.devices,
            'eval_metric': self.eval_metric,
            'custom_metric': self.custom_metric,
            'verbose': self.log_frequency,
        }

        if self.task_type == 'classification':
            from catboost import CatBoostClassifier as CatBoost
        elif self.task_type == 'regression':
            from catboost import CatBoostRegressor as CatBoost
        elif self.task_type == 'ranking':
            from catboost import CatBoostRanker as CatBoost
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")

        from .catboost_consts import cb_keys
        for key in cb_keys[self.task_type]:
            v = self.get_hparam(key, None)
            if v is not None:
                if key in cb_kwargs:
                    # logger.error(f"CB init: Overriding key {key} with value {v}")
                    continue
                cb_kwargs[key] = self.hparams[key]

        return CatBoost(**cb_kwargs)

    @cached_property
    def info_re_pattern(self):
        # Regular expression pattern to capture iteration number and then any number of key-value metrics
        pattern = r'(?P<iteration>\d+):\t((?P<metric_name>\w+):\s(?P<metric_value>[\d.\w]+)\s*(?:\(\d+\))?\s*)+'

        # Compiling the pattern
        compiled_pattern = re.compile(pattern)
        return compiled_pattern

    def log_cerr(self, err):
        logger.error(f"CB: {err}")

    def postprocess_epoch(self, info, **kwargs):

        # Searching the string
        match = self.info_re_pattern.search(info)

        if match:
            # Extracting iteration number
            iteration = int(match.group('iteration'))
            self.reporter.set_iteration(iteration)

            # Extracting metrics
            metrics_string = info[info.index('\t') + 1:].strip()  # Get the substring after the iteration
            metrics_parts = re.findall(r'(\w+):\s([\d.\w]+)\s*(?:\(\d+\))?', metrics_string)

            # Converting metric parts into a dictionary
            metrics = {}
            for name, value in metrics_parts:
                v, u = parse_string_number(value, timedelta_format=False, return_units=True)
                name = f"{name}[sec]" if u else name
                metrics[name] = v

            for k, v in metrics.items():
                self.report_scalar(k, v, subset='eval', epoch=iteration)

            self.reporter.post_epoch('eval', self._t0, training=True)
            # post epoch
            self.epoch = iteration + self.log_frequency
            self.calculate_objective_and_report(self.epoch)

            if self.experiment:
                self.experiment.save_model_results(self.reporter, self, self.epoch, visualize_weights=False,
                                                   store_results=False, save_checkpoint=False)

            self.reporter_pre_epoch(self.epoch)

        logger.debug(f"CB: {info}")

    def reporter_pre_epoch(self, epoch, batch_size=None):

        if batch_size is None:
            batch_size = self._batch_size
        else:
            self._batch_size = batch_size

        self.reporter.reset_epoch(epoch, total_epochs=self.epoch)

        # due to the catboost behavior where the first logging interval is of size 1
        # where the rest are of size self.log_frequency
        if epoch > 0:
            batch_size = batch_size * self.log_frequency

        self._t0 = self.reporter.pre_epoch('eval', batch_size=batch_size)

    def log_cout(self, *args, **kwargs):
        try:
            self.postprocess_epoch(*args, **kwargs)
        except Exception as e:
            logger.error(f"CB: {e}")
            from ..utils import beam_traceback
            logger.error(beam_traceback())

    @cached_property
    def n_epochs(self):
        return self.get_hparam('iterations', 1000)

    def _fit(self, x=None, y=None, dataset=None, eval_set=None, cat_features=None, text_features=None,
             embedding_features=None, sample_weight=None, **kwargs):

        self.set_objective()

        if self.experiment:
            self.experiment.prepare_experiment_for_run()

        if dataset is None:
            from ..dataset import TabularDataset
            dataset = TabularDataset(x_train=x, y_train=y, x_test=eval_set[0], y_test=eval_set[1],
                                     cat_features=cat_features, text_features=text_features,
                                     embedding_features=embedding_features, sample_weight=sample_weight)

        self.set_train_reporter(first_epoch=0, n_epochs=self.n_epochs)

        train_pool = dataset.train_pool
        self.epoch = 0
        self.reporter_pre_epoch(0, batch_size=len(train_pool.get_label()))

        snapshot_file = None
        if self.experiment:
            snapshot_file = self.experiment.snapshot_file

        self.model.fit(train_pool, eval_set=dataset.eval_pool, log_cout=self.log_cout,
                       snapshot_interval=self.get_hparam('snapshot_interval'),
                       save_snapshot=self.get_hparam('save_snapshot'),
                       snapshot_file=snapshot_file, log_cerr=self.log_cerr, **kwargs)

        if self.experiment:
            self.experiment.save_state(self)

    def predict(self, x, **kwargs):
        return self.model.predict(as_numpy(x), **kwargs)

    def __sklearn_clone__(self):
        # to be used with sklearn clone
        return CBAlgorithm(self.hparams)

    @classmethod
    @property
    def excluded_attributes(cls):
        return super(CBAlgorithm, cls).excluded_attributes.union(['model'])

    def load_state_dict(self, path, ext=None, exclude: set | list = None, **kwargs):

        path = beam_path(path)
        with local_copy(path.joinpath('model.cb'), as_beam_path=False) as p:
            self.model.load_model(p)

        return super().load_state_dict(path, ext, exclude, **kwargs)

    def save_state_dict(self, state, path, ext=None,  exclude: set | list = None, **kwargs):

        super().save_state_dict(state, path, ext, exclude, **kwargs)

        path = beam_path(path)
        with local_copy(path.joinpath('model.cb'), as_beam_path=False) as p:
            self.model.save_model(p)
