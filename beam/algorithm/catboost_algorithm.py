import re

from ..utils import parse_string_number, as_numpy, cached_property, set_seed
from ..experiment.utils import build_device_list
from .config import CatboostConfig

from .core_algorithm import Algorithm
from ..logging import beam_logger as logger


class CBAlgorithm(Algorithm):

    def __init__(self, hparams=None, name=None, **kwargs):

        super().__init__(hparams=hparams, name=name, _config_scheme=CatboostConfig,  **kwargs)
        self._t0 = None
        self._batch_size = None

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
        objective_name = self.get_hparam('objective', self.eval_metric)
        return self.get_optimization_mode(objective_mode, objective_name)

    @cached_property
    def model(self):

        seed = self.get_hparam('seed')
        if seed == 0:
            seed = None

        cb_kwargs = {
            'learning_rate': self.get_hparam('learning_rate'),
            'n_estimators': self.get_hparam('n_estimators'),
            'random_seed': seed,
            'l2_leaf_reg': self.get_hparam('l2_leaf_reg'),
            'border_count': self.get_hparam('border_count'),
            'depth': self.get_hparam('depth'),
            'random_strength': self.get_hparam('random_strength'),
            'task_type': self.device_type,
            'devices': self.devices,
            'loss_function': self.get_hparam('loss_function'),
            'eval_metric': self.eval_metric,
            'custom_metric': self.custom_metric,
            'verbose': self.get_hparam('log_frequency'),
        }

        if self.task_type == 'classification':
            from catboost import CatBoostClassifier as CatBoost
        elif self.task_type == 'regression':
            from catboost import CatBoostRegressor as CatBoost
        elif self.task_type == 'ranking':
            from catboost import CatBoostRanker as CatBoost
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")

        return CatBoost(**cb_kwargs)

    @cached_property
    def info_re_pattern(self):
        # Regular expression pattern to capture iteration number and then any number of key-value metrics
        pattern = r'(?P<iteration>\d+):\t((?P<metric_name>\w+):\s(?P<metric_value>[\d.\w]+)\s*(?:\(\d+\))?\s*)+'

        # Compiling the pattern
        compiled_pattern = re.compile(pattern)
        return compiled_pattern

    def err_stream(self, err):
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
            metrics = {name: parse_string_number(value) for name, value in metrics_parts}

            # logger.info(metrics)

            for k, v in metrics.items():
                self.report_scalar(k, v, subset='eval', epoch=iteration)

            self.reporter.post_epoch('eval', self._t0, training=True)

            if self.experiment:
                self.experiment.save_model_results(self.reporter, self, iteration, visualize_weights=False,
                                                   store_results=False,)

            # post epoch
            self.epoch += 1
            self.reporter.reset_epoch(iteration, total_epochs=self.epoch)
            self._t0 = self.reporter.pre_epoch('eval', batch_size=self._batch_size, training=True)

        else:
            logger.info(f"CB: {info}")

    def _fit(self, x=None, y=None, dataset=None, eval_set=None, cat_features=None, text_features=None,
             embedding_features=None, sample_weight=None, **kwargs):

        if dataset is None:
            from ..dataset import TabularDataset
            dataset = TabularDataset(x_train=x, y_train=y, x_test=eval_set[0], y_test=eval_set[1],
                                     cat_features=cat_features, text_features=text_features,
                                     embedding_features=embedding_features, sample_weight=sample_weight)

        self.set_train_reporter(first_epoch=0, n_epochs=self.get_hparam('cb_n_estimators'))
        # self.set_reporter(BeamReport(objective=self.get_hparam('objective'),
        #                              objective_mode=self.optimization_mode))
        self.reporter.reset_epoch(0, total_epochs=self.epoch)
        train_pool = dataset.train_pool
        self._batch_size = len(train_pool.get_label())
        self._t0 = self.reporter.pre_epoch('eval', batch_size=self._batch_size)
        self.model.fit(train_pool, eval_set=dataset.eval_pool, log_cout=self.postprocess_epoch,
                       log_cerr=self.err_stream, **kwargs)

        # self.model.fit(dataset.train_pool, eval_set=dataset.eval_pool, **kwargs)

    def predict(self, x, **kwargs):
        return self.model.predict(as_numpy(x), **kwargs)

    def __sklearn_clone__(self):
        # to be used with sklearn clone
        return CBAlgorithm(self.hparams)