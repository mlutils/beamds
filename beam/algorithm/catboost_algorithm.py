import re

from ..utils import parse_string_number, as_numpy, cached_property
from ..experiment.utils import build_device_list
from ..config import CatboostConfig

from .core_algorithm import Algorithm
from ..logging import beam_logger as logger


class CBAlgorithm(Algorithm):

    def __init__(self, hparams=None, name=None, **kwargs):

        super().__init__(hparams=hparams, name=name, _config_scheme=CatboostConfig,  **kwargs)

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
    def model(self):
        cb_kwargs = {
            'learning_rate': self.get_hparam('lr'),
            'n_estimators': self.get_hparam('cb_n_estimators'),
            'random_seed': self.get_hparam('seed'),
            'l2_leaf_reg': self.get_hparam('cb_l2_leaf_reg'),
            'border_count': self.get_hparam('cb_border_count'),
            'depth': self.get_hparam('cb_depth'),
            'random_strength': self.get_hparam('cb_random_strength'),
            'task_type': self.device_type,
            'devices': self.devices,
            'loss_function': self.get_hparam('loss_function'),
            'eval_metric': self.eval_metric,
            'custom_metric': self.custom_metric,
            'verbose': self.get_hparam('cb_log_resolution'),
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

    def postprocess_epoch(self, info, **kwargs):

        # Searching the string
        match = self.info_re_pattern.search(info)

        if match:
            # Extracting iteration number
            iteration = match.group('iteration')

            # Extracting metrics
            metrics_string = info[info.index('\t') + 1:].strip()  # Get the substring after the iteration
            metrics_parts = re.findall(r'(\w+):\s([\d.\w]+)\s*(?:\(\d+\))?', metrics_string)

            # Converting metric parts into a dictionary
            metrics = {name: parse_string_number(value) for name, value in metrics_parts}

            logger.info(metrics)

            for k, v in metrics.items():
                self.report_scalar(k, v, subset='eval', epoch=iteration)

            self.experiment.save_model_results(self.model, None, iteration)

            # post epoch
            self.epoch += 1
            self.reporter.reset_epoch(iteration, total_epochs=self.epoch)

    def _fit(self, X, y, eval_set=None, beam_postprocess=True, **kwargs):

        from catboost import Pool

        log_cout = None
        if beam_postprocess:
            log_cout = self.postprocess_epoch

        train_set = Pool(as_numpy(X), as_numpy(y))
        if eval_set is not None:
            eval_set = Pool(as_numpy(eval_set[0]), as_numpy(eval_set[1]))

        self.set_train_reporter(first_epoch=0, n_epochs=self.get_hparam('cb_n_estimators'))

        return self.model.fit(train_set, eval_set=eval_set, log_cout=log_cout, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(as_numpy(X), **kwargs)

    def __sklearn_clone__(self):
        # to be used with sklearn clone
        return CBAlgorithm(self.hparams)