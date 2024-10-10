import re

from sklearn.metrics import fbeta_score

from ..path import beam_path
from ..experiment import BeamReport
from ..path import local_copy
from ..utils import parse_string_number, as_numpy, cached_property, set_seed
from ..experiment.utils import build_device_list
from .config import CatboostConfig

from .core_algorithm import Algorithm
from ..logging import beam_logger as logger, BeamError
from ..type import check_type, Types
from ..dataset import TabularDataset


class FBetaMetric:
    def __init__(self, beta=1., threshold=0.5):
        self.beta = beta
        self.threshold = threshold

    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        approx = approxes[0]

        y_pred = (approx > self.threshold).astype(int)
        y_true = target.astype(int)
        score = fbeta_score(y_true, y_pred, beta=self.beta)

        return score, 1


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
        return 'CPU' if self.get_hparam('device', 'cpu') == 'cpu' else 'GPU'

    @property
    def devices(self):
        device_list = build_device_list(self.hparams)
        device_list = [d.index for d in device_list]
        return device_list

    @property
    def task_type(self):
        return self._task_type(self.hparams)

    @staticmethod
    def _task_type(hparams):
        tp = hparams.get('cb_task', 'classification')
        assert tp in ['classification', 'regression', 'ranking'], f"Invalid task type: {tp}"
        return tp

    @property
    def eval_metric(self):
        return self._eval_metric(self.hparams)

    @staticmethod
    def _eval_metric(hparams):
        if CBAlgorithm._task_type(hparams) == 'regression':
            em = 'RMSE'
        else:
            em = 'Accuracy'
        return hparams.get('eval_metric', em)

    @property
    def custom_metric(self):
        if self.task_type == 'regression':
            cm = []
        else:
            cm = ['Precision', 'Recall']
        return self.get_hparam('custom_metric', cm)

    @staticmethod
    def expected_objective_name(hparams):
        objective_name = hparams.get('objective', CBAlgorithm._eval_metric(hparams))
        if type(objective_name) is list:
            objective_name = objective_name[0]
        return objective_name

    @property
    def catboost_kwargs(self):

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

        from .catboost_consts import cb_keys
        for key in cb_keys[self.task_type]:
            v = self.get_hparam(key, None)
            if v is not None:
                if key in cb_kwargs:
                    # logger.error(f"CB init: Overriding key {key} with value {v}")
                    continue
                cb_kwargs[key] = self.hparams[key]

        # here we fix broken configurations that can be the result of a hpo tuning
        if cb_kwargs.get('max_leaves', None) is not None:
            if cb_kwargs.get('grow_policy', None) not in ['Lossguide', None]:
                logger.warning(f"Beam-Catboost: Ignoring max_leaves with grow_policy: {cb_kwargs['grow_policy']}")
                cb_kwargs.pop('max_leaves')

        # if 'grow_policy' not in cb_kwargs or cb_kwargs['grow_policy'] == 'SymmetricTree':
        # if 'grow_policy' in cb_kwargs and cb_kwargs['grow_policy'] != 'SymmetricTree':
        if cb_kwargs.get('boosting_type', None) is not None:
            if cb_kwargs.get('grow_policy', None) not in ['SymmetricTree', None]:
                logger.warning(f"Beam-Catboost: Ignoring boosting_type with grow_policy: {cb_kwargs['grow_policy']}")
                cb_kwargs.pop('boosting_type')

        # if cb_kwargs.get('od_type', None) in ['IncToDec', None]:

        if cb_kwargs.get('early_stopping_rounds', None) is not None:
            if cb_kwargs.get('od_wait', None) is not None:
                logger.warning(f"Beam-Catboost: Ignoring early_stopping_rounds with od_type: {cb_kwargs['od_wait']}")
                cb_kwargs.pop('early_stopping_rounds')

        if cb_kwargs.get('od_pval', None) is not None:
            if cb_kwargs.get('od_type', None) == 'Iter':
                logger.warning(f"Beam-Catboost: Ignoring od_pval with od_type: {cb_kwargs['od_type']}")
                cb_kwargs.pop('od_pval')

        if cb_kwargs.get('bagging_temperature', None) is not None:
            if cb_kwargs.get('bootstrap_type', None) not in ['Bayesian', None]:
                logger.warning(f"Beam-Catboost: Ignoring bagging_temperature with bootstrap_type: "
                               f"{cb_kwargs['bootstrap_type']}")
                cb_kwargs.pop('bagging_temperature')

        if self.device_type == 'CPU':
            if cb_kwargs.get('leaf_estimation_backtracking', None) == 'Armijo':
                logger.warning(f"Beam-Catboost: Backtracking type Armijo is supported only on GPU, ignoring it "
                               f"(defaulting to AnyImprovement)")
                cb_kwargs.pop('leaf_estimation_backtracking')
        else:
            if cb_kwargs.get('rsm', None) is not None:
                logger.warning(f"Beam-Catboost: rsm on GPU is supported for pairwise modes only (ignoring it)")
                cb_kwargs.pop('rsm')

        return cb_kwargs

    @cached_property
    def model(self):

        if self.task_type == 'classification':
            from catboost import CatBoostClassifier as CatBoost
        elif self.task_type == 'regression':
            from catboost import CatBoostRegressor as CatBoost
        elif self.task_type == 'ranking':
            from catboost import CatBoostRanker as CatBoost
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")

        return CatBoost(**self.catboost_kwargs)

    @cached_property
    def info_re_pattern(self):
        # Regular expression pattern to capture iteration number and then any number of key-value metrics
        pattern = r'(?P<iteration>\d+):\t((?P<metric_name>\w+):\s(?P<metric_value>[\d.\w]+)\s*(?:\(\d+\))?\s*)+'

        # Compiling the pattern
        compiled_pattern = re.compile(pattern)
        return compiled_pattern

    def log_cerr(self, err):
        logger.error(f"Beam-Catboost: {err}")

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

            self.reporter.post_epoch('eval', self._t0, track_objective=True)
            # post epoch
            self.epoch = iteration + self.log_frequency
            self.calculate_objective_and_report(self.epoch)

            if self.experiment:
                self.experiment.save_model_results(self.reporter, self, self.epoch, visualize_weights=False,
                                                   store_results=False, save_checkpoint=False)

            self.reporter_pre_epoch(self.epoch)

        logger.debug(f"Beam-Catboost: {info}")

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
            logger.debug(f"CB Error: {e}")
            from ..utils import beam_traceback
            logger.debug(beam_traceback())
            raise e

    @cached_property
    def n_epochs(self):
        return self.get_hparam('iterations', 1000)

    @cached_property
    def train_reporter(self):
        return BeamReport(objective='best', optimization_mode=self.optimization_mode,
                          aux_objectives=['loss'], aux_objectives_modes=['min'])

    def _fit(self, x=None, y=None, dataset=None, eval_set=None, cat_features=None, text_features=None,
             embedding_features=None, sample_weight=None, **kwargs):

        if x is not None:
            if isinstance(x, TabularDataset):
                dataset = x
                x = None

        if self.experiment:
            self.experiment.prepare_experiment_for_run()

        if dataset is None:
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

        try:
            self.model.fit(train_pool, eval_set=dataset.eval_pool, log_cout=self.log_cout,
                           snapshot_interval=self.get_hparam('snapshot_interval'),
                           save_snapshot=self.get_hparam('save_snapshot'),
                           snapshot_file=snapshot_file, log_cerr=self.log_cerr, **kwargs)

        except SystemError as e:
            if self.hpo == 'optuna':
                from optuna.exceptions import TrialPruned
                raise TrialPruned(f"Trial pruned: {e}")
            else:
                raise e

        except Exception as e:
            logger.error(f"Beam-Catboost: {e}")
            logger.debug(f"Beam-Catboost: {e}")
            raise BeamError(f"CatBoost: {e}")

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

        if path.joinpath('model.cb').exists():
            with local_copy(path.joinpath('model.cb'), as_beam_path=False) as p:
                self.model.load_model(p)
        elif path.joinpath('model.pkl').exists():
            self.model = path.joinpath('model.pkl').read()
        else:
            raise FileNotFoundError(f"Model file not found in {path}")

        return super().load_state_dict(path, ext, exclude, **kwargs)

    def save_state_dict(self, state, path, ext=None,  exclude: set | list = None, **kwargs):

        super().save_state_dict(state, path, ext, exclude, **kwargs)

        path = beam_path(path)

        if self.model.is_fitted():
            with local_copy(path.joinpath('model.cb'), as_beam_path=False) as p:
                self.model.save_model(p)
        else:
            path.joinpath('model.pkl').write(self.model)
