from functools import cached_property
from catboost import CatBoostClassifier, CatBoostRegressor, CatBoost

from ..processor import Processor
from ..experiment.utils import build_device_list


class CBAlgorithm(Processor):

    def __init__(self, hparams, **kwargs):

        super().__init__(hparams=hparams, **kwargs)

    @property
    def task_type(self):
        return 'CPU' if self.get_hparam('device', 'cpu') else 'GPU'

    @property
    def devices(self):
        device_list = build_device_list(self.hparams)
        device_list = [d.index for d in device_list]
        return device_list

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
            'task_type': self.task_type,
            'devices': self.devices,
            'loss_function': loss_function,
            'eval_metric': eval_metric,
            'custom_metric': custom_metric,
            'verbose': 50,
        }

        return CatBoost(**cb_kwargs)

    def after_iteration(self, info):
        print(info)
        return False  # return True to stop training


    def fit(self, X, y):
        self.model.fit(X, y, log_cout=self.after_iteration)

    def predict(self, X):
        return self.model.predict(X)