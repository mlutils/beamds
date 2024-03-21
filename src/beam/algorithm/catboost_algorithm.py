from ..core import Processor
from ..utils import lazy_property


class CBAlgorithm(Processor):

    def __init__(self, hparams, **kwargs):

        super().__init__(hparams=hparams, **kwargs)


    @lazy_property
    def model(self):
        cb_kwargs = {  # 'learning_rate': 1e-2,
            'n_estimators': self.get_hparam('cb_n_estimators', 2000),
            'random_seed': self.get_hparam('seed', 0),
            'l2_leaf_reg': self.get_hparam('cb_l2_leaf_reg', 1e-4),
            'border_count': self.get_hparam('cb_border_count', 128),
            'depth': self.get_hparam('cb_depth', 14),
            'random_strength': self.get_hparam('cb_random_strength', .5),
            'task_type': 'CPU' if self.get_hparam('device', 'cpu') else 'GPU',
            'devices': str(exp.vars_args['device']),
            'loss_function': loss_function,
            'eval_metric': eval_metric,
            'custom_metric': custom_metric,
            'verbose': 50,
        }