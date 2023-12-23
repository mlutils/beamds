# from examples.example_utils import add_beam_to_path
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


import torch
from torch import distributions
import torchvision
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd

from src.beam import beam_arguments, Experiment
from src.beam.tabular import TabularDataset, TabularTransformer, TabularHparams, DeepTabularAlg
from src.beam import beam_logger as logger
from src.beam.utils import get_public_ip
from src.beam.auto import AutoBeam


def train_catboost(dataset, exp, **kwargs):

    from catboost import CatBoostRegressor, CatBoostClassifier
    if dataset.task_type == 'regression':
        catboost_model = CatBoostRegressor
        loss_function = 'RMSE'
        eval_metric = 'RMSE'
        # custom_metric = ['MAE', 'MAPE']
        custom_metric = []
    else:
        catboost_model = CatBoostClassifier
        loss_function = 'MultiClass'
        eval_metric = 'Accuracy'
        custom_metric = ['Precision', 'Recall']

    x_train_mixed = pd.concat([pd.DataFrame(xi) for xi in [dataset.x_train_cat, dataset.x_train_num_scaled]], axis=1)
    x_val_mixed = pd.concat([pd.DataFrame(xi) for xi in [dataset.x_val_cat, dataset.x_val_num_scaled]], axis=1)
    x_test_mixed = pd.concat([pd.DataFrame(xi) for xi in [dataset.x_test_cat, dataset.x_test_num_scaled]], axis=1)

    x_train_mixed.columns = np.arange(x_train_mixed.shape[1])
    x_val_mixed.columns = np.arange(x_val_mixed.shape[1])
    x_test_mixed.columns = np.arange(x_test_mixed.shape[1])

    exp.results_dir.mkdir(parents=True, exist_ok=True)

    # log_config = {'log_file': str(exp.results_dir.joinpath('catboost.log'))}
    fit_kwargs = dict(cat_features=np.arange(dataset.x_train_cat.shape[1]), early_stopping_rounds=500)
    fit_args = {**fit_kwargs, 'X': x_train_mixed, 'y': dataset.y_train, 'eval_set': (x_val_mixed, dataset.y_val),
                }

    cb_kwargs = {  # 'learning_rate': 1e-2,
        'n_estimators': 2000,
        'random_seed': hparams.seed,
        'l2_leaf_reg': 1e-4,
        'border_count': 128,
        'depth': 14,
        'random_strength': .5,
        'task_type': 'GPU',
        'devices': str(exp.vars_args['device']),
        'loss_function': loss_function,
        'eval_metric': eval_metric,
        'custom_metric': custom_metric,
        'verbose': 50,
    }

    cb = catboost_model(**cb_kwargs)
    cb.fit(**fit_args)

    exp.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    cb.save_model(str(exp.checkpoints_dir.joinpath(f'cb.dump')))

def get_paths():

    ip = get_public_ip()

    if ip.startswith('199'):
        data_path = '/mnt/data/dataset/tabular/'
        logs_path = '/home/shared/results/'
    else:
        data_path = '/home/mlspeech/elads/data/tabular/data/'
        logs_path = '/dsi/shared/elads/elads/data/tabular/results/'

    return data_path, logs_path


if __name__ == '__main__':

    data_path, logs_path = get_paths()

    # kwargs_base = dict(algorithm='catboost_default',
    #                    data_path=data_path,
    #                    logs_path=logs_path,
    #                    copy_code=False, dynamic_masking=False,
    #                    tensorboard=True, stop_at=0.98, n_gpus=1, device=1, n_quantiles=6, catboost=True,
    #                    rulenet=False)

    kwargs_base = dict(algorithm='debug_reporter', data_path=data_path, logs_path=logs_path,
                       scheduler=False, device_placement=False, device=0, n_gpus=2,
                       copy_code=False, dynamic_masking=False, comet=False, tensorboard=True, n_epochs=100,
                       stop_at=-.57, n_quantiles=6, label_smoothing=.2,
                       model_dtype='float32', training_framework='deepspeed',
                       compile_train=False, sparse_embedding=False)

    kwargs_all = {}

    kwargs_all['california_housing'] = dict(batch_size=128)
    # kwargs_all['adult'] = dict(batch_size=128)
    # kwargs_all['helena'] = dict(batch_size=256, mask_rate=0.25, dropout=0.25, transformer_dropout=.25,
    #                             minimal_mask_rate=.2, maximal_mask_rate=.4,
    #                             label_smoothing=.25, n_quantiles=6, dynamic_masking=False)
    # kwargs_all['jannis'] = dict(batch_size=256)
    # kwargs_all['higgs_small'] = dict(batch_size=256)
    # kwargs_all['aloi'] = dict(batch_size=256)
    # kwargs_all['year'] = dict(batch_size=512)
    # kwargs_all['covtype'] = dict(batch_size=512, n_quantiles=10)

    for k in kwargs_all.keys():

        logger.info(f"Starting a new experiment with dataset: {k}")
        hparams = {**kwargs_base}
        hparams.update(kwargs_all[k])
        hparams['dataset_name'] = k
        hparams['identifier'] = k
        hparams = TabularHparams(hparams)

        exp = Experiment(hparams)
        dataset = TabularDataset(hparams)

        if hparams.rulenet:

            logger.info(f"Training a RuleNet predictor")
            # net = TabularTransformer(hparams, dataset.n_classes, dataset.n_tokens, dataset.cat_mask)
            # alg = DeepTabularAlg(hparams, networks=net)

            alg = exp.fit(alg=DeepTabularAlg, dataset=TabularDataset,
                          alg_kwargs={'net_kwargs': {'n_classes': dataset.n_classes, 'n_tokens': dataset.n_tokens,
                                                                                  'cat_mask': dataset.cat_mask}})
            logger.info(f"Training finished, reloading best model")
            exp.reload_checkpoint(alg)
            alg.set_best_masking()

            predictions = alg.evaluate('test')
            logger.info(f"Test objective: {predictions.statistics['test']['scalar']['objective'].values}")
            exp.results_dir.joinpath('predictions.pt').write(predictions)

            # store to bundle

            path = '/workspace/serve/bundle'
            logger.info(f"Storing bundle to: {path}")
            AutoBeam.to_bundle(alg, path)

        if hparams.catboost:

            logger.info(f"Training a Catboost predictor")
            train_catboost(dataset, exp)


