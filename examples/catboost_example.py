import numpy as np
import pandas as pd

from beam import beam_arguments, Experiment, Timer
from beam.tabular import TabularDataset, TabularConfig, DeepTabularAlg
from beam import beam_logger as logger
from beam.utils import get_public_ip
from beam.auto import AutoBeam


def my_log_func(*args, **kwargs):
    print('in my log func')
    print(len(args))
    print(kwargs.keys())

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
                'log_cout': my_log_func}

    cb_kwargs = {  # 'learning_rate': 1e-2,
        'n_estimators': hparams.get('cb_n_estimators', 2000),
        'random_seed': hparams.seed,
        'l2_leaf_reg': hparams.get('cb_l2_leaf_reg', 1e-4),
        'border_count': hparams.get('cb_border_count', 128),
        'depth': hparams.get('cb_depth', 14),
        'random_strength': hparams.get('cb_random_strength', .5),
        'task_type': 'CPU' if hparams.get('device', 'cpu') else 'GPU',
        'devices': str(exp.vars_args['device']),
        'loss_function': loss_function,
        'eval_metric': eval_metric,
        'custom_metric': custom_metric,
        'verbose': 5,
    }

    cb = catboost_model(**cb_kwargs)

    logger.info(f"Results will be saved to: {exp.results_dir}")
    with Timer(name='Catboost training time', logger=logger) as t:
        cb.fit(**fit_args)

    exp.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    cb.save_model(str(exp.checkpoints_dir.joinpath(f'cb.dump')))

    # Assuming 'cb' is your trained CatBoost model object
    metrics = cb.get_evals_result()
    metrics['elapsed_time'] = t.elapsed
    metrics['n_estimators'] = cb_kwargs['n_estimators']

    # Specify the file path where you want to save the metrics
    metrics_file_path = exp.results_dir.joinpath('catboost_metrics.json')
    metrics_file_path.write(metrics, indent=4)


def get_paths():

    ip = get_public_ip()

    if ip.startswith('199'):
        data_path = '/mnt/data/dataset/tabular/'
        logs_path = '/home/shared/results/'
    else:
        data_path = '/home/mlspeech/elads/data/tabular/data/'
        # logs_path = '/dsi/shared/elads/elads/data/tabular/results/'
        logs_path = '/home/mlspeech/elads/data/tabular/results/'

    return data_path, logs_path


if __name__ == '__main__':

    data_path, logs_path = get_paths()

    kwargs_base = dict(algorithm='catboost_timing',
                       data_path=data_path,
                       logs_path=logs_path, cb_n_estimators=100,
                       copy_code=False, dynamic_masking=False,
                       tensorboard=True, stop_at=0.98, n_gpus=1, device='cpu', n_quantiles=6, catboost=True)

    # kwargs_base = dict(algorithm='debug', data_path=data_path, logs_path=logs_path,
    #                    scheduler='one_cycle', device_placement=True, device=1, n_gpus=1,
    #                    copy_code=False, dynamic_masking=False, comet=False, tensorboard=True, n_epochs=10,
    #                    n_quantiles=6, label_smoothing=.2,
    #                    model_dtype='float16', training_framework='torch', federated_runner=False,
    #                    compile_train=False, sparse_embedding=False, compile_network=False, mlflow=False,
    #                    n_decoder_layers=4)

    kwargs_all = {}

    kwargs_all['california_housing'] = dict(batch_size=128, stop_at=-.43)
    # kwargs_all['adult'] = dict(batch_size=128)
    # kwargs_all['helena'] = dict(batch_size=256, mask_rate=0.25, dropout=0.25, transformer_dropout=.25,
    #                             minimal_mask_rate=.2, maximal_mask_rate=.4,
    #                             label_smoothing=.25, n_quantiles=6, dynamic_masking=False, cb_depth=12)
    # kwargs_all['jannis'] = dict(batch_size=256, cb_depth=14)
    # kwargs_all['higgs_small'] = dict(batch_size=256, cb_depth=14)
    # kwargs_all['aloi'] = dict(batch_size=256, cb_depth=10)

    # kwargs_all['year'] = dict(batch_size=256, emb_dim=128, n_decoder_layers=4, n_encoder_layers=4,
    #                       n_quantiles=7, n_rules=128, n_transformer_head=4, transformer_hidden_dim=256, cb_depth=10)

    # kwargs_all['year'] = dict(batch_size=512, cb_depth=14)
    # kwargs_all['covtype'] = dict(batch_size=512, n_quantiles=12, cb_depth=14)

    for k in kwargs_all.keys():

        logger.info(f"Starting a new experiment with dataset: {k}")
        hparams = {**kwargs_base}
        hparams.update(kwargs_all[k])
        hparams['dataset_name'] = k
        hparams['identifier'] = k
        hparams = TabularConfig(hparams)

        exp = Experiment(hparams)
        dataset = TabularDataset(hparams)


        # logger.info(f"Training a RuleNet predictor")
        # # net = TabularTransformer(hparams, dataset.n_classes, dataset.n_tokens, dataset.cat_mask)
        # # alg = DeepTabularAlg(hparams, networks=net)
        #
        # alg = exp.fit(alg=DeepTabularAlg, dataset=TabularDataset,
        #               alg_kwargs={'net_kwargs': {'n_classes': dataset.n_classes, 'n_tokens': dataset.n_tokens,
        #                                                                       'cat_mask': dataset.cat_mask},
        #                           'task_type': dataset.task_type,
        #                           'y_sigma': dataset.y_sigma},)
        #
        # logger.info(f"Training finished, reloading best model")
        #
        # exp.reload_checkpoint(alg)
        # alg.set_best_masking()
        #
        # predictions = alg.evaluate('validation')
        # logger.info(f"Validation objective: {predictions.statistics['validation'][Types.scalar]['objective'].values}")
        # exp.results_dir.joinpath('predictions.pt').write(predictions)
        #
        # # store to bundle
        #
        # path = '/workspace/serve/bundle'
        # logger.info(f"Storing bundle to: {path}")
        # AutoBeam.to_bundle(alg, path)

        logger.info(f"Training a Catboost predictor")
        train_catboost(dataset, exp)


