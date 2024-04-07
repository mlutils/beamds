import os

import torch
from ray.tune.schedulers import ASHAScheduler

# available_devices = [0, 1, 2, 3]
# available_devices = [0, 1]
# n_jobs = len(available_devices)
n_jobs = 4
# available_devices = [0]
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in available_devices])
# n_jobs = 1


from examples.tabular_example import get_paths
from src.beam.hpo import HPOConfig, RayHPO
from src.beam.tabular import TabularDataset, TabularTransformer, TabularConfig, DeepTabularAlg
from src.beam import beam_logger as logger, beam_path

if __name__ == '__main__':

    data_path, logs_path = get_paths()
    max_quantiles = 100
    # max_quantiles = 2
    n_decoder_layers = 4
    # n_decoder_layers = 0
    dropout = 0.
    transformer_dropout = 0.

    # algorithm_name = 'hpo_no_decoder'
    # algorithm_name = 'hpo_no_dropout'
    # algorithm_name = 'hpo_no_quantiles'
    algorithm_name = 'hpo_full'

    kwargs_base = dict(algorithm=algorithm_name, data_path=data_path, logs_path=logs_path,
                       copy_code=False, dynamic_masking=False, early_stopping_patience=30, n_epochs=100,
                       n_quantiles=max_quantiles, dropout=dropout, transformer_dropout=transformer_dropout,
                       tensorboard=False, stop_at=0.98, n_gpus=1, device=0, label_smoothing=.2,
                       n_decoder_layers=n_decoder_layers)

    hpo_config = HPOConfig(n_trials=1000, train_timeout=60 * 60 * 24, gpus_per_trial=1,
                           cpus_per_trial=6, n_jobs=n_jobs, hpo_path=os.path.join(logs_path, 'hpo'))

    run_names = {}
    # run_names = dict(aloi='/dsi/shared/elads/elads/data/tabular/results/hpo/aloi_hp_optimization_20240113_172324')

    kwargs_all = {}

    # kwargs_all['california_housing'] = dict(batch_size=128)
    # kwargs_all['adult'] = dict(batch_size=128)
    # kwargs_all['helena'] = dict(batch_size=256)
    # kwargs_all['jannis'] = dict(batch_size=256)
    # kwargs_all['higgs_small'] = dict(batch_size=256)
    # kwargs_all['aloi'] = dict(batch_size=256)
    # kwargs_all['year'] = dict(batch_size=512)
    kwargs_all['covtype'] = dict(batch_size=512)

    for k in kwargs_all.keys():

        logger.info(f"Starting a new experiment with dataset: {k}")
        hparams = {**kwargs_base}
        hparams.update(kwargs_all[k])
        hparams['dataset_name'] = k
        hparams['identifier'] = k
        hparams = TabularConfig(hparams)

        dataset = TabularDataset(hparams)
        # dataset = TabularDataset

        logger.info(f"Training a RuleNet predictor")
        # net = TabularTransformer(hparams, dataset.n_classes, dataset.n_tokens, dataset.cat_mask)
        # alg = DeepTabularAlg(hparams, networks=net)

        # study = beam_hpo('ray', hparams, alg=DeepTabularAlg, dataset=dataset, print_results=False,
        #                     hpo_config=hpo_config,
        #                 alg_kwargs={'net_kwargs': {'n_classes': dataset.n_classes, 'n_tokens': dataset.n_tokens,
        #                                           'cat_mask': dataset.cat_mask, }})

        def post_train_hook(alg=None, report=None, hparams=None, suggestion=None, experiment=None, **kwargs):
            experiment.reload_best_checkpoint(alg)
            predictions = alg.evaluate('validation')
            objective = predictions.statistics.data['objective']
            logger.info(f"Post-train validation objective: {objective}")
            alg.report(objective=float(objective), epoch=alg.epoch+1)

        study = RayHPO(hparams, alg=DeepTabularAlg, dataset=TabularDataset, print_results=False,
                         hpo_config=hpo_config, post_train_hook=post_train_hook,
                         alg_kwargs={'net_kwargs': {'n_classes': dataset.n_classes, 'n_tokens': dataset.n_tokens,
                                                    'cat_mask': dataset.cat_mask, },
                                     'task_type': dataset.task_type,
                                     'y_sigma': dataset.y_sigma})

        study.uniform('lr-dense', 1e-4, 1e-2)
        study.uniform('lr-sparse', 1e-3, 1e-1)
        study.categorical('batch_size', [hparams.batch_size // 4, hparams.batch_size // 2, hparams.batch_size,
                                    hparams.batch_size * 2])
        study.uniform('dropout', 0., 0.5)
        study.categorical('emb_dim', [64, 128, 256])
        study.categorical('n_rules', [64, 128, 256])
        study.categorical('n_quantiles', [2, 6, 10, 16, 20, 40, max_quantiles])
        study.categorical('n_encoder_layers ', [1, 2, 4, 8])
        study.categorical('n_decoder_layers', [1, 2, 4, 8])
        study.categorical('n_transformer_head', [1, 2, 4, 8])
        study.categorical('transformer_hidden_dim', [128, 256, 512])

        study.uniform('mask_rate', 0., 0.4)
        study.uniform('rule_mask_rate', 0., 0.4)
        study.uniform('transformer_dropout', 0., 0.4)
        study.uniform('label_smoothing', 0., 0.4)

        scheduler = ASHAScheduler(
            # metric="objective",  # Replace with your objective metric
            # mode="max",  # Use "max" or "min" depending on your objective
            max_t=kwargs_base['n_epochs']+2,  # Adjust this based on your maximum iterations
            grace_period=20,  # The minimum number of iterations for a trial
            reduction_factor=2,  # Factor for reducing the number of trials each round
            # time_attr="iter"  # Set to 'iter' to match your progress tracking
        )

        # start pruning after max_t epochs
        tune_config_kwargs = dict(scheduler=scheduler)

        other_kwargs = {}
        if k in run_names.keys():
            other_kwargs['restore_path'] = run_names[k]

        study.run(tune_config_kwargs=tune_config_kwargs, **other_kwargs)

        logger.info(f"Done HPO for dataset: {k}")
