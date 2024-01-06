import os

from ray.tune.schedulers import ASHAScheduler

available_devices = [0, 1, 2, 3]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in available_devices])
n_jobs = len(available_devices)


from examples.tabular_example import get_paths
from src.beam.hpo import beam_hpo, HPOConfig
from src.beam.tabular import TabularDataset, TabularTransformer, TabularConfig, DeepTabularAlg
from src.beam import beam_logger as logger, beam_path

if __name__ == '__main__':

    data_path, logs_path = get_paths()

    kwargs_base = dict(algorithm='hpo_debug', data_path=data_path, logs_path=logs_path,
                       copy_code=False, dynamic_masking=False, early_stopping_patience=30, n_epochs=100,
                       tensorboard=False, stop_at=0.98, n_gpus=1, device=0, n_quantiles=6, label_smoothing=.2)

    hpo_config = HPOConfig(n_trials=1000, train_timeout=60 * 60 * 24, gpus_per_trial=1,
                           cpus_per_trial=10, n_jobs=n_jobs, hpo_path=os.path.join(logs_path, 'hpo'))

    kwargs_all = {}

    # kwargs_all['california_housing'] = dict(batch_size=128)
    # kwargs_all['adult'] = dict(batch_size=128)
    # kwargs_all['helena'] = dict(batch_size=256, mask_rate=0.25, dropout=0.25, transformer_dropout=.25,
    #                             minimal_mask_rate=.2, maximal_mask_rate=.4,
    #                             label_smoothing=.25, n_quantiles=6, dynamic_masking=False)
    # kwargs_all['jannis'] = dict(batch_size=256)
    # kwargs_all['higgs_small'] = dict(batch_size=256)
    # kwargs_all['aloi'] = dict(batch_size=256)
    # kwargs_all['year'] = dict(batch_size=512)
    kwargs_all['covtype'] = dict(batch_size=1024, n_quantiles=10)

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
        cwd = beam_path(os.getcwd())
        study = beam_hpo('ray', hparams, alg=DeepTabularAlg, dataset=TabularDataset, print_results=False,
                         hpo_config=hpo_config,
                         alg_kwargs={'net_kwargs': {'n_classes': dataset.n_classes, 'n_tokens': dataset.n_tokens,
                                                    'cat_mask': dataset.cat_mask, }})

        study.uniform('lr-dense', 1e-4, 1e-2)
        study.uniform('lr-sparse', 1e-3, 1e-1)
        study.categorical('batch_size', [hparams.batch_size // 4, hparams.batch_size // 2, hparams.batch_size,
                                    hparams.batch_size * 2])
        study.uniform('dropout', 0., 0.5)
        study.categorical('emb_dim', [64, 128, 256])
        study.categorical('n_rules', [64, 128, 256])
        study.categorical('n_quantiles', [2, 6, 10, 16, 20, 40, 100])
        study.categorical('n_encoder_layers', [1, 2, 4, 8])
        study.categorical('n_decoder_layers', [1, 2, 4, 8])
        study.categorical('n_transformer_head', [1, 2, 4, 8])
        study.categorical('transformer_hidden_dim', [128, 256, 512])

        study.uniform('mask_rate', 0., 0.4)
        study.uniform('rule_mask_rate', 0., 0.4)
        study.uniform('transformer_dropout', 0., 0.4)
        study.uniform('label_smoothing', 0., 0.4)

        tune_config_kwargs = dict(scheduler=ASHAScheduler())
        study.run()

        logger.info(f"Done HPO for dataset: {k}")
