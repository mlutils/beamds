import os
from ray.tune.schedulers import ASHAScheduler

n_jobs = 4

from beam.hpo import HPOConfig, RayHPO
from beam.algorithm import CBAlgorithm, CatboostExperimentConfig
from beam import beam_logger as logger, beam_path

if __name__ == '__main__':

    algorithm_name = 'hpo_cb'

    kwargs_base = dict(algorithm=algorithm_name)

    hpo_config = HPOConfig(n_trials=1000, train_timeout=60 * 60 * 24, gpus_per_trial=1,
                           cpus_per_trial=6, n_jobs=n_jobs, hpo_path=os.path.join('/tmp', 'hpo'))

    run_names = {}

    kwargs_all = {}

    kwargs_all['covtype'] = dict(batch_size=512)

    logger.info(f"Starting a new experiment with dataset: {k}")
    hparams = {**kwargs_base}
    hparams.update(kwargs_all[k])
    hparams['dataset_name'] = k
    hparams['identifier'] = k
    hparams = TabularConfig(hparams)

    dataset = TabularDataset(hparams)
    # dataset = TabularDataset

    logger.info(f"Training a RuleNet predictor")

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

