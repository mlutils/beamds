import os
from ray.tune.schedulers import ASHAScheduler

from beam.hpo import HPOConfig, RayHPO, OptunaHPO
from beam.algorithm import CBAlgorithm, CatboostExperimentConfig
from beam import beam_logger as logger


def optimize_catboost(dataset, hpo_config, config):

    logger.info(f"Training a RuleNet predictor")

    study = RayHPO(config, alg=CBAlgorithm, dataset=dataset, print_results=False,  hpo_config=hpo_config)
    # study = OptunaHPO(config, alg=CBAlgorithm, dataset=dataset, print_results=False,  hpo_config=hpo_config)

    # study.linspace('iterations', 100, 1000)
    study.linspace('iterations', 50, 200)

    study.loguniform('learning_rate', 0.001, 1.)
    # study.linspace('depth', 1, 16)
    study.linspace('depth', 3, 6)

    study.loguniform('l2_leaf_reg', 0.1, 10.)
    study.categorical('boosting_type', ['Ordered', 'Plain'])
    study.categorical('auto_class_weights', ['Balanced', 'SqrtBalanced', 'None'])
    study.uniform('bagging_fraction', 0., 1.)
    study.loguniform('bagging_temperature', 0.1, 10.)
    study.logspace('border_count', 5, 10, base=2, dtype=int)
    study.categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'No', 'MVS'])
    study.categorical('feature_border_type',
                      ['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'GreedyLogSum', 'MinEntropy'])
    study.categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide'])
    study.categorical('leaf_estimation_backtracking', ['No', 'AnyImprovement', 'Armijo'])
    study.logspace('leaf_estimation_iterations', 0, 2, 20, dtype=int)
    study.categorical('leaf_estimation_method', ['Newton', 'Gradient'])
    study.linspace('max_leaves', 2, 64)
    study.loguniform('min_data_in_leaf', 1, 100)
    study.linspace('early_stopping_rounds', 1, 100)
    study.categorical('od_type', ['IncToDec', 'Iter'])
    study.linspace('od_wait', 1, 100)
    study.loguniform('od_pval', 1e-10, 1e-2)
    study.categorical('sampling_frequency', ['PerTree', 'PerTreeLevel'])
    study.loguniform('l1_leaf_reg', 0.1, 10.)
    study.loguniform('rsm', 0.1, 1.)
    study.loguniform('random_strength', 0.1, 20.)
    study.logspace('one_hot_max_size', 0, 2, 5, base=10, dtype=int)

    scheduler = None
    if hpo_config.get('max_iterations') is not None:
        scheduler = ASHAScheduler(
            # metric="objective",  # Replace with your objective metric
            # mode="max",  # Use "max" or "min" depending on your objective
            max_t=hpo_config.get('max_iterations'),  # Adjust this based on your maximum iterations
            grace_period=hpo_config.get('grace_period'),  # The minimum number of iterations for a trial
            reduction_factor=hpo_config.get('iterations'),  # Factor for reducing the number of trials each round
            # time_attr="iter"  # Set to 'iter' to match your progress tracking
        )

    # start pruning after max_t epochs
    tune_config_kwargs = dict(scheduler=scheduler)

    study.run(tune_config_kwargs=tune_config_kwargs)
    # study.run()

    logger.info(f"Done HPO")


def main():

    from beam.dataset import TabularDataset
    from examples.cb_example import preprocess_covtype

    hpo_config = HPOConfig(n_trials=40, train_timeout=60 * 60 * 24, gpus_per_trial=.1,
                           cpus_per_trial=6, n_jobs=10, hpo_path=os.path.join('/tmp', 'hpo'))

    config = CatboostExperimentConfig(loss_function='MultiClass', objective='accuracy', device=0)
    data = preprocess_covtype()
    dataset = TabularDataset(x=data['x'], y=data['y'], cat_features=['Wilderness_Area', 'Soil_Type'])

    optimize_catboost(dataset, hpo_config, config)


if __name__ == '__main__':
    main()


