
from ..config import BeamParam, DeviceConfig, ExperimentConfig
from ..similarity import SimilarityConfig, TFIDFConfig


class TextGroupExpansionConfig(SimilarityConfig, TFIDFConfig):

    # "en_core_web_trf"

    defaults = {
        'chunksize': 1000,
        'n_workers': 40,
        'mp_method': 'apply_async',
        'store_chunk': True,
        'store_path': None,
        'store_suffix': '.parquet',
        'override': False,
        'sparse_framework': 'scipy',
    }
    parameters = [
        BeamParam('tokenizer', type=str, default="BAAI/bge-base-en-v1.5", help='Tokenizer model'),
        BeamParam('dense-model', type=str, default="BAAI/bge-base-en-v1.5", help='Dense model for text similarity'),
        BeamParam('dense_model_device', type=str, default='cuda', help='Device for dense model'),
        BeamParam('tokenizer-chunksize', type=int, default=10000, help='Chunksize for tokenizer'),
        BeamParam('batch_size', int, 32, 'Batch size for dense model'),
        BeamParam('k-sparse', int, 50, 'Number of sparse similarities to include in the dataset'),
        BeamParam('k-dense', int, 50, 'Number of dense similarities to include in the dataset'),
        BeamParam('threshold', float, 0.5, 'Threshold for prediction model'),
        BeamParam('svd-components', int, 64, 'Number of PCA components to use to compress the tfidf vectors'),
        BeamParam('pca-components', int, 64, 'Number of PCA components to use to compress the dense vectors'),
        BeamParam('pu-n-estimators', int, 20, 'Number of estimators for the PU classifier'),
        BeamParam('pu-verbose', int, 10, 'Verbosity level for the PU classifier'),
        BeamParam('classifier-type', str, None, 'can be one of [None, catboost, rf]'),
        BeamParam('early_stopping_rounds', int, None, 'Early stopping rounds for the classifier'),
    ]

# class CatBoostClassifier(iterations=None,
#                          learning_rate=None,
#                          depth=None,
#                          l2_leaf_reg=None,
#                          model_size_reg=None,
#                          rsm=None,
#                          loss_function=None,
#                          border_count=None,
#                          feature_border_type=None,
#                          per_float_feature_quantization=None,
#                          input_borders=None,
#                          output_borders=None,
#                          fold_permutation_block=None,
#                          od_pval=None,
#                          od_wait=None,
#                          od_type=None,
#                          nan_mode=None,
#                          counter_calc_method=None,
#                          leaf_estimation_iterations=None,
#                          leaf_estimation_method=None,
#                          thread_count=None,
#                          random_seed=None,
#                          use_best_model=None,
#                          verbose=None,
#                          logging_level=None,
#                          metric_period=None,
#                          ctr_leaf_count_limit=None,
#                          store_all_simple_ctr=None,
#                          max_ctr_complexity=None,
#                          has_time=None,
#                          allow_const_label=None,
#                          classes_count=None,
#                          class_weights=None,
#                          auto_class_weights=None,
#                          one_hot_max_size=None,
#                          random_strength=None,
#                          name=None,
#                          ignored_features=None,
#                          train_dir=None,
#                          custom_loss=None,
#                          custom_metric=None,
#                          eval_metric=None,
#                          bagging_temperature=None,
#                          save_snapshot=None,
#                          snapshot_file=None,
#                          snapshot_interval=None,
#                          fold_len_multiplier=None,
#                          used_ram_limit=None,
#                          gpu_ram_part=None,
#                          allow_writing_files=None,
#                          final_ctr_computation_mode=None,
#                          approx_on_full_history=None,
#                          boosting_type=None,
#                          simple_ctr=None,
#                          combinations_ctr=None,
#                          per_feature_ctr=None,
#                          task_type=None,
#                          device_config=None,
#                          devices=None,
#                          bootstrap_type=None,
#                          subsample=None,
#                          sampling_unit=None,
#                          dev_score_calc_obj_block_size=None,
#                          max_depth=None,
#                          n_estimators=None,
#                          num_boost_round=None,
#                          num_trees=None,
#                          colsample_bylevel=None,
#                          random_state=None,
#                          reg_lambda=None,
#                          objective=None,
#                          eta=None,
#                          max_bin=None,
#                          scale_pos_weight=None,
#                          gpu_cat_features_storage=None,
#                          data_partition=None
#                          metadata=None,
#                          early_stopping_rounds=None,
#                          cat_features=None,
#                          grow_policy=None,
#                          min_data_in_leaf=None,
#                          min_child_samples=None,
#                          max_leaves=None,
#                          num_leaves=None,
#                          score_function=None,
#                          leaf_estimation_backtracking=None,
#                          ctr_history_unit=None,
#                          monotone_constraints=None,
#                          feature_weights=None,
#                          penalties_coefficient=None,
#                          first_feature_use_penalties=None,
#                          model_shrink_rate=None,
#                          model_shrink_mode=None,
#                          langevin=None,
#                          diffusion_temperature=None,
#                          posterior_sampling=None,
#                          boost_from_average=None,
#                          text_features=None,
#                          tokenizers=None,
#                          dictionaries=None,
#                          feature_calcers=None,
#                          text_processing=None,
#                          fixed_binary_splits=None)


class CatboostConfig(DeviceConfig):
    # catboost
    parameters = [
        BeamParam('cb-task', str, 'classification', 'The task type for the catboost model '
                                                    '[classification|regression|ranking]'),
        BeamParam('log-frequency', int, 10, 'The frequency (in epochs) of the logging for the catboost model'),

        BeamParam('loss_function', str, 'Logloss', 'The loss function for the catboost model'),
        # learning rate is drawn from other configurations
        BeamParam('n_estimators', int, 200, 'The number of trees in the catboost model', tags='tune'),
        BeamParam('l2_leaf_reg', float, 1e-2, 'The L2 regularization for the catboost model', tags='tune'),
        BeamParam('border_count', int, 128, 'The border count for the catboost model', tags='tune'),
        BeamParam('depth', int, 6, 'The depth of the trees in the catboost model', tags='tune'),
        BeamParam('random_strength', float, .5, 'The random strength for the catboost model', tags='tune'),
        BeamParam('lr', float, 1e-1, 'The learning rate for the catboost model', tags='tune'),
        BeamParam('eval_metric', str, None, 'The evaluation metric for the catboost model, '
                                               'if None, it is set to RMSE for regression and '
                                               'Accuracy for classification'),
        BeamParam('custom_metric', list, None, 'The custom metric for the catboost model, '
                                                  'if None, it is set to MAE, MAPE for regression and '
                                                  'Precision, Recall for classification'),
        BeamParam('feature_border_type', str,
                  'GreedyLogSum', 'The feature border type for the catboost model. '
                                  'Possible values: [Median, Uniform, UniformAndQuantiles, '
                                  'MaxLogSum, GreedyLogSum, MinEntropy].'),
        BeamParam('per_float_feature_quantization', str, None,
                  'The per float feature quantization for the catboost model. '
                  'See https://catboost.ai/en/docs/references/training-parameters/quantization'),
        BeamParam('input_borders', str, None, 'path to input file with borders used in numeric features binarization.'),
        BeamParam('output_borders', str, None, 'path to output file with borders used in numeric features binarization.'),
        BeamParam('fold_permutation_block', int, None, 'The fold permutation block for the catboost model'),
        BeamParam('od_pval', float, None, 'The od pval for the catboost model'),
        BeamParam('od_wait', int, None, 'The od wait for the catboost model'),
        BeamParam('od_type', str, None, 'The od type for the catboost model'),
        BeamParam('nan_mode', str, None, 'The nan mode for the catboost model'),
        BeamParam('counter_calc_method', str, None, 'The counter calc method for the catboost model'),
        BeamParam('leaf_estimation_iterations', int, None, 'The leaf estimation iterations for the catboost model'),
        BeamParam('leaf_estimation_method', str, None, 'The leaf estimation method for the catboost model'),
        BeamParam('thread_count', int, None, 'The thread count for the catboost model'),
        BeamParam('use_best_model', bool, None, 'The use best model for the catboost model'),

    ]


class CatboostExperimentConfig(CatboostConfig, ExperimentConfig):
    defaults = {'project': 'cb_beam', 'algorithm': 'CBAlgorithm'}
