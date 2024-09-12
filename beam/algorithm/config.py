
from ..config import BeamParam, DeviceConfig, ExperimentConfig, BeamConfig
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


class CatboostConfig(DeviceConfig):
    """
    CatBoost configuration with detailed parameter documentation.
    References:
    - https://catboost.ai/docs/references/training-parameters.html
    - https://docs.aws.amazon.com/sagemaker/latest/dg/catboost-hyperparameters.html
    """

    defaults = {'objective': None, 'objective_to_report': 'best'}

    # CatBoost parameters
    parameters = [
        BeamParam(
            'cb-task',
            str,
            'classification',
            'The task type for the CatBoost model. '
            'Default: classification. '
            'Options: [classification, regression, ranking]. '
            'Determines the type of task and influences the loss function and evaluation metrics.'
        ),
        BeamParam(
            'log-frequency',
            int,
            10,
            'The frequency (in iterations) of logging for the CatBoost model. '
            'Default: 10. '
            'Range: [1, ∞). '
            'Affects how often training progress is reported.'
        ),

        # Core parameters
        BeamParam(
            'loss_function',
            str,
            'Logloss',
            'The loss function for the CatBoost model. '
            'Default: Logloss for classification. '
            'Options: [Logloss, RMSE, MAE, Quantile, MAPE, Poisson, etc.]. '
            'Determines the optimization objective and affects model predictions.'
        ),
        BeamParam(
            'eval_metric',
            str,
            None,
            'The evaluation metric for the CatBoost model. '
            'Default: Auto-detected based on task type. '
            'Options: [Accuracy, AUC, RMSE, MAE, etc.]. '
            'Used for evaluating the performance of the model on validation data.'
        ),
        BeamParam(
            'custom_metric',
            list,
            None,
            'The custom metric for the CatBoost model. '
            'Default: None. '
            'Options: [Precision, Recall, F1, etc.]. '
            'Provides additional metrics for evaluation.'
        ),

        # Training parameters
        BeamParam(
            'iterations',
            int,
            None,
            'The number of trees (iterations) in the CatBoost model. '
            'Default: 1000. '
            'Range: [1, ∞). '
            'Higher values may improve performance but can increase training time and risk overfitting.'
        ),
        BeamParam(
            'learning_rate',
            float,
            None,
            'The learning rate for the CatBoost model. '
            'Default: 0.03. '
            'Range: (0.0, 1.0]. '
            'Controls the step size at each iteration while moving towards a minimum of the loss function.'
        ),
        BeamParam(
            'depth',
            int,
            None,
            'The depth of the trees in the CatBoost model. '
            'Default: 6. '
            'Range: [1, 16]. '
            'Deeper trees can capture more complex patterns but may lead to overfitting.'
        ),
        BeamParam(
            'l2_leaf_reg',
            float,
            None,
            'The L2 regularization term on the cost function. '
            'Default: 3.0. '
            'Range: (0, ∞). '
            'Helps prevent overfitting by penalizing large weights.'
        ),

        # Overfitting detection
        BeamParam(
            'od_pval',
            float,
            None,
            'The threshold for the overfitting detector. '
            'Default: None. '
            'Range: (0, 1). '
            'Stops training if the performance on the validation set does not improve by this value.'
            'For best results, it is recommended to set a value in the range [ 1e-10 ; 1e-2 ]'
        ),
        BeamParam(
            'od_wait',
            int,
            None,
            'Number of iterations to wait after the overfitting criterion is reached before stopping training. '
            'Default: 20. '
            'Range: [1, ∞). '
            'Prevents premature stopping by allowing continued training for a set number of iterations.'
        ),
        BeamParam(
            'od_type',
            str,
            None,
            'The overfitting detection type. '
            'Default: IncToDec. '
            'Options: [IncToDec, Iter]. '
            'Determines how overfitting is detected during training.'
        ),

        # Regularization parameters
        BeamParam(
            'bagging_temperature',
            float,
            None,
            'Controls the Bayesian bootstrap and helps in reducing overfitting by using random weights. '
            'Default: 1.0. '
            'Range: [0.0, ∞). '
            'Higher values increase randomness, helping to reduce overfitting.'
        ),
        BeamParam(
            'random_strength',
            float,
            None,
            'The amount of randomness to use for scoring splits when the tree structure is selected. '
            'Use this parameter to avoid overfitting the model.'
            'Default: 1.0. '
            'Range: [0.0, ∞). '
            'Adds randomness to scoring splits, helping prevent overfitting.'
        ),

        # Feature processing
        BeamParam(
            'border_count',
            int,
            None,
            'The number of splits for numerical features (max_bin). '
            'Default: The default value depends on the processing unit type and other parameters: '
            'CPU: 254 '
            'GPU in PairLogitPairwise and YetiRankPairwise modes: 32 '
            'GPU in all other modes: 128 '
            'Range: [1, 65535]. '
            'Affects the granularity of feature discretization; higher values can improve accuracy but increase complexity.'
        ),
        BeamParam(
            'feature_border_type',
            str,
            None,
            'The feature border type. '
            'Default: GreedyLogSum. '
            'Options: [Median, Uniform, UniformAndQuantiles, MaxLogSum, GreedyLogSum, MinEntropy]. '
            'Determines how feature borders are selected, impacting split decisions.'
        ),
        BeamParam(
            'per_float_feature_quantization',
            str,
            None,
            'The per float feature quantization. '
            'Default: None. '
            'See: https://catboost.ai/en/docs/references/training-parameters/quantization. '
            'Allows custom quantization for specific features.'
        ),

        # Advanced tree options
        BeamParam(
            'grow_policy',
            str,
            None,
            'Defines how to perform greedy tree construction. '
            'Default: SymmetricTree. '
            'Options: [SymmetricTree, Depthwise, Lossguide]. '
            'Determines the strategy for tree growth, affecting complexity and interpretability.'
        ),
        BeamParam(
            'max_leaves',
            int,
            None,
            'The maximum number of leaves in the resulting tree. '
            'Default: None. '
            'Range: [2, 64]. '
            'Applicable for Lossguide grow policy; limits the complexity of the tree.'
        ),

        # Sampling and randomness
        BeamParam(
            'rsm',
            float,
            None,
            'Random subspace method for feature selection. '
            'Default: 1.0. '
            'Range: (0.0, 1.0]. '
            'Percentage of features used at each split selection, allowing randomness in feature selection.'
        ),
        BeamParam(
            'sampling_frequency',
            str,
            None,
            'Frequency to sample weights and objects during tree building. '
            'Default: PerTree. '
            'Options: [PerTree, PerTreeLevel]. '
            'Determines how often samples are drawn during training.'
        ),
        BeamParam(
            'bootstrap_type',
            str,
            None,
            'The bootstrap type. '
            'Default: Bayesian. '
            'Options: [Bayesian, Bernoulli, No, MVS]. '
            'Controls how samples are drawn for training, affecting robustness and variance.'
        ),

        # Leaf estimation
        BeamParam(
            'leaf_estimation_iterations',
            int,
            None,
            'The number of iterations to calculate values in leaves. '
            'Default: 1. '
            'Range: [1, ∞). '
            'Higher values improve accuracy at the cost of increased training time.'
        ),
        BeamParam(
            'leaf_estimation_method',
            str,
            None,
            'The method used to calculate values in leaves. '
            'Default: Newton. '
            'Options: [Newton, Gradient]. '
            'Determines the approach for estimating leaf values, affecting convergence speed and accuracy.'
        ),

        # Logging and output
        BeamParam(
            'snapshot_interval',
            int,
            600,
            'The snapshot interval for model saving [in seconds]. '
            'Default: 600. '
            'Range: [1, ∞). '
            'Controls how often model snapshots are saved, useful for resuming training.'
        ),
        BeamParam(
            'boosting_type',
            str,
            'Plain',
            'Controls the boosting scheme. '
            'Default: Plain. '
            'Options: [Ordered, Plain]. '
            'Ordered is used to eliminate the effect of a training set order.'
        ),
        BeamParam(
            'allow_const_label',
            bool,
            False,
            'Allows training on a dataset with constant labels. '
            'Default: False. '
            'Useful for experimentation or testing.'
        ),

        # Training parameters
        BeamParam(
            'auto_class_weights',
            str,
            None,
            'Automatically calculates class weights for imbalanced datasets. '
            'Default: None. '
            'Options: [Balanced, SqrtBalanced, None].'
        ),

        # Regularization parameters
        BeamParam(
            'l1_leaf_reg',
            float,
            None,
            'L1 regularization term on weights. '
            'Default: 0.0. '
            'Range: (0, ∞). '
            'Helps prevent overfitting by penalizing large weights.'
        ),

        # Feature processing
        BeamParam(
            'one_hot_max_size',
            int,
            None,
            'Maximum size of the categorical feature for one-hot encoding. '
            'Default: 2. '
            'Range: [1, ∞). '
            'Larger sizes use a more efficient embedding.'
        ),

        # Advanced tree options
        BeamParam(
            'min_data_in_leaf',
            int,
            None,
            'Minimum number of samples per leaf. '
            'Default: 1. '
            'Range: [1, ∞). '
            'Controls leaf size and can affect overfitting and generalization.'
        ),

        # Sampling and randomness
        BeamParam(
            'bagging_fraction',
            float,
            None,
            'Fraction of samples to use in each iteration. '
            'Default: 1.0. '
            'Range: (0.0, 1.0]. '
            'Controls randomness and variance.'
        ),

        # Leaf estimation
        BeamParam(
            'leaf_estimation_backtracking',
            str,
            None,
            'Backtracking type used for leaf estimation. '
            'Default: AnyImprovement. '
            'Options: [No, AnyImprovement, Armijo]. '
            'Affects convergence and accuracy.'
        ),

    ]


class CatboostExperimentConfig(CatboostConfig, ExperimentConfig):
    defaults = {'project': 'cb_beam', 'algorithm': 'CBAlgorithm'}
