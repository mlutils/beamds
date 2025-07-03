from ..config import BeamConfig, BeamParam

class BayesianConfig(BeamConfig):
    parameters = [
        BeamParam(name="acquisition_function", type=str, default="LogExpectedImprovement",
                  help="Acquisition function to use for Bayesian optimization. Choices: [ExpectedImprovement, "
                       "ProbabilityOfImprovement, UpperConfidenceBound, PosteriorMean]"),
        BeamParam(name="n_initial_points", type=int, default=5,
                  help="Number of initial points to sample before starting Bayesian optimization."),
        BeamParam(name="buffer_size", type=int, default=int(1e6),
                  help="Size of the buffer to store samples for Bayesian optimization."),
        BeamParam(name="device", type=str, default="cpu",
                  help="Device to use for Bayesian optimization. Choices: [cpu, cuda, mps, 0, 1, 2, ...] "
                       "where numbers represent GPU indices."),
        BeamParam(name="dtype", type=str, default="float32",
                  help="Data type to use for Bayesian optimization. Choices: [float32, float64]"),
        BeamParam(name="likelihood", type=str, default="GaussianLikelihood",
                  help="Likelihood to use for Bayesian optimization. Choices: [GaussianLikelihood, "
                       "BernoulliLikelihood, PoissonLikelihood]"),
        BeamParam(name="likelihood_kwargs", type=dict, default={'noise': 0.1},
                  help="Additional keyword arguments for the likelihood."),
        BeamParam(name="acquisition_kwargs", type=dict, default={},
                  help="Additional keyword arguments for the acquisition function."),
        BeamParam(name="num_restarts", type=int, default=200,
                  help="Number of restarts for the optimization process."),
        BeamParam(name="sequential_opt", type=bool, default=True,
                  help="Whether to perform sequential optimization or not."),
        BeamParam(name="raw_samples", type=int, default=512,
                  help="Number of raw samples to generate for Bayesian optimization."),
        BeamParam(name="aquisition_options", type=dict, default={},
                  help="Additional options for the acquisition function optimization."),
        BeamParam(name="batch_size", type=int, default=1,
                  help="Batch size for the optimization process."),
        BeamParam(name='global_upper_bound', type=float, default=1e6,
                    help="Global upper bound for the optimization process. Used to limit the search space."),
        BeamParam(name="n_categorical_features_threshold", type=int, default=5,
                  help="Threshold for the number of categorical features to use a different acquisition function "
                       "(optimize_acqf_mixed_alternating instead of optimize_acqf_mixed)."),
        BeamParam(name="start_fitting_after_n_points", type=int, default=10,
                    help="Number of points after which to start fitting the model."),
        BeamParam(name="fit_every_n_points", type=int, default=100,
                    help="Number of points after which to re-fit the model again during optimization."),
        BeamParam(name="incremental_fit", type=str, default='fantasy',
                  help="Method to use for incremental fitting. Choices: [fantasy, full, none]. "
                       "Fantasy uses fantasy points to update the model without re-fitting."),
        BeamParam(name="continuous_kernel", type=str, default=None,
                    help="Kernel to use for continuous features in Bayesian optimization. Choices: [RBFKernel, "
                         "MaternKernel, RationalQuadraticKernel]"),
        BeamParam(name="continuous_kernel_kwargs", type=dict, default={},
                    help="Additional keyword arguments for the continuous kernel."),
        BeamParam(name="categorical_optimizer", type=str, default="auto",
                  help="Optimizer for pure categorical problems. Choices: [auto, grid, random]. "
                       "Auto uses grid for small spaces (<1000 combinations) and random for larger spaces."),
        BeamParam(name="initialization_method", type=str, default="sobol",
                  help="Method for initial sampling when replay buffer is below training minimum. "
                       "Choices: [uniform, sobol, halton, random]."),
        BeamParam(name="constraint_method", type=str, default="penalty",
                  help="Method for handling output constraints. Choices: [penalty, feasibility]. "
                       "Penalty applies penalties for violations, feasibility uses constraint-aware acquisition."),
        BeamParam(name="penalty_weight", type=float, default=10.0,
                  help="Weight for constraint violation penalties when using penalty method."),
        BeamParam(name="constraint_tolerance", type=float, default=1e-3,
                  help="Tolerance for constraint violations when using feasibility method."),
    ]

class BayesianHPOServiceConfig(BayesianConfig):
    parameters = [
        BeamParam(name="embedding_model", type=str, default="jinaai/jina-embeddings-v3",
                  help="Embedding model to encode text content for Bayesian optimization."),
        BeamParam(name="truncate_dim", type=int, default=32,
                  help="Dimension to truncate the embeddings to for Bayesian optimization."),
        BeamParam(name="dataset", type=str, default=None,
                  help="Database URI for logging experiment statistics. Example: 'ibis-sqlite:///experiments.db/hpo_logs' "
                       "or 'ibis-bigquery:///project/dataset/table'. If None, database logging is disabled."),
        BeamParam(name="experiment_name", type=str, default="bayesian_optimization",
                  help="Name for the experiment (used as a column in database logging)."),
        BeamParam(name="log_suggestions", type=bool, default=True,
                  help="Whether to log parameter suggestions to the database."),
        BeamParam(name="log_results", type=bool, default=True,
                  help="Whether to log experiment results to the database."),
    ]