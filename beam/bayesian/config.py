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
                  help="Device to use for Bayesian optimization. Choices: [cpu, cuda]"),
        BeamParam(name="dtype", type=str, default="float32",
                  help="Data type to use for Bayesian optimization. Choices: [float32, float64]"),
        BeamParam(name="likelihood", type=str, default="GaussianLikelihood",
                  help="Likelihood to use for Bayesian optimization. Choices: [GaussianLikelihood, "
                       "BernoulliLikelihood, PoissonLikelihood]"),
        BeamParam(name="likelihood_kwargs", type=dict, default={'noise': 0.1},
                  help="Additional keyword arguments for the likelihood."),
        BeamParam(name="acquisition_kwargs", type=dict, default={},
                  help="Additional keyword arguments for the acquisition function."),
        BeamParam(name="num_restarts", type=int, default=5,
                  help="Number of restarts for the optimization process."),
        BeamParam(name="sequential_opt", type=bool, default=True,
                  help="Whether to perform sequential optimization or not."),
        BeamParam(name="raw_samples", type=int, default=1000,
                  help="Number of raw samples to generate for Bayesian optimization."),
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
    ]

class BayesianHPOServiceConfig(BayesianConfig):
    parameters = [
        BeamParam(name="embedding_model", type=str, default="jinaai/jina-embeddings-v3",
                  help="Embedding model to encode text content for Bayesian optimization."),
        BeamParam(name="truncate_dim", type=int, default=32,
                  help="Dimension to truncate the embeddings to for Bayesian optimization."),
    ]