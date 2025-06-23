from ..config import BeamConfig, BeamParam



params = [
    BeamParam(name="acquisition_function", type=str, default="EI",
              help="Acquisition function to use for Bayesian optimization. Choices: [EI, PI, UCB, GP_EI, "
                   "GP_PI, GP_UCB]"),
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
    BeamParam(name="n_categorical_features_threshold", type=int, default=5,
              help="Threshold for the number of categorical features to use a different acquisition function "
                   "(optimize_acqf_mixed_alternating instead of optimize_acqf_mixed)."),
]
class BayesianConfig(BeamConfig):
    pass

