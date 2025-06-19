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

]
class BayesianConfig(BeamConfig):
    pass

