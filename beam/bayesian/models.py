import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

class GPClassificationModel(SingleTaskGP):
    """
    Single‐task GP classifier with a Bernoulli likelihood.
    """

    def __init__(self, train_X: torch.Tensor, train_Y: torch.Tensor):
        """
        Args:
            train_X: (n × d) Tensor of inputs.
            train_Y: (n × 1) Tensor of {0,1} labels.
        """
        # Initialize the GP “prior” (mean + kernel) and register train data
        super().__init__(train_X=train_X, train_Y=train_Y)
        # Attach a Bernoulli likelihood (uses the logistic link by default)
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    def fit(self, lr: float = 0.1, max_iter: int = 100):
        """
        Fit hyperparameters by maximizing the exact marginal log likelihood.
        """
        self.train()
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        # This will run L-BFGS-B under the hood with sensible defaults
        fit_gpytorch_mll(mll, options={"lr": lr, "maxiter": max_iter})

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns P(y=1 | X), shape (m × 1).
        """
        self.eval()
        with torch.no_grad():
            post = self.posterior(X)           # latent Normal posterior
            dist = self.likelihood(post)      # Bernoulli posterior
            return dist.probs                  # probabilities in [0,1]

    def predict(self, X: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Returns hard labels {0,1} by thresholding the probabilities.
        """
        return (self.predict_proba(X) >= threshold).float()
