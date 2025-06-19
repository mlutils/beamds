from typing import Optional

import torch
from botorch import fit_gpytorch_mll
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel, MultiTaskGPyTorchModel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import Likelihood

from ..processor import Processor

from .hp_scheme import BaseParameters
from .config import BayesianConfig
from ..type import check_type, Types
from ..utils import as_tensor
from ..dataset import LazyReplayBuffer


class BayesianBeam(Processor):

    def __init__(self, x_scheme, *args, c_scheme=None, **kwargs):
        super().__init__(*args, _config_scheme=BayesianConfig, **kwargs)

        if check_type(x_scheme).minor == Types.dict:
            x_scheme = BaseParameters.from_json_schema(x_scheme)
        self.x_scheme = x_scheme
        if check_type(c_scheme).minor == Types.dict:
            c_scheme = BaseParameters.from_json_schema(c_scheme)
        self.c_scheme = c_scheme
        self.gp = None
        self.acquisition = None
        self.prior = None
        self.belief = None
        self.likelihood = None
        self.rb = LazyReplayBuffer()

    def get_gp_model(self, has_categorical: bool = False) -> type[BatchedMultiOutputGPyTorchModel] | type[MultiTaskGPyTorchModel]:
        """
        Get the Gaussian Process model.
        :return: The Gaussian Process model.
        """
        gp_model = self.hparams.get('gp_model', 'SingleTaskGP')
        if gp_model == 'SingleTaskGP':
            if has_categorical:
                from botorch.models import MixedSingleTaskGP
                return MixedSingleTaskGP
            else:
                from botorch.models import SingleTaskGP
                return SingleTaskGP
        elif gp_model == 'MultiTaskGP':
            from botorch.models import MultiTaskGP
            return MultiTaskGP
        elif gp_model == 'GPClassificationModel':
            from .models import GPClassificationModel
            return GPClassificationModel
        else:
            raise ValueError(f"Unsupported Gaussian Process model: {gp_model}. Supported models are: "
                             "'SingleTaskGP', 'MultiTaskGP', 'MultiOutputGP'.")

    def get_likelihood(self) -> Likelihood:
        """
        Get the likelihood for the Gaussian Process model.
        :return: The likelihood class.
        """
        likelihood = self.hparams.get('likelihood', 'GaussianLikelihood')
        if likelihood == 'GaussianLikelihood':
            from gpytorch.likelihoods import GaussianLikelihood
            ll = GaussianLikelihood
        elif likelihood == 'BernoulliLikelihood':
            from gpytorch.likelihoods import BernoulliLikelihood
            ll = BernoulliLikelihood
        elif likelihood == 'LaplaceLikelihood':
            from gpytorch.likelihoods import LaplaceLikelihood
            ll = LaplaceLikelihood
        else:
            raise ValueError(f"Unsupported likelihood: {likelihood}. Supported likelihoods are: "
                             "'GaussianLikelihood', 'BernoulliLikelihood', 'PoissonLikelihood'.")

        return ll(**self.hparams.get('likelihood_kwargs', {}))

    def to_tensor(self, x: list[dict], y: list, c: Optional[list[dict]] = None) -> tuple:
        """
        Convert input features and context features to tensors.
        :param x: Input features.
        :param c: Context features (optional).
        :return: Tuple of tensors (x_tensor, c_tensor).
        """
        if not isinstance(x, list) or not all(isinstance(item, dict) for item in x):
            raise TypeError("Input features `x` must be a list of dictionaries.")
        if c is not None and (not isinstance(c, list) or not all(isinstance(item, dict) for item in c)):
            raise TypeError("Context features `c` must be a list of dictionaries.")

        x_num, x_cat = self.x_scheme.encode(x)
        c_num, c_cat = self.c_scheme.encode(c) if c is not None else None, None
        y = as_tensor(y)
        return x_num, x_cat, y, c_num, c_cat

    def train(self, x: list[dict], y: list, c: Optional[list[dict]] = None, **kwargs):
        """
        Initialize the Bayesian model with the provided data.
        :param x: Input features.
        :param y: Target values (optional).
        :param kwargs: Additional keyword arguments for initialization.
        :param c: Context features (optional).
        """

        x_num, x_cat, y, c_num, c_cat = self.to_tensor(x, y, c)
        model = self.get_gp_model(has_categorical=len(x_cat) or (c is not None and len(c_cat)))
        self.rb.store_batch(x_num=x_num, x_cat=x_cat, y=y, c_num=c_num, c_cat=c_cat)

        # get all the replay buffer data
        d = self.rb[:]

        x_num = d['x_num']
        x_cat = d['x_cat']
        y = d['y']
        if c_num is not None:
            x_num = torch.cat([x_num, d['c_num']], dim=-1)
        if c_cat is not None:
            x_cat = torch.cat([x_cat, d['c_cat']], dim=-1)

        if len(x_cat):
            x = torch.cat([x_num, x_cat], dim=-1)
            kwargs['categorical_features'] = torch.arange(x_num.shape[1], x.shape[1], dtype=torch.long)
        else:
            x = x_num

        ll = self.get_likelihood()
        self.gp = model(train_X=x, train_Y=y, likelihood=ll, **kwargs)
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)

        status = {'gp': str(self.gp)}

        return status


    def sample(self, c=None, n_samples=1, **kwargs):
        """
        Sample from the Bayesian model.
        :param c: Context features (optional).
        :param n_samples: Number of samples to generate.
        :param kwargs: Additional keyword arguments for sampling.
        :return: Generated samples.
        """
        raise NotImplementedError("Sample method must be implemented in subclasses.")
