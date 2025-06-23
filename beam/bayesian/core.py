from typing import Optional
import inspect

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel, MultiTaskGPyTorchModel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import Likelihood
from collections import namedtuple
from pydantic import BaseModel

from ..processor import Processor

from .hp_scheme import BaseParameters
from .config import BayesianConfig
from ..type import check_type, Types
from ..utils import as_tensor
from ..dataset import LazyReplayBuffer
from ..logging import beam_logger as logger


class Solution(BaseModel):
    x_num: Optional[torch.Tensor] = None
    x_cat: Optional[torch.Tensor] = None
    y: Optional[torch.Tensor] = None
    c_num: Optional[torch.Tensor] = None
    c_cat: Optional[torch.Tensor] = None


class Status(BaseModel):
    gp: Optional[torch.nn.Module] = None
    message: str = ""
    solution: Optional[Solution] = None
    acq_val: Optional[torch.Tensor] = None
    candidates: Optional[list[BaseParameters]] = None
    debug: Optional[dict] = None


class BayesianBeam(Processor):

    def __init__(self, x_scheme, *args, c_scheme=None, bounds=None, **kwargs):
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
        self._has_categorical = None
        self._x_bounds = None
        self._optimizer_acqf = None
        self._x_cat_cartesian_product_list = None
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

    def get_acquisition_function(self, model, **kwargs):
        """
        Get the acquisition function for Bayesian optimization.
        :param model: The Gaussian Process model.
        :param kwargs: Additional keyword arguments for the acquisition function.
        :return: The acquisition function.
        """
        acq_func = self.hparams.get('acquisition_function', 'EI')
        acquisition_kwargs = self.hparams.get('acquisition_kwargs', {})
        kwargs = {**acquisition_kwargs, **kwargs}
        if acq_func == 'EI':
            from botorch.acquisition import ExpectedImprovement
            return ExpectedImprovement(model=model, **kwargs)
        elif acq_func == 'PI':
            from botorch.acquisition import ProbabilityOfImprovement
            return ProbabilityOfImprovement(model=model, **kwargs)
        elif acq_func == 'UCB':
            from botorch.acquisition import UpperConfidenceBound
            return UpperConfidenceBound(model=model, **kwargs)
        elif acq_func == 'qExpectedImprovement':
            from botorch.acquisition import qExpectedImprovement
            return qExpectedImprovement(model=model, best_f=self.best_f, **kwargs)
        elif acq_func == 'PosteriorMean':
            from botorch.acquisition import PosteriorMean
            return PosteriorMean(model=model, **kwargs)
        else:
            raise ValueError(f"Unsupported acquisition function: {acq_func}. Supported functions are: "
                             "'EI', 'PI', 'UCB'.")

    @property
    def x_cat_cartesian_product_list(self) -> list[dict[int, float]]:
        """
        Get the Cartesian product of categorical features.
        :return: List of dictionaries representing the Cartesian product of categorical features.
        """

        if self._x_cat_cartesian_product_list is None:
            if not self.has_categorical():
                return []

            from itertools import product
            cat_features = self.x_scheme.cat_fields_to_index_map
            cartesian_product = product(*[self.x_scheme.get_feature_values(k) for k in cat_features])
            cartesian_prod = [dict(zip(cat_features.values(), values)) for values in cartesian_product]

            self._x_cat_cartesian_product_list = cartesian_prod

        return self._x_cat_cartesian_product_list

    def optimize(self, acq, q=1, **kwargs):

        num_restarts = self.hparams.get('num_restarts', 5)
        num_restarts = kwargs.pop('num_restarts', num_restarts)

        sequential = self.hparams.get('sequential_opt', True)
        sequential = kwargs.pop('sequential_opt', sequential)


        if self.has_categorical():

            if self.len_x_cat >= self.hparams.get('n_categorical_features_threshold', 5):
                from botorch.optim import optimize_acqf_mixed_alternating
                optimizer = optimize_acqf_mixed_alternating

                discrete_dims = list(range(self.len_x_num, self.len_x_num + self.len_x_cat))
                kwargs['discrete_dims'] = discrete_dims
            else:
                from botorch.optim import optimize_acqf_mixed
                optimizer = optimize_acqf_mixed
                kwargs['fixed_features_list'] = self.x_cat_cartesian_product_list
        else:
            from botorch.optim import optimize_acqf
            optimizer = optimize_acqf
            kwargs['sequential'] = sequential

            self._optimizer_acqf = optimizer, kwargs

        best_x, acq_val = optimizer(acq, self.x_bounds, q=q, num_restarts=num_restarts, **kwargs)

        return best_x, acq_val

    def to_tensor(self, x: Optional[list[dict]] = None, y: Optional[list] = None, c: Optional[list[dict]] = None) -> Solution:
        """
        Convert input features and context features to tensors.
        :param x: Input features.
        :param y: Target values (optional).
        :param c: Context features (optional).
        :return: Tuple of tensors (x_tensor, c_tensor).

        """
        if not isinstance(x, list) or not all(isinstance(item, dict) for item in x):
            raise TypeError("Input features `x` must be a list of dictionaries.")
        if c is not None and (not isinstance(c, list) or not all(isinstance(item, dict) for item in c)):
            raise TypeError("Context features `c` must be a list of dictionaries.")

        x_num, x_cat = self.x_scheme.encode_batch(x) if x is not None else (None, None)
        c_num, c_cat = self.c_scheme.encode_batch(c) if c is not None else (None, None)

        return Solution(x_num=x_num, x_cat=x_cat, y=as_tensor(y) if y is not None else None,
                        c_num=c_num, c_cat=c_cat)

    @property
    def len_x_num(self) -> int:
        """
        Get the number of numeric features in the input scheme.
        :return: Number of numeric features.
        """
        return self.x_scheme.len_x_num

    @property
    def len_x_cat(self) -> int:
        """
        Get the number of categorical features in the input scheme.
        :return: Number of categorical features.
        """
        return self.x_scheme.len_x_cat

    @property
    def len_c_num(self) -> int:
        """
        Get the number of numeric context features.
        :return: Number of numeric context features.
        """
        return self.c_scheme.len_x_num if self.c_scheme else 0

    @property
    def len_c_cat(self) -> int:
        """
        Get the number of categorical context features.
        :return: Number of categorical context features.
        """
        return self.c_scheme.len_x_cat if self.c_scheme else 0

    def has_categorical(self, s=None):
        if s is not None:
            self._has_categorical = len(s.x_cat) or (s.c_cat is not None and len(s.c_cat))
        return self._has_categorical

    def train(self, x: list[dict], y: list, c: Optional[list[dict]] = None, debug=False, **kwargs):
        """
        Initialize the Bayesian model with the provided data.
        :param x: Input features.
        :param y: Target values (optional).
        :param kwargs: Additional keyword arguments for initialization.
        :param c: Context features (optional).
        """

        s = self.to_tensor(x, y, c)
        model = self.get_gp_model(has_categorical=self.has_categorical(s))
        self.rb.store_batch(x_num=s.x_num, x_cat=s.x_cat, y=s.y, c_num=s.c_num, c_cat=s.c_cat)

        # get all the replay buffer data
        d = self.rb[:]

        x_num = d['x_num']
        x_cat = d['x_cat']
        y = d['y']
        x = torch.cat([x_num, x_cat], dim=-1)

        if c is not None:
            x = torch.cat([x, d['c_cat'], d['c_num']], dim=-1)
            cat_features = list(range(self.len_x_num, self.len_x_num + self.len_x_cat + self.len_c_cat))
        else:
            cat_features = list(range(self.len_x_num, self.len_x_num + self.len_x_cat))

        if len(cat_features):
            kwargs['categorical_features'] = cat_features

        ll = self.get_likelihood()
        self.gp = model(train_X=x, train_Y=y, likelihood=ll, **kwargs)
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)

        message = f"Model trained successfully with {len(x)} samples."
        logger.info(message)

        if debug:
            metadata = {
                'x_num': x_num,
                'x_cat': x_cat,
                'y': y,
                'c_num': s.c_num,
                'c_cat': s.c_cat,
                'model': self.gp.__class__.__name__,
                'likelihood': ll.__class__.__name__,
                'num_features': self.total_n_features
            }
        else:
            metadata = {}

        return Status(gp=self.gp, message=message, debug=metadata)

    @property
    def best_f(self):
        """
        Get the best observed value.
        :return: The best observed value.
        """
        y = self.rb[:]['y']
        if y is not None and len(y) > 0:
            return torch.max(y).item()
        return None

    @property
    def total_n_features(self) -> int:
        """
        Get the total number of features (numeric + categorical).
        :return: Total number of features.
        """
        return self.len_x_num + self.len_x_cat + self.len_c_num + self.len_c_cat

    @property
    def x_bounds(self) -> dict:
        if self._x_bounds is None:
            bounds = self.x_scheme.get_bounds()
            indexed_bounds = {}
            d_num = self.x_scheme.num_fields_to_index_map
            d_cat = self.x_scheme.cat_fields_to_index_map
            for k, b in bounds.items():
                if k in d_num:
                    indexed_bounds[d_num[k]] = b
                elif k in d_cat:
                    indexed_bounds[d_cat[k] + self.len_x_num] = b
                else:
                    raise ValueError(f"Feature {k} not found in input scheme.")
            self._x_bounds = indexed_bounds
        return self._x_bounds

    def sample(self, c=None, n_samples=1, debug=False, **kwargs) -> Status:
        """
        Sample from the Bayesian model.
        :param c: Context features (optional).
        :param n_samples: Number of samples to generate.
        :param kwargs: Additional keyword arguments for sampling.
        :return: Generated samples.
        """

        if self.gp is None:
            message = "Model is not trained yet. Please train the model before sampling."
            logger.error(message)
            return Status(gp=None, message=message)

        acq = self.get_acquisition_function(self.gp, **kwargs)

        if c is not None:
            s = self.to_tensor(c=c)
            c = torch.cat([s.c_cat, s.c_num], dim=-1)
            columns = list(range(self.len_x_num + self.len_x_cat, self.total_n_features))
            acq = FixedFeatureAcquisitionFunction(acq, d=self.total_n_features,
                                                      columns=columns, values=c.squeeze(0))
            fixed_features = {k: v for k, v in self.x_scheme.encode(c).items()
                              if k in self.x_scheme.cat_fields_to_index_map}

        best_x, acq_val = self.optimize(acq, q=n_samples, **kwargs)

        decoded = [self.x_scheme.decode(xi) for xi in best_x]

        message = f"Generated {n_samples} samples with acquisition value: {acq_val}"
        logger.info(message)

        if debug:
            metadata = {
                'best_x': best_x,
                'acq_val': acq_val,
                'x_bounds': self.x_bounds,
                'n_samples': n_samples,
            }
        else:
            metadata = {}

        return Status(candidates=decoded, debug=metadata, message=message)

