from dataclasses import dataclass
from typing import Optional
import inspect

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel, MultiTaskGPyTorchModel, GPyTorchModel
from gpytorch.kernels import Kernel
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

@dataclass
class Solution:
    x_num: Optional[torch.Tensor] = None
    x_cat: Optional[torch.Tensor] = None
    y: Optional[torch.Tensor] = None
    c_num: Optional[torch.Tensor] = None
    c_cat: Optional[torch.Tensor] = None


@dataclass
class Status:
    gp: Optional[torch.nn.Module] = None
    message: str = ""
    solution: Optional[Solution] = None
    acq_val: Optional[torch.Tensor] = None
    candidates: Optional[list[BaseParameters]] = None
    debug: Optional[dict] = None
    config: Optional[dict] = None


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
        self.acquisitions = None
        self.prior = None
        self.belief = None
        self.likelihood = None
        self._has_categorical = None
        self._x_bounds = None
        self._optimizer_acqf = None
        self._x_cat_cartesian_product_list = None
        self.new_points = 0
        self.rb = LazyReplayBuffer(size=self.hparams.get('buffer_size', 1000))

    def reset_acquisitions(self):
        self.acquisitions = {'single': None, 'batch': None}

    def build_continuous_kernel(self, **kwargs) -> Optional[Kernel]:

        kind = self.hparams.get('continuous_kernel', None)
        kernel_kwargs = self.hparams.get('continuous_kernel_kwargs', {})
        if kind is None:
            return None
        elif kind == 'RBFKernel':
            from gpytorch.kernels import RBFKernel, ScaleKernel
            return RBFKernel(**kernel_kwargs)
        elif kind == 'MaternKernel':
            from gpytorch.kernels import MaternKernel, ScaleKernel
            nu = kernel_kwargs.pop('nu', 1.5)
            return ScaleKernel(MaternKernel(nu=nu, **kernel_kwargs))
        elif kind == 'RationalQuadraticKernel':
            from gpytorch.kernels import RationalQuadraticKernel, ScaleKernel
            return ScaleKernel(RationalQuadraticKernel(**kernel_kwargs))
        else:
            logger.error(f"Unsupported continuous kernel: {kind}.")
            return None

    def build_gp_model(self, x, y, cat_features: list, **kwargs) -> GPyTorchModel:
        """
        Get the Gaussian Process model.
        :return: The Gaussian Process model.
        """

        ll = self.get_likelihood()
        has_categorical = len(cat_features) > 0

        cont_kernel = self.build_continuous_kernel(**kwargs)
        gp_model = self.hparams.get('gp_model', 'SingleTaskGP')
        if gp_model == 'SingleTaskGP':
            if has_categorical:
                from botorch.models import MixedSingleTaskGP
                return MixedSingleTaskGP(x, y, likelihood=ll, cat_dims=cat_features, cont_kernel_factory=cont_kernel,
                                         **kwargs)
            else:
                from botorch.models import SingleTaskGP
                return SingleTaskGP(x, y, likelihood=ll, covar_module=cont_kernel, **kwargs)
        elif gp_model == 'MultiTaskGP':
            from botorch.models import MultiTaskGP
            return MultiTaskGP(x, y, likelihood=ll, **kwargs)
        elif gp_model == 'GPClassificationModel':
            from .models import GPClassificationModel
            return GPClassificationModel(x, y)
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

    def build_acquisition_function(self, model, q=1, **kwargs):
        """
        Get the acquisition function for Bayesian optimization.
        :param model: The Gaussian Process model.
        :param q: Number of points to sample in batch (default is 1).
        :param kwargs: Additional keyword arguments for the acquisition function.
        :return: The acquisition function.
        """
        acq_func = self.hparams.get('acquisition_function', 'LogExpectedImprovement')
        acquisition_kwargs = self.hparams.get('acquisition_kwargs', {})
        kwargs = {**acquisition_kwargs, **kwargs}
        use_q = self.hparams.batch_size > 1 or q > 1
        if acq_func == 'LogExpectedImprovement':
            if use_q:
                from botorch.acquisition import qLogExpectedImprovement
                return qLogExpectedImprovement(model, best_f=self.best_f, **kwargs)
            else:
                from botorch.acquisition import LogExpectedImprovement
                return LogExpectedImprovement(model, best_f=self.best_f, **kwargs)
        elif acq_func == 'ExpectedImprovement':
            if use_q:
                from botorch.acquisition import qExpectedImprovement
                return qExpectedImprovement(model, best_f=self.best_f, **kwargs)
            else:
                from botorch.acquisition import ExpectedImprovement
                return ExpectedImprovement(model, best_f=self.best_f, **kwargs)
        elif acq_func == 'ProbabilityOfImprovement':
            if use_q:
                from botorch.acquisition import qProbabilityOfImprovement
                return qProbabilityOfImprovement(model, best_f=self.best_f, **kwargs)
            else:
                from botorch.acquisition import ProbabilityOfImprovement
                return ProbabilityOfImprovement(model, **kwargs)
        elif acq_func == 'UpperConfidenceBound':
            if use_q:
                from botorch.acquisition import qUpperConfidenceBound
                return qUpperConfidenceBound(model, **kwargs)
            else:
                from botorch.acquisition import UpperConfidenceBound
                return UpperConfidenceBound(model, **kwargs)
        elif acq_func == 'PosteriorMean':
            if use_q:
                from botorch.acquisition.analytic import ScalarizedPosteriorMean
                return ScalarizedPosteriorMean(model, **kwargs)
            else:
                from botorch.acquisition import PosteriorMean
                return PosteriorMean(model, **kwargs)
        else:
            raise ValueError(f"Unsupported acquisition function: {acq_func}. Supported functions are: "
                             "'ExpectedImprovement', 'ProbabilityOfImprovement', "
                             "'UpperConfidenceBound', 'PosteriorMean'.")

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
            cat_features = self.x_scheme.cat_fields_to_index_map  # {name: idx_in_cat}
            print(cat_features)
            cartesian_values = product(*[self.x_scheme.get_feature_values(k, encoded=True)
                                         for k in cat_features.keys()])

            cartesian_prod = [
                {self.len_x_num + idx: float(val)  # correct global index ✅
                 for idx, val in zip(cat_features.values(), combo)}
                for combo in cartesian_values
            ]
            self._x_cat_cartesian_product_list = cartesian_prod

        return self._x_cat_cartesian_product_list

    @property
    def discrete_choices(self) -> list[torch.Tensor]:
        discrete_choices = [
            torch.tensor(self.x_scheme.get_feature_values(name, encoded=True))  # choices for each cat dim
            for name in self.x_scheme.cat_fields_to_index_map
        ]
        return discrete_choices

    def optimize(self, acq, q=1, **kwargs):

        num_restarts = self.hparams.get('num_restarts', 5)
        num_restarts = kwargs.pop('num_restarts', num_restarts)

        sequential = self.hparams.get('sequential_opt', True)
        sequential = kwargs.pop('sequential_opt', sequential)

        raw_samples = self.hparams.get('raw_samples', 1000)
        raw_samples = kwargs.pop('raw_samples', raw_samples)

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

        best_x, acq_val = optimizer(acq, self.x_bounds, q=q, num_restarts=num_restarts, raw_samples=raw_samples,
                                    **kwargs)

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
        if y is not None:
            y = as_tensor(y)
            if len(y.shape) == 1:
                y = y.unsqueeze(-1)
        else:
            y = None

        return Solution(x_num=x_num, x_cat=x_cat, y=y, c_num=c_num, c_cat=c_cat)

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

    def reset(self):
        """
        Reset the Bayesian model and the replay buffer.
        """
        self.gp = None
        self.rb.reset()
        self._has_categorical = None
        self._x_bounds = None
        self._x_cat_cartesian_product_list = None
        message = "Model and replay buffer reset successfully."
        logger.info(message)
        return Status(gp=None, message=message)

    def reshape_batch(self, v):
        if self.hparams.batch_size > 1:
            b = self.hparams.batch_size
            # Reshape x_num and x_cat to have batch size as the second dimension
            if v is not None:
                # truncate x_num to the nearest multiple of b
                v = v[(len(v) - len(v) % b):]
                v = v.view(-1, b, v.shape[-1])
        return v

    def get_replay_buffer(self, d=None):

        # get all the replay buffer data
        if d is None:
            d = self.rb[:]

        x_num, x_cat = self.reshape_batch(d['x_num']), self.reshape_batch(d['x_cat'])
        y = self.reshape_batch(d['y'])
        c_cat, c_num = self.reshape_batch(d['c_cat']), self.reshape_batch(d['c_num'])

        if self.hparams.batch_size > 1:
            b = self.hparams.batch_size
            # Reshape x_num and x_cat to have batch size as the second dimension
            if x_num is not None:
                # truncate x_num to the nearest multiple of b
                x_num = x_num[(len(x_num) - len(x_num) % b):]
                x_num = x_num.view(-1, b, self.len_x_num).mean(dim=1)
            if x_cat is not None:
                # truncate x_cat to the nearest multiple of b
                x_cat = x_cat[(len(x_cat) - len(x_cat) % b):]
                x_cat = x_cat.view(-1, b, self.len_x_cat).mean(dim=1)

        x = torch.cat([x_num, x_cat], dim=-1)

        if c_num is not None:
            x = torch.cat([x, c_cat, c_num], dim=-1)
            cat_features = list(range(self.len_x_num, self.len_x_num + self.len_x_cat + self.len_c_cat))
        else:
            cat_features = list(range(self.len_x_num, self.len_x_num + self.len_x_cat))

        return x, y, cat_features

    def train(self, x: list[dict], y: list, c: Optional[list[dict]] = None, debug=False, **kwargs):
        """
        Initialize the Bayesian model with the provided data.
        :param x: Input features.
        :param y: Target values (optional).
        :param kwargs: Additional keyword arguments for initialization.
        :param c: Context features (optional).
        """

        s = self.to_tensor(x, y, c)
        self.rb.store_batch(x_num=s.x_num, x_cat=s.x_cat, y=s.y, c_num=s.c_num, c_cat=s.c_cat)

        self.new_points += len(y)

        if len(self.rb) < self.hparams.start_fitting_after_n_points:
            message = f"Not enough points to train the model. New points: {self.new_points}, " \
                      f"Total points: {len(self.rb)}, Start fitting after N points: {self.hparams.start_fitting_after_n_points}."
            logger.info(message)
            return Status(gp=None, message=message)

        if self.new_points < self.hparams.fit_every_n_points:

            incremental_fit = self.hparams.incremental_fit

            if incremental_fit == 'none':
                message = f"Skipping model training. New points: {self.new_points}, " \
                          f"Total points: {len(self.rb)}, Fit every N points: {self.hparams.fit_every_n_points}."
                logger.info(message)
                return Status(gp=self.gp, message=message)

            elif incremental_fit == 'fantasy':
                x_star, y_star, cat_features = self.get_replay_buffer({'x_num': s.x_num, 'x_cat': s.x_cat,
                                                                       'y': s.y, 'c_num': s.c_num, 'c_cat': s.c_cat})
                self.gp.condition_on_observations(X=x_star, Y=y_star)

                message = f"Model updated with {len(x_star)} fantasy points. New points: {self.new_points}, " \
                          f"Total points: {len(self.rb)}, Fit every N points: {self.hparams.fit_every_n_points}."
                logger.info(message)
                return Status(gp=self.gp, message=message)

            elif incremental_fit == 'full':
                x, y, cat_features = self.get_replay_buffer()
                self.gp.set_train_data(inputs=x, targets=y, strict=False)
                message = f"Model set_train_data with {len(x)} samples. New points: {self.new_points}, " \
                          f"Total points: {len(self.rb)}, Fit every N points: {self.hparams.fit_every_n_points}."
                logger.info(message)
                return Status(gp=self.gp, message=message)

            else:
                message = "Invalid incremental_fit method. Supported methods are: 'fantasy', 'full', 'none'."
                logger.error(message)
                return Status(gp=None, message="Invalid incremental_fit method.")

        # if we are here, we are training the model from scratch
        self.new_points = 0

        # set this boolean if has_categorical is not set yet
        self.has_categorical(s)

        x, y, cat_features = self.get_replay_buffer()
        self.reset_acquisitions()

        self.gp = self.build_gp_model(x=x, y=y, cat_features=cat_features, **kwargs)

        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)

        message = f"Model trained successfully with {len(x)} samples."
        logger.info(message)

        if debug:
            metadata = {
                'x_num': s.x_num,
                'x_cat': s.x_cat,
                'y': s.y,
                'c_num': s.c_num,
                'c_cat': s.c_cat,
                'model': self.gp.__class__.__name__,
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

            logger.debug(f"Indexed bounds: {indexed_bounds}")

            # --- tensor scaffolding ------------------------------------------------
            d = self.total_n_features
            # if gp not trained yet, use CPU / float32; will be moved later
            try:
                train_X = self.gp.train_inputs[0]
                dtype, device = train_X.dtype, train_X.device
            except AttributeError:
                dtype, device = torch.float32, torch.device("cpu")

            gub = self.hparams.get('global_upper_bound', 1e6)
            lower = torch.full((d,), -gub, dtype=dtype, device=device)
            upper = torch.full((d,), gub, dtype=dtype, device=device)

            # --- fill in user-supplied bounds -------------------------------------
            for j, (lo, hi) in indexed_bounds.items():
                lower[j] = lo
                upper[j] = hi

            self._x_bounds = torch.stack([lower, upper])

            logger.debug(f"X bounds: {self._x_bounds}")

        return self._x_bounds

    def sample(self, c=None, n_samples=None, debug=False, **kwargs) -> Status:
        """
        Sample from the Bayesian model.
        :param c: Context features (optional).
        :param n_samples: Number of samples to generate.
        :param kwargs: Additional keyword arguments for sampling.
        :return: Generated samples.
        """
        if n_samples is None:
            n_samples = self.hparams.batch_size

        if self.gp is None:
            message = "Model is not trained yet. Please train the model before sampling."
            logger.error(message)
            return Status(gp=None, message=message)

        acq_type = 'single' if n_samples == 1 else 'batch'
        if self.acquisitions.get(acq_type) is None:
            acq = self.build_acquisition_function(self.gp, q=n_samples, **kwargs)

            if c is not None:
                s = self.to_tensor(c=c)
                c = torch.cat([s.c_cat, s.c_num], dim=-1)
                columns = list(range(self.len_x_num + self.len_x_cat, self.total_n_features))
                acq = FixedFeatureAcquisitionFunction(acq, d=self.total_n_features,
                                                          columns=columns, values=c.squeeze(0))
            self.acquisitions[acq_type] = acq
        else:
            logger.debug(f"Using cached acquisition function for {acq_type} sampling.")
            acq = self.acquisitions[acq_type]

        best_x, acq_val = self.optimize(acq, q=n_samples, **kwargs)

        logger.debug(f"Best x: {best_x}, acq_val: {acq_val}")

        best_x_num = best_x[:, :self.len_x_num]
        best_x_cat = best_x[:, self.len_x_num:self.len_x_num + self.len_x_cat]

        decoded = self.x_scheme.decode_batch(best_x_num, best_x_cat)

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

