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
from ..utils import as_tensor, beam_device
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
        self.constraint_gp = None  # GP model for constraints
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

    @classmethod
    @property
    def special_state_attributes(cls) -> set[str]:
        """
        Define which attributes should be saved as part of the state.
        This integrates with the Processor base class state management.
        """
        base_attrs = super().special_state_attributes
        bayesian_attrs = {
            'gp_state_dict',      # GP model parameters
            'replay_buffer_data', # Training data
            'x_scheme_dict',      # Parameter schema
            'c_scheme_dict',      # Context schema (if exists)
            'optimization_state', # Other optimization state
            'hparams'            # Keep hparams from base class
        }
        return base_attrs.union(bayesian_attrs)

    @classmethod  
    @property
    def excluded_attributes(cls) -> set[str]:
        """
        Define attributes that should not be saved in state.
        """
        base_excluded = super().excluded_attributes
        bayesian_excluded = {
            'acquisitions',                    # Rebuilt on load
            '_x_bounds',                      # Rebuilt on load
            '_optimizer_acqf',                # Rebuilt on load  
            '_x_cat_cartesian_product_list',  # Rebuilt on load
        }
        return base_excluded.union(bayesian_excluded)

    def __getstate__(self):
        """
        Prepare state for serialization, integrating with Processor's save_state system.
        """
        if self.in_beam_pickle():
            # When saving state via Processor.save_state(), prepare special attributes
            self._prepare_state_for_saving()
        
        # Call parent's __getstate__ to handle the standard exclusions
        return super().__getstate__()

    def _prepare_state_for_saving(self):
        """
        Prepare Bayesian-specific state attributes for saving.
        """
        # Save GP model state if it exists
        if self.gp is not None:
            self.gp_state_dict = self.gp.state_dict()
        else:
            self.gp_state_dict = None
            
        # Save replay buffer data
        self.replay_buffer_data = list(self.rb)
        
        # Save schemas
        self.x_scheme_dict = self.x_scheme.model_dump()
        self.c_scheme_dict = self.c_scheme.model_dump() if self.c_scheme else None
        
        # Save other optimization state
        self.optimization_state = {
            'best_f': self.best_f,
            'new_points': self.new_points,
            'has_categorical': self._has_categorical,
            'device': str(self.device),
        }

    def __setstate__(self, state):
        """
        Restore state after deserialization.
        """
        # Call parent's __setstate__ first
        super().__setstate__(state)
        
        # Initialize attributes that aren't saved
        self._reset_transient_attributes()

    def _reset_transient_attributes(self):
        """
        Reset attributes that are rebuilt rather than saved.
        """
        self.acquisitions = None
        self._x_bounds = None  
        self._optimizer_acqf = None
        self._x_cat_cartesian_product_list = None

    def load_state(self, path=None, state=None, **kwargs):
        """
        Override load_state to handle Bayesian-specific restoration.
        """
        # Call parent's load_state first
        super().load_state(path=path, state=state, **kwargs)
        
        # Restore Bayesian-specific state if it exists
        self._restore_bayesian_state()

    def _restore_bayesian_state(self):
        """
        Restore Bayesian optimization state after loading.
        """
        try:
            # Restore replay buffer if data exists
            if hasattr(self, 'replay_buffer_data') and self.replay_buffer_data:
                # Clear existing buffer and restore data
                self.rb = LazyReplayBuffer(size=self.hparams.get('buffer_size', 1000))
                for item in self.replay_buffer_data:
                    self.rb.add(**item)
                logger.info(f"Restored {len(self.replay_buffer_data)} samples to replay buffer")
            
            # Restore schemas (they should already be restored, but ensure they're BaseParameters objects)
            if hasattr(self, 'x_scheme_dict') and self.x_scheme_dict:
                if not isinstance(self.x_scheme, BaseParameters):
                    self.x_scheme = BaseParameters.from_json_schema(self.x_scheme_dict)
                    
            if hasattr(self, 'c_scheme_dict') and self.c_scheme_dict:
                if not isinstance(self.c_scheme, BaseParameters):
                    self.c_scheme = BaseParameters.from_json_schema(self.c_scheme_dict)
                elif self.c_scheme_dict is None:
                    self.c_scheme = None
            
            # Restore optimization state
            if hasattr(self, 'optimization_state') and self.optimization_state:
                self.new_points = self.optimization_state.get('new_points', 0)
                self._has_categorical = self.optimization_state.get('has_categorical', None)
                
            # Restore GP model if state exists and we have training data
            if (hasattr(self, 'gp_state_dict') and self.gp_state_dict and 
                hasattr(self, 'replay_buffer_data') and self.replay_buffer_data):
                
                try:
                    # Rebuild GP model from training data
                    x, y, cat_features = self.get_replay_buffer()
                    self.gp = self.build_gp_model(x=x, y=y, cat_features=cat_features)
                    
                    # Load the saved state
                    self.gp.load_state_dict(self.gp_state_dict)
                    logger.info("GP model state restored successfully")
                    
                except Exception as e:
                    logger.warning(f"Failed to restore GP model: {e}. Will retrain on next use.")
                    self.gp = None
            
            # Reset acquisitions (they will be rebuilt when needed)
            self.reset_acquisitions()
            
            logger.info("Bayesian optimization state restored successfully")
            
        except Exception as e:
            logger.error(f"Error restoring Bayesian state: {e}")
            # Initialize with safe defaults if restoration fails
            self._reset_transient_attributes()

    def save_optimization_state(self, path: str) -> None:
        """
        Convenience method to save optimization state using Processor's save_state.
        Supports all storage backends (local, S3, GCS, etc.) via beam_path.
        """
        self.save_state(path)
        logger.info(f"Bayesian optimization state saved to {path}")

    def load_optimization_state(self, path: str) -> None:
        """
        Convenience method to load optimization state using Processor's load_state.
        Supports all storage backends (local, S3, GCS, etc.) via beam_path.
        """
        self.load_state(path)
        logger.info(f"Bayesian optimization state loaded from {path}")

    def reset_acquisitions(self):
        self.acquisitions = {'single': None, 'batch': None}

    @property
    def device(self):
        """Get the standardized device for Bayesian optimization."""
        return beam_device(self.hparams.get('device', 'cpu'))

    def _get_dtype(self, device=None):
        """Get appropriate dtype for the given device. MPS doesn't support float64."""
        if device is None:
            device = self.device
        return torch.float32 if device.type == 'mps' else torch.float64

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
        logger.info(f"Building GP model: x.shape={x.shape}, y.shape={y.shape}, n_cat_features={len(cat_features)}")
        
        is_multi_objective = y.shape[-1] > 1 if y is not None else False
        if is_multi_objective:
            logger.info(f"Multi-objective GP detected with {y.shape[-1]} objectives")
        else:
            logger.debug(f"Single-objective GP model")
            
        if cat_features:
            logger.debug(f"Categorical features at indices: {cat_features}")

        ll = self.get_likelihood()
        has_categorical = len(cat_features) > 0

        cont_kernel = self.build_continuous_kernel(**kwargs)
        gp_model = self.hparams.get('gp_model', 'SingleTaskGP')
        
        logger.debug(f"Using GP model: {gp_model}, has_categorical: {has_categorical}")
        if gp_model == 'SingleTaskGP':
            if has_categorical:
                from botorch.models import MixedSingleTaskGP
                logger.debug(f"Creating MixedSingleTaskGP with cat_dims={cat_features}")
                return MixedSingleTaskGP(x, y, likelihood=ll, cat_dims=cat_features, cont_kernel_factory=cont_kernel,
                                         **kwargs)
            else:
                from botorch.models import SingleTaskGP
                logger.debug(f"Creating SingleTaskGP with continuous kernel")
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
        elif likelihood == 'SoftmaxLikelihood':
            from gpytorch.likelihoods import SoftmaxLikelihood
            ll = SoftmaxLikelihood
        elif likelihood == 'DirichletClassificationLikelihood':
            from gpytorch.likelihoods import DirichletClassificationLikelihood
            ll = DirichletClassificationLikelihood
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
        
        logger.info(f"Building acquisition function: {acq_func} (q={q}, use_q={use_q})")
        if acquisition_kwargs:
            logger.debug(f"Acquisition kwargs: {acquisition_kwargs}")
        if acq_func == 'LogExpectedImprovement':
            if use_q:
                from botorch.acquisition import qLogExpectedImprovement
                base_acq = qLogExpectedImprovement(model, best_f=self.best_f, **kwargs)
            else:
                from botorch.acquisition import LogExpectedImprovement
                base_acq = LogExpectedImprovement(model, best_f=self.best_f, **kwargs)
        elif acq_func == 'ExpectedImprovement':
            if use_q:
                from botorch.acquisition import qExpectedImprovement
                base_acq = qExpectedImprovement(model, best_f=self.best_f, **kwargs)
            else:
                from botorch.acquisition import ExpectedImprovement
                base_acq = ExpectedImprovement(model, best_f=self.best_f, **kwargs)
        elif acq_func == 'ProbabilityOfImprovement':
            if use_q:
                from botorch.acquisition import qProbabilityOfImprovement
                base_acq = qProbabilityOfImprovement(model, best_f=self.best_f, **kwargs)
            else:
                from botorch.acquisition import ProbabilityOfImprovement
                base_acq = ProbabilityOfImprovement(model, **kwargs)
        elif acq_func == 'UpperConfidenceBound':
            if use_q:
                from botorch.acquisition import qUpperConfidenceBound
                base_acq = qUpperConfidenceBound(model, **kwargs)
            else:
                from botorch.acquisition import UpperConfidenceBound
                base_acq = UpperConfidenceBound(model, **kwargs)
        elif acq_func == 'PosteriorMean':
            if use_q:
                from botorch.acquisition.analytic import ScalarizedPosteriorMean
                base_acq = ScalarizedPosteriorMean(model, **kwargs)
            else:
                from botorch.acquisition import PosteriorMean
                base_acq = PosteriorMean(model, **kwargs)
        
        # Multi-objective acquisition functions
        elif acq_func == 'qEHVI':
            from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
            from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
            
            logger.debug(f"Building qEHVI acquisition function with kwargs: {kwargs}")
            
            # qEHVI requires a partitioning for hypervolume computation
            if 'partitioning' not in kwargs:
                # Get reference point from kwargs or use a default
                ref_point = kwargs.get('ref_point')
                if ref_point is None:
                    # Use a conservative default reference point
                    num_objectives = getattr(self, '_objectives_info', {})
                    if num_objectives:
                        ref_point = []
                        for obj_name, direction in num_objectives.items():
                            if direction == 'maximize':
                                ref_point.append(0.0)  # Worst case for maximization
                            else:  # minimize
                                ref_point.append(1000.0)  # Worst case for minimization
                    else:
                        ref_point = [0.0, 1000.0]  # Default 2-objective case

                    kwargs['ref_point'] = ref_point
                    logger.info(f"Using default reference point for qEHVI: {ref_point}")

                # Create partitioning using current observations if available
                try:
                    # Get current training data to initialize partitioning
                    if hasattr(self, 'rb') and len(self.rb) > 0:
                        training_data = self.rb[:]
                        if 'y' in training_data and training_data['y'] is not None:
                            Y_observed = training_data['y']

                            # Create partitioning from observed data
                            ref_point_tensor = torch.tensor(ref_point, dtype=Y_observed.dtype, device=Y_observed.device)
                            partitioning = FastNondominatedPartitioning(
                                ref_point=ref_point_tensor,
                                Y=Y_observed
                            )
                            kwargs['partitioning'] = partitioning
                            logger.debug(f"Created partitioning from {len(Y_observed)} observations")
                        else:
                            logger.warning("No training data available for partitioning, using ref_point only")
                    else:
                        logger.warning("No replay buffer data available for partitioning")

                except Exception as e:
                    logger.error(f"Failed to create qEHVI partitioning: {e}, using ref_point only")
                    logger.debug(f"Partitioning error details: {type(e).__name__}: {str(e)}")

            base_acq = qExpectedHypervolumeImprovement(model, **kwargs)

        elif acq_func in ['qLogEHVI', 'qLogExpectedHypervolumeImprovement']:
            from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement
            from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

            logger.debug(f"Building qLogEHVI acquisition function with kwargs: {kwargs}")

            # qLogEHVI requires a partitioning for hypervolume computation
            if 'partitioning' not in kwargs:
                # Handle reference point with robust ordering
                ref_point = kwargs.get('ref_point')

                # If ref_point is provided as a dict (named), convert to tensor order
                if isinstance(ref_point, dict):
                    logger.info(f"Using named reference point: {ref_point}")
                    # Convert dict to list using the actual tensor objective order
                    objectives_info = getattr(self, '_objectives_info', {})
                    if objectives_info and hasattr(self, '_y_scheme'):
                        # Use same ordering logic as to_tensor method
                        objective_names = [name for name in self._y_scheme.model_fields.keys() if name in objectives_info]
                        ref_point_list = []
                        for obj_name in objective_names:
                            if obj_name in ref_point:
                                value = ref_point[obj_name]
                                # Apply same transformation as in to_tensor
                                if objectives_info[obj_name] == 'minimize':
                                    value = -value  # Convert to maximization
                                ref_point_list.append(value)
                            else:
                                # Auto-generate missing values
                                if objectives_info[obj_name] == 'maximize':
                                    ref_point_list.append(0.0)  # Conservative for maximize
                                else:
                                    ref_point_list.append(-1000.0)  # Conservative for minimize (becomes positive)
                        ref_point = ref_point_list
                        logger.info(f"Converted named reference point to tensor order: {ref_point}")
                    else:
                        logger.warning("Named reference point provided but no objectives info available")
                        ref_point = None

                # Auto-generate reference point if not provided or conversion failed
                if ref_point is None:
                    logger.info("Auto-generating reference point from data statistics")
                    ref_point = self._auto_generate_reference_point()
                    if ref_point is not None:
                        logger.info(f"Auto-generated reference point: {ref_point}")

                # Final fallback to conservative default
                if ref_point is None:
                    num_objectives = getattr(self, '_objectives_info', {})
                    if num_objectives:
                        ref_point = []
                        for obj_name, direction in num_objectives.items():
                            if direction == 'maximize':
                                ref_point.append(0.0)  # Worst case for maximization
                            else:  # minimize
                                ref_point.append(1000.0)  # Worst case for minimization
                    else:
                        ref_point = [0.0, 1000.0]  # Default 2-objective case

                    logger.warning(f"Using fallback reference point: {ref_point}")

                kwargs['ref_point'] = ref_point

                # Create partitioning using current observations if available
                try:
                    # Get current training data to initialize partitioning
                    if hasattr(self, 'rb') and len(self.rb) > 0:
                        training_data = self.rb[:]
                        if 'y' in training_data and training_data['y'] is not None:
                            Y_observed = training_data['y']

                            # Validate data for qLogEHVI partitioning
                            if Y_observed.shape[0] < 2:
                                logger.warning(f"Only {Y_observed.shape[0]} observation(s) available, qLogEHVI may not work optimally")

                            # Create partitioning from observed data
                            ref_point_tensor = torch.tensor(ref_point, dtype=Y_observed.dtype, device=Y_observed.device)

                            # Validate reference point vs observations for hypervolume computation
                            dominated_count = torch.all(Y_observed >= ref_point_tensor, dim=1).sum().item()
                            logger.debug(f"Reference point validation: {dominated_count}/{Y_observed.shape[0]} points dominate ref_point")

                            if dominated_count == 0:
                                logger.warning("Reference point is not dominated by any observations - this may cause qLogEHVI issues")
                                # Try to adjust reference point automatically
                                adjusted_ref_point = []
                                for i in range(Y_observed.shape[1]):
                                    min_val = Y_observed[:, i].min().item()
                                    adjusted_ref_point.append(min_val - 0.01)
                                ref_point_tensor = torch.tensor(adjusted_ref_point, dtype=Y_observed.dtype, device=Y_observed.device)
                                kwargs['ref_point'] = adjusted_ref_point
                                logger.info(f"Auto-adjusted reference point to: {adjusted_ref_point}")

                            partitioning = FastNondominatedPartitioning(
                                ref_point=ref_point_tensor,
                                Y=Y_observed
                            )
                            kwargs['partitioning'] = partitioning
                            logger.debug(f"Created qLogEHVI partitioning from {len(Y_observed)} observations")

                            # Log some partitioning diagnostics
                            try:
                                pareto_mask = partitioning.pareto_Y.shape[0] if hasattr(partitioning, 'pareto_Y') else 'unknown'
                                logger.debug(f"Partitioning diagnostics: pareto_points={pareto_mask}, ref_point={ref_point}")
                            except:
                                logger.debug(f"Partitioning created successfully with ref_point={ref_point}")

                        else:
                            logger.warning("No training data available for partitioning, using ref_point only")
                    else:
                        logger.warning("No replay buffer data available for partitioning")

                except Exception as e:
                    logger.error(f"Failed to create qLogEHVI partitioning: {e}, using ref_point only")
                    logger.debug(f"Partitioning error details: {type(e).__name__}: {str(e)}")
                    # Remove the partitioning from kwargs if it failed
                    kwargs.pop('partitioning', None)

            base_acq = qLogExpectedHypervolumeImprovement(model, **kwargs)

        elif acq_func in ['qNEHVI', 'qNoisyExpectedHypervolumeImprovement']:
            from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
            logger.debug(f"Building qNEHVI acquisition function with kwargs: {kwargs}")
            base_acq = qNoisyExpectedHypervolumeImprovement(model, **kwargs)

        # Advanced acquisition functions
        elif acq_func == 'qKnowledgeGradient':
            from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
            logger.debug(f"Building qKnowledgeGradient acquisition function")
            base_acq = qKnowledgeGradient(model, **kwargs)

        elif acq_func == 'ThompsonSampling':
            from botorch.acquisition.probabilistic import ThompsonSampling as TSAcquisition
            logger.debug(f"Building ThompsonSampling acquisition function")
            base_acq = TSAcquisition(model, **kwargs)

        else:
            supported_funcs = [
                'ExpectedImprovement', 'LogExpectedImprovement', 'ProbabilityOfImprovement',
                'UpperConfidenceBound', 'PosteriorMean', 'qEHVI', 'qLogEHVI', 'qNEHVI',
                'qKnowledgeGradient', 'ThompsonSampling'
            ]
            raise ValueError(f"Unsupported acquisition function: {acq_func}. "
                           f"Supported functions are: {supported_funcs}")

        # Apply constraint handling if using feasibility method
        if (self.hparams.get('constraint_method') == 'feasibility' and
            self.constraint_gp is not None and
            hasattr(self, '_constraints_info') and self._constraints_info):

            logger.info("Applying constraint-aware acquisition function")
            return self._build_constrained_acquisition(base_acq, model, **kwargs)

        return base_acq

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
        device = self.device
        discrete_choices = [
            torch.tensor(self.x_scheme.get_feature_values(name, encoded=True), device=device)  # choices for each cat dim
            for name in self.x_scheme.cat_fields_to_index_map
        ]
        return discrete_choices

    def optimize(self, acq, q=1, bounds=None, **kwargs):
        logger.info(f"Starting acquisition optimization: q={q}, has_categorical={self.has_categorical()}, has_numerical={self.has_numerical()}")

        num_restarts = self.hparams.get('num_restarts', 200)
        num_restarts = kwargs.pop('num_restarts', num_restarts)

        sequential = self.hparams.get('sequential_opt', True)
        sequential = kwargs.pop('sequential_opt', sequential)

        raw_samples = self.hparams.get('raw_samples', 512)
        raw_samples = kwargs.pop('raw_samples', raw_samples)

        acquisition_options = self.hparams.get('aquisition_options', {})

        logger.debug(f"Optimization settings: num_restarts={num_restarts}, raw_samples={raw_samples}, sequential={sequential}")

        # Handle pure categorical optimization
        if self.has_categorical() and not self.has_numerical():
            logger.debug(f"Using pure categorical optimization")
            return self._optimize_categorical(acq, q=q, **kwargs)

        if self.has_categorical():
            # For mixed problems with few categorical vars, use the fixed features approach
            if self.len_x_cat >= self.hparams.get('n_categorical_features_threshold', 5):
                from botorch.optim import optimize_acqf_mixed_alternating
                optimizer = optimize_acqf_mixed_alternating
                logger.debug(f"Using mixed alternating optimization for {self.len_x_cat} categorical features")

                discrete_dims = list(range(self.len_x_num, self.len_x_num + self.len_x_cat))
                kwargs['discrete_dims'] = discrete_dims

                # set options for max_discrete_values
            else:
                from botorch.optim import optimize_acqf_mixed
                optimizer = optimize_acqf_mixed
                logger.debug(f"Using mixed optimization with fixed features for {self.len_x_cat} categorical features")
                kwargs['fixed_features_list'] = self.x_cat_cartesian_product_list
        else:
            from botorch.optim import optimize_acqf
            optimizer = optimize_acqf
            logger.debug(f"Using continuous optimization")
            kwargs['sequential'] = sequential

            self._optimizer_acqf = optimizer, kwargs

        # Use provided bounds or default to x_bounds
        bounds_to_use = bounds if bounds is not None else self.x_bounds
        logger.debug(f"Starting optimization with {optimizer.__name__}, bounds shape: {bounds_to_use.shape}")
        best_x, acq_val = optimizer(acq, bounds_to_use, q=q, num_restarts=num_restarts, raw_samples=raw_samples,
                                    options=acquisition_options, **kwargs)

        logger.info(f"Optimization completed: best_x.shape={best_x.shape}, acq_val={acq_val}")
        return best_x, acq_val

    def _optimize_categorical(self, acq, q=1, **kwargs):
        """
        Optimize acquisition function for pure categorical problems using discrete optimization.
        """
        import torch
        from itertools import product

        categorical_optimizer = self.hparams.get('categorical_optimizer', 'auto')

        # Get discrete choices for each categorical dimension
        discrete_choices = self.discrete_choices

        # Calculate total number of combinations
        total_combinations = 1
        for choices in discrete_choices:
            total_combinations *= len(choices)

        # Auto-select strategy based on problem size
        if categorical_optimizer == 'auto':
            if total_combinations <= 1000:
                categorical_optimizer = 'grid'
            else:
                categorical_optimizer = 'random'

        device = self.device
        dtype = self._get_dtype(device)

        if categorical_optimizer == 'grid':
            # Grid search over all combinations
            candidates_list = list(product(*[choices.tolist() for choices in discrete_choices]))
            candidates = torch.tensor(candidates_list, dtype=dtype, device=device)

        elif categorical_optimizer == 'random':
            # Random sampling from categorical space
            n_candidates = min(kwargs.get('raw_samples', 512), total_combinations)
            candidates = []
            for _ in range(n_candidates):
                candidate = []
                for choices in discrete_choices:
                    idx = torch.randint(0, len(choices), (1,))
                    candidate.append(choices[idx].item())
                candidates.append(candidate)
            candidates = torch.tensor(candidates, dtype=dtype, device=device)

        else:
            raise ValueError(f"Unsupported categorical optimizer: {categorical_optimizer}. "
                           f"Supported: ['auto', 'grid', 'random']")

        # Evaluate acquisition function on all candidates
        with torch.no_grad():
            acq_values = acq(candidates.unsqueeze(-2))  # Add batch dimension for BoTorch

        # Select top q candidates
        if q == 1:
            best_idx = torch.argmax(acq_values)
            best_x = candidates[best_idx].unsqueeze(0)
            acq_val = acq_values[best_idx]
        else:
            # For batch acquisition, select top q unique candidates
            top_indices = torch.topk(acq_values.flatten(), min(q, len(candidates))).indices
            best_x = candidates[top_indices]
            acq_val = acq_values[top_indices]

        return best_x, acq_val

    def to_tensor(self, x: Optional[list[dict]] = None, y: Optional[list] = None, c: Optional[list[dict]] = None) -> Solution:
        """
        Convert input features and context features to tensors.
        :param x: Input features.
        :param y: Target values (optional).
        :param c: Context features (optional).
        :return: Tuple of tensors (x_tensor, c_tensor).

        """
        if x is not None and (not isinstance(x, list) or not all(isinstance(item, dict) for item in x)):
            raise TypeError("Input features `x` must be a list of dictionaries.")
        if c is not None and (not isinstance(c, list) or not all(isinstance(item, dict) for item in c)):
            raise TypeError("Context features `c` must be a list of dictionaries.")

        # BoTorch works better with float64 for numerical stability, but MPS doesn't support float64
        device = self.device
        dtype = self._get_dtype(device)

        x_num, x_cat = self.x_scheme.encode_batch(x, dtype=dtype) if x is not None else (None, None)
        c_num, c_cat = self.c_scheme.encode_batch(c, dtype=dtype) if c is not None else (None, None)

        # Move tensors to the correct device
        if x_num is not None:
            x_num = x_num.to(device=device)
        if x_cat is not None:
            x_cat = x_cat.to(device=device)
        if c_num is not None:
            c_num = c_num.to(device=device)
        if c_cat is not None:
            c_cat = c_cat.to(device=device)

        if y is not None:
            # Handle multi-objective y_data using y_scheme (BaseParameters)
            if hasattr(self, '_y_scheme') and self._y_scheme is not None:
                # Multi-objective case: use y_scheme for encoding like x_scheme and c_scheme
                logger.info(f"Processing multi-objective y_data with {len(y)} samples using y_scheme")

                if isinstance(y[0], dict):
                    logger.debug(f"Converting multi-objective y_data from dict format using y_scheme BaseParameters")

                    # Validate y_data against y_scheme
                    try:
                        # Test first entry to ensure schema compatibility
                        test_obj = self._y_scheme(**y[0])
                        logger.debug(f"y_scheme validation successful for first entry: {list(y[0].keys())}")
                    except Exception as e:
                        logger.warning(f"y_data doesn't match y_scheme perfectly: {e}, continuing with available fields")

                    # Extract only objectives for GP training (ignore constraints)
                    objectives_info = getattr(self, '_objectives_info', {})
                    if objectives_info:
                        logger.info(f"Extracting {len(objectives_info)} objectives for GP training: {list(objectives_info.keys())}")

                        y_tensors = []
                        objective_transformations = []

                        for y_dict in y:
                            y_row = []
                            # Use schema order instead of alphabetical sort for consistent objective ordering
                            # This ensures the tensor order matches the y_scheme field order
                            objective_names = [name for name in self._y_scheme.model_fields.keys() if name in objectives_info]
                            for obj_name in objective_names:  # Schema order, not alphabetical
                                if obj_name in y_dict:
                                    value = y_dict[obj_name]
                                    original_value = value
                                    # Convert to maximization if needed for consistent GP training
                                    if objectives_info[obj_name] == 'minimize':
                                        value = -value  # Convert minimize to maximize
                                        objective_transformations.append(f"{obj_name}: {original_value} -> {value} (minimize->maximize)")
                                    else:
                                        objective_transformations.append(f"{obj_name}: {value} (maximize)")
                                    y_row.append(value)
                                else:
                                    logger.warning(f"Objective '{obj_name}' missing from y_data entry: {y_dict}")
                                    y_row.append(0.0)  # Default value
                            y_tensors.append(y_row)

                        if objective_transformations:
                            logger.debug(f"Objective transformations applied: {objective_transformations[:3]}{'...' if len(objective_transformations) > 3 else ''}")

                        y = torch.tensor(y_tensors, dtype=dtype, device=device)
                        logger.info(f"Created multi-objective tensor: shape={y.shape}, dtype={y.dtype}")
                    else:
                        logger.warning("Multi-objective enabled but no objectives_info found, treating as scalar")
                        y = as_tensor([list(yi.values())[0] if yi else 0.0 for yi in y], dtype=dtype, device=device)
                        y = y.unsqueeze(-1)
                else:
                    # Assume y is already in tensor format (2D array)
                    logger.debug(f"y_data already in tensor format, converting to torch tensor")
                    y = as_tensor(y, dtype=dtype, device=device)
                    if len(y.shape) == 1:
                        y = y.unsqueeze(-1)
                    logger.info(f"Multi-objective tensor from array: shape={y.shape}, dtype={y.dtype}")
            else:
                # Single-objective case (backward compatibility) - simple list/scalar handling
                logger.debug(f"Processing single-objective y_data with {len(y) if isinstance(y, list) else 1} samples")
                y = as_tensor(y, dtype=dtype, device=device)
                if len(y.shape) == 1:
                    y = y.unsqueeze(-1)
                logger.debug(f"Single-objective tensor: shape={y.shape}, dtype={y.dtype}")
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

    def has_numerical(self):
        """Check if the optimization problem has numerical/continuous variables."""
        return self.len_x_num > 0 or self.len_c_num > 0

    def generate_initial_samples(self, n_samples: int, method: str = None) -> list[dict]:
        """
        Generate initial samples using specified initialization method.

        Args:
            n_samples: Number of samples to generate
            method: Initialization method ('uniform', 'sobol', 'halton', 'random')

        Returns:
            List of parameter dictionaries
        """
        if method is None:
            method = self.hparams.get('initialization_method', 'sobol')

        logger.info(f"Generating {n_samples} initial samples using {method} method")

        device = self.device
        dtype = self._get_dtype(device)

        try:
            if method == 'uniform':
                samples = self._generate_uniform_samples(n_samples, dtype, device)
            elif method == 'sobol':
                samples = self._generate_sobol_samples(n_samples, dtype, device)
            elif method == 'halton':
                samples = self._generate_halton_samples(n_samples, dtype, device)
            elif method == 'random':
                samples = self._generate_random_samples(n_samples, dtype, device)
            else:
                logger.error(f"Unsupported initialization method: {method}")
                raise ValueError(f"Unsupported initialization method: {method}. "
                               f"Supported methods: ['uniform', 'sobol', 'halton', 'random']")

            logger.info(f"Successfully generated {len(samples)} {method} samples")
            return samples

        except Exception as e:
            logger.error(f"Failed to generate {method} samples: {e}")
            logger.debug(f"Sample generation error details: {type(e).__name__}: {str(e)}")
            raise

    def _generate_uniform_samples(self, n_samples: int, dtype: torch.dtype, device: torch.device) -> list[dict]:
        """Generate samples using uniform sampling within bounds."""
        samples = []

        # Get bounds for numerical features
        bounds_map = self.x_scheme.get_bounds()

        for _ in range(n_samples):
            sample = {}

            # Sample numerical features
            for name, width in self.x_scheme.num_fields_w:
                if name in bounds_map:
                    low, high = bounds_map[name]
                    if low is None or high is None:
                        # Use default range if no bounds specified
                        low = low if low is not None else 0.0
                        high = high if high is not None else 1.0
                else:
                    # Default range for unbounded features
                    low, high = 0.0, 1.0

                if width == 1:
                    value = torch.rand(1, dtype=dtype, device=device) * (high - low) + low
                    sample[name] = value.item()
                else:
                    # Fixed-width array
                    values = torch.rand(width, dtype=dtype, device=device) * (high - low) + low
                    sample[name] = values.tolist()

            # Sample categorical features
            for name in self.x_scheme._cat_fields:
                choices = self.x_scheme.get_feature_values(name, encoded=False)
                choice_idx = torch.randint(0, len(choices), (1,)).item()
                sample[name] = choices[choice_idx]

            samples.append(sample)

        return samples

    def _generate_sobol_samples(self, n_samples: int, dtype: torch.dtype, device: torch.device) -> list[dict]:
        """Generate samples using BoTorch's optimized Sobol sequence."""
        try:
            from botorch.utils.sampling import draw_sobol_samples
        except ImportError:
            logger.warning("BoTorch's draw_sobol_samples not available, falling back to PyTorch SobolEngine")
            return self._generate_sobol_samples_fallback(n_samples, dtype, device)

        # Build bounds tensor for BoTorch
        bounds_list = []

        # Add numerical feature bounds
        bounds_map = self.x_scheme.get_bounds()
        for name, width in self.x_scheme.num_fields_w:
            if name in bounds_map:
                low, high = bounds_map[name]
                if low is None or high is None:
                    low = low if low is not None else 0.0
                    high = high if high is not None else 1.0
            else:
                low, high = 0.0, 1.0

            for _ in range(width):
                bounds_list.append([low, high])

        # Add categorical feature bounds (0 to 1 for unit sampling)
        for name in self.x_scheme._cat_fields:
            bounds_list.append([0.0, 1.0])

        if not bounds_list:
            return []

        # Create bounds tensor for BoTorch: shape (2, d)
        bounds = torch.tensor(bounds_list, dtype=dtype, device=device).T

        # Generate Sobol samples using BoTorch
        sobol_samples = draw_sobol_samples(
            bounds=bounds,
            n=n_samples,
            q=1,  # Single point per sample
            seed=torch.randint(0, 2**31, (1,)).item()
        ).squeeze(1)  # Remove q dimension

        # Convert tensor samples back to parameter dictionaries
        samples = []
        for i in range(n_samples):
            sample = {}
            dim_idx = 0

            # Map numerical features
            for name, width in self.x_scheme.num_fields_w:
                if width == 1:
                    sample[name] = sobol_samples[i, dim_idx].item()
                    dim_idx += 1
                else:
                    # Fixed-width array
                    values = [sobol_samples[i, dim_idx + w].item() for w in range(width)]
                    sample[name] = values
                    dim_idx += width

            # Map categorical features
            for name in self.x_scheme._cat_fields:
                choices = self.x_scheme.get_feature_values(name, encoded=False)
                unit_value = sobol_samples[i, dim_idx].item()
                choice_idx = int(unit_value * len(choices))
                choice_idx = min(choice_idx, len(choices) - 1)  # Ensure valid index
                sample[name] = choices[choice_idx]
                dim_idx += 1

            samples.append(sample)

        return samples

    def _generate_sobol_samples_fallback(self, n_samples: int, dtype: torch.dtype, device: torch.device) -> list[dict]:
        """Fallback Sobol implementation using PyTorch's SobolEngine."""
        from torch.quasirandom import SobolEngine

        # Total dimensions = numerical (considering widths) + categorical
        n_dims = self.len_x_num + self.len_x_cat
        if n_dims == 0:
            return []

        # Generate Sobol sequence
        sobol = SobolEngine(dimension=n_dims, scramble=True)
        sobol_samples = sobol.draw(n_samples).to(dtype=dtype, device=device)

        # Get bounds for numerical features
        bounds_map = self.x_scheme.get_bounds()

        samples = []
        for i in range(n_samples):
            sample = {}
            dim_idx = 0

            # Map numerical features
            for name, width in self.x_scheme.num_fields_w:
                if name in bounds_map:
                    low, high = bounds_map[name]
                    if low is None or high is None:
                        low = low if low is not None else 0.0
                        high = high if high is not None else 1.0
                else:
                    low, high = 0.0, 1.0

                if width == 1:
                    unit_value = sobol_samples[i, dim_idx]
                    value = unit_value * (high - low) + low
                    sample[name] = value.item()
                    dim_idx += 1
                else:
                    # Fixed-width array
                    values = []
                    for w in range(width):
                        unit_value = sobol_samples[i, dim_idx + w]
                        value = unit_value * (high - low) + low
                        values.append(value.item())
                    sample[name] = values
                    dim_idx += width

            # Map categorical features
            for name in self.x_scheme._cat_fields:
                choices = self.x_scheme.get_feature_values(name, encoded=False)
                unit_value = sobol_samples[i, dim_idx]
                choice_idx = int(unit_value * len(choices))
                choice_idx = min(choice_idx, len(choices) - 1)  # Ensure valid index
                sample[name] = choices[choice_idx]
                dim_idx += 1

            samples.append(sample)

        return samples

    def _generate_halton_samples(self, n_samples: int, dtype: torch.dtype, device: torch.device) -> list[dict]:
        """Generate samples using scipy's optimized Halton sequence."""
        try:
            from scipy.stats import qmc
        except ImportError:
            logger.warning("scipy.stats.qmc not available, falling back to Sobol sampling")
            return self._generate_sobol_samples(n_samples, dtype, device)

        # Total dimensions = numerical (considering widths) + categorical
        n_dims = self.len_x_num + self.len_x_cat
        if n_dims == 0:
            return []

        # Use scipy's Halton sampler with scrambling for better uniformity
        sampler = qmc.Halton(d=n_dims, scramble=True)
        halton_samples = sampler.random(n=n_samples)

        # Convert to tensor
        halton_samples = torch.tensor(halton_samples, dtype=dtype, device=device)

        # Get bounds for numerical features
        bounds_map = self.x_scheme.get_bounds()

        samples = []
        for i in range(n_samples):
            sample = {}
            dim_idx = 0

            # Map numerical features
            for name, width in self.x_scheme.num_fields_w:
                if name in bounds_map:
                    low, high = bounds_map[name]
                    if low is None or high is None:
                        low = low if low is not None else 0.0
                        high = high if high is not None else 1.0
                else:
                    low, high = 0.0, 1.0

                if width == 1:
                    unit_value = halton_samples[i, dim_idx]
                    value = unit_value * (high - low) + low
                    sample[name] = value.item()
                    dim_idx += 1
                else:
                    # Fixed-width array
                    values = []
                    for w in range(width):
                        unit_value = halton_samples[i, dim_idx + w]
                        value = unit_value * (high - low) + low
                        values.append(value.item())
                    sample[name] = values
                    dim_idx += width

            # Map categorical features
            for name in self.x_scheme._cat_fields:
                choices = self.x_scheme.get_feature_values(name, encoded=False)
                unit_value = halton_samples[i, dim_idx]
                choice_idx = int(unit_value * len(choices))
                choice_idx = min(choice_idx, len(choices) - 1)  # Ensure valid index
                sample[name] = choices[choice_idx]
                dim_idx += 1

            samples.append(sample)

        return samples

    def _generate_random_samples(self, n_samples: int, dtype: torch.dtype, device: torch.device) -> list[dict]:
        """Generate samples using pure random sampling."""
        samples = []

        # Get bounds for numerical features
        bounds_map = self.x_scheme.get_bounds()

        for _ in range(n_samples):
            sample = {}

            # Sample numerical features
            for name, width in self.x_scheme.num_fields_w:
                if name in bounds_map:
                    low, high = bounds_map[name]
                    if low is None or high is None:
                        low = low if low is not None else 0.0
                        high = high if high is not None else 1.0
                else:
                    low, high = 0.0, 1.0

                if width == 1:
                    value = torch.rand(1, dtype=dtype, device=device) * (high - low) + low
                    sample[name] = value.item()
                else:
                    # Fixed-width array
                    values = torch.rand(width, dtype=dtype, device=device) * (high - low) + low
                    sample[name] = values.tolist()

            # Sample categorical features
            import random
            for name in self.x_scheme._cat_fields:
                choices = self.x_scheme.get_feature_values(name, encoded=False)
                sample[name] = random.choice(choices)

            samples.append(sample)

        return samples

    def reset(self):
        """
        Reset the Bayesian model and the replay buffer.
        """
        self.gp = None
        self.constraint_gp = None
        self.rb.reset()
        self._has_categorical = None
        self._x_bounds = None
        self._x_cat_cartesian_product_list = None
        # Clear original y_data for constraints
        if hasattr(self, '_original_y_data'):
            self._original_y_data = []
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
        # Note: y_original is not used in GP training, only for constraint extraction

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

        # Handle cases where x_num or x_cat might be empty/None
        # BoTorch requires all tensors to have the same dtype and device
        device = self.device
        dtype = self._get_dtype(device)

        tensors_to_cat = []
        if x_num is not None and x_num.numel() > 0:
            tensors_to_cat.append(x_num.to(dtype=dtype, device=device))
        if x_cat is not None and x_cat.numel() > 0:
            tensors_to_cat.append(x_cat.to(dtype=dtype, device=device))

        if not tensors_to_cat:
            raise ValueError("Both x_num and x_cat are empty - no features to train on")

        x = torch.cat(tensors_to_cat, dim=-1)

        if c_num is not None:
            tensors_to_cat = [x]
            if c_cat is not None and c_cat.numel() > 0:
                tensors_to_cat.append(c_cat.to(dtype=dtype, device=device))
            if c_num.numel() > 0:
                tensors_to_cat.append(c_num.to(dtype=dtype, device=device))
            x = torch.cat(tensors_to_cat, dim=-1)
            cat_features = list(range(self.len_x_num, self.len_x_num + self.len_x_cat + self.len_c_cat))
        else:
            cat_features = list(range(self.len_x_num, self.len_x_num + self.len_x_cat))

        # Ensure y is also on the correct device and dtype
        if y is not None:
            y = y.to(dtype=dtype, device=device)

        return x, y, cat_features

    def train(self, x: list[dict], y: list, c: Optional[list[dict]] = None, debug=False, **kwargs):
        """
        Initialize the Bayesian model with the provided data.
        :param x: Input features.
        :param y: Target values (optional).
        :param kwargs: Additional keyword arguments for initialization.
        :param c: Context features (optional).
        """
        logger.info(f"Training GP model with {len(x)} samples, debug={debug}")

        if c is not None:
            logger.debug(f"Context features provided: {len(c)} samples")

        s = self.to_tensor(x, y, c)
        self.rb.store_batch(x_num=s.x_num, x_cat=s.x_cat, y=s.y, c_num=s.c_num, c_cat=s.c_cat)

        # Store original y_data separately for constraint extraction
        if not hasattr(self, '_original_y_data'):
            self._original_y_data = []
        self._original_y_data.extend(y)

        self.new_points += len(y)
        logger.debug(f"Replay buffer status: {len(self.rb)} total samples, {self.new_points} new points")

        if len(self.rb) < self.hparams.start_fitting_after_n_points:
            message = f"Not enough points to train the model. New points: {self.new_points}, " \
                      f"Total points: {len(self.rb)}, Start fitting after N points: {self.hparams.start_fitting_after_n_points}."
            logger.info(message)
            return Status(gp=None, message=message)

        if self.new_points < self.hparams.fit_every_n_points and self.gp is not None:
            logger.debug(f"Checking incremental training: {self.new_points} new points < {self.hparams.fit_every_n_points} threshold")

            incremental_fit = self.hparams.incremental_fit
            logger.debug(f"Using incremental fit strategy: {incremental_fit}")

            if incremental_fit == 'none':
                message = f"Skipping model training. New points: {self.new_points}, " \
                          f"Total points: {len(self.rb)}, Fit every N points: {self.hparams.fit_every_n_points}."
                logger.info(message)
                return Status(gp=self.gp, message=message)

            elif incremental_fit == 'fantasy':
                logger.debug(f"Applying fantasy model conditioning with new data")
                x_star, y_star, cat_features = self.get_replay_buffer({'x_num': s.x_num, 'x_cat': s.x_cat,
                                                                       'y': s.y, 'c_num': s.c_num, 'c_cat': s.c_cat})
                self.gp.condition_on_observations(X=x_star, Y=y_star)

                message = f"Model updated with {len(x_star)} fantasy points. New points: {self.new_points}, " \
                          f"Total points: {len(self.rb)}, Fit every N points: {self.hparams.fit_every_n_points}."
                logger.info(message)
                return Status(gp=self.gp, message=message)

            elif incremental_fit == 'full':
                logger.debug(f"Applying full data update to existing model")
                x, y, cat_features = self.get_replay_buffer()
                self.gp.set_train_data(inputs=x, targets=y, strict=False)
                message = f"Model set_train_data with {len(x)} samples. New points: {self.new_points}, " \
                          f"Total points: {len(self.rb)}, Fit every N points: {self.hparams.fit_every_n_points}."
                logger.info(message)
                return Status(gp=self.gp, message=message)

            else:
                message = f"Invalid incremental_fit method: {incremental_fit}. Supported methods are: 'fantasy', 'full', 'none'."
                logger.error(message)
                return Status(gp=None, message="Invalid incremental_fit method.")

        # if we are here, we are training the model from scratch
        logger.info(f"Training model from scratch - resetting new_points counter")
        self.new_points = 0

        # set this boolean if has_categorical is not set yet
        self.has_categorical(s)
        logger.debug(f"Problem type: has_categorical={self.has_categorical()}, has_numerical={self.has_numerical()}")

        x, y, cat_features = self.get_replay_buffer()
        logger.debug(f"Training data prepared: x.shape={x.shape}, y.shape={y.shape}, cat_features={len(cat_features)}")

        self.reset_acquisitions()
        logger.debug(f"Acquisition functions reset")

        try:
            self.gp = self.build_gp_model(x=x, y=y, cat_features=cat_features, **kwargs)
            logger.debug(f"GP model created: {type(self.gp).__name__}")

            logger.debug(f"Starting marginal log-likelihood optimization")
            mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
            fit_gpytorch_mll(mll)
            logger.info(f"GP model hyperparameters optimized successfully")

            # Train constraint models if using feasibility method and constraints exist
            if (self.hparams.get('constraint_method') == 'feasibility' and
                hasattr(self, '_constraints_info') and self._constraints_info):

                logger.info(f"Training constraint models for feasibility method: {list(self._constraints_info.keys())}")
                self._train_constraint_models(x, y, cat_features)

            message = f"Model trained successfully with {len(x)} samples."
            logger.info(message)

        except Exception as e:
            logger.error(f"Failed to train GP model: {e}")
            logger.debug(f"Training error details: {type(e).__name__}: {str(e)}")
            raise

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
            # Use appropriate dtype for device compatibility
            try:
                train_X = self.gp.train_inputs[0]
                dtype, device = train_X.dtype, train_X.device
            except AttributeError:
                device = self.device
                dtype = self._get_dtype(device)

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

    def process_the_context(self, c=None, n_samples=None):
        if c is None:
            logger.debug("No context features provided for processing.")
            return None

        logger.debug(f"Processing context features for sampling")
        s = self.to_tensor(c=c)

        # Build context tensor from available components
        c_components = []
        if s.c_cat is not None and s.c_cat.numel() > 0:
            c_components.append(s.c_cat)
        if s.c_num is not None and s.c_num.numel() > 0:
            c_components.append(s.c_num)

        c_tensor = None
        if c_components:
            c_tensor = torch.cat(c_components, dim=-1)

            # Ensure c_tensor has the right dimensions
            if c_tensor.dim() == 1:
                if n_samples is not None:
                    c_tensor = c_tensor.unsqueeze(0).expand(n_samples, -1)
                else:
                    c_tensor = c_tensor.unsqueeze(0)  # Add batch dimension if missing
            elif c_tensor.dim() > 2:
                c_tensor = c_tensor.view(1, -1)  # Flatten if needed

        return c_tensor

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

        logger.info(f"Sampling {n_samples} candidates from GP model")

        if self.gp is None:
            message = "Model is not trained yet. Please train the model before sampling."
            logger.error(message)
            return Status(gp=None, message=message)

        acq_type = 'single' if n_samples == 1 else 'batch'
        logger.debug(f"Acquisition type: {acq_type}")

        # Handle context features before building acquisition
        context_fixed_acq = None
        if c is not None:
            c_tensor = self.process_the_context(c=c, n_samples=n_samples)

            if c_tensor is not None:
                logger.debug(f"Processing context tensor: shape={c_tensor.shape}, device={c_tensor.device}")

                # Create a simple key for this context configuration
                context_key = f"{acq_type}_ctx_{c_tensor.shape[1]}"  # Use dimension as key
                logger.debug(f"Context key: {context_key}, tensor shape: {c_tensor.shape}")

                if self.acquisitions.get(context_key) is None:
                    logger.debug(f"Building new context-aware acquisition function")
                    base_acq = self.build_acquisition_function(self.gp, q=n_samples, **kwargs)

                    # Fix context features
                    columns = list(range(self.len_x_num + self.len_x_cat, self.total_n_features))
                    logger.debug(f"Context columns to fix: {columns}, total_features: {self.total_n_features}")
                    logger.debug(f"Context tensor for fixing: shape={c_tensor.shape}")

                    try:
                        context_fixed_acq = FixedFeatureAcquisitionFunction(
                            base_acq,
                            d=self.total_n_features,
                            columns=columns,
                            values=c_tensor
                        )
                        self.acquisitions[context_key] = context_fixed_acq
                        logger.debug(f"Context features fixed successfully at columns {columns}")
                    except Exception as e:
                        logger.error(f"Failed to create FixedFeatureAcquisitionFunction: {e}")
                        logger.debug(f"Falling back to base acquisition without context fixing")
                        context_fixed_acq = base_acq
                else:
                    logger.debug(f"Using cached context-aware acquisition function")
                    context_fixed_acq = self.acquisitions[context_key]

                # Use context-aware acquisition if available, otherwise build standard one
        if context_fixed_acq is not None:
            acq = context_fixed_acq
        else:
            if self.acquisitions.get(acq_type) is None:
                logger.debug(f"Building new acquisition function for {acq_type} sampling")
                acq = self.build_acquisition_function(self.gp, q=n_samples, **kwargs)
                self.acquisitions[acq_type] = acq
            else:
                logger.debug(f"Using cached acquisition function for {acq_type} sampling.")
                acq = self.acquisitions[acq_type]

        try:
            # Use appropriate bounds based on whether we have context features
            if context_fixed_acq is not None:
                # When using FixedFeatureAcquisitionFunction, only optimize over x features
                # Slice bounds to include only x_num + x_cat dimensions
                d_x = self.len_x_num + self.len_x_cat
                bounds_to_use = self.x_bounds[:, :d_x]
                logger.debug(f"Using x-only bounds for context-aware optimization: {bounds_to_use.shape}")
            else:
                # Standard optimization over all features
                bounds_to_use = self.x_bounds
                logger.debug(f"Using full bounds for standard optimization: {bounds_to_use.shape}")

            best_x, acq_val = self.optimize(acq, q=n_samples, bounds=bounds_to_use, **kwargs)
            logger.debug(f"Optimization result: best_x.shape={best_x.shape}, acq_val={acq_val}")

            best_x_num = best_x[:, :self.len_x_num]
            best_x_cat = best_x[:, self.len_x_num:self.len_x_num + self.len_x_cat]

            decoded = self.x_scheme.decode_batch(best_x_num, best_x_cat)
            logger.debug(f"Decoded {len(decoded)} parameter dictionaries")

            message = f"Generated {n_samples} samples with acquisition value: {acq_val}"
            logger.info(message)
            
        except Exception as e:
            logger.error(f"Failed to optimize acquisition function: {e}")
            logger.debug(f"Optimization error details: {type(e).__name__}: {str(e)}")
            raise

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

    def _train_constraint_models(self, x: torch.Tensor, y: torch.Tensor, cat_features: list):
        """Train GP models for constraints when using feasibility method."""
        try:
            # Extract constraint values from y tensor based on y_scheme
            constraint_data = self._extract_constraint_data(y)
            
            if constraint_data is None or len(constraint_data) == 0:
                logger.warning("No constraint data available for training constraint models")
                return
            
            device = x.device
            dtype = x.dtype
            
            # Stack all constraint values into a single tensor
            constraint_names = list(constraint_data.keys())
            constraint_values = torch.stack([constraint_data[name] for name in constraint_names], dim=-1)
            
            logger.debug(f"Training constraint model with {len(constraint_names)} constraints: {constraint_names}")
            logger.debug(f"Constraint data shape: {constraint_values.shape}")
            
            # Build and train constraint GP model
            # Use same model type as main GP but for constraints
            self.constraint_gp = self.build_gp_model(x=x, y=constraint_values, cat_features=cat_features)
            
            # Fit constraint model
            constraint_mll = ExactMarginalLogLikelihood(self.constraint_gp.likelihood, self.constraint_gp)
            fit_gpytorch_mll(constraint_mll)
            
            logger.info(f"Constraint models trained successfully for {len(constraint_names)} constraints")
            
        except Exception as e:
            logger.error(f"Failed to train constraint models: {e}")
            logger.debug(f"Constraint training error: {type(e).__name__}: {str(e)}")
            self.constraint_gp = None

    def _extract_constraint_data(self, y: torch.Tensor) -> dict:
        """Extract constraint values from original y_data stored separately."""
        if not hasattr(self, '_constraints_info') or not self._constraints_info:
            return None
        
        if not hasattr(self, '_original_y_data') or not self._original_y_data:
            logger.warning("No original y_data found - constraint extraction not possible")
            return None
        
        constraint_data = {}
        original_y_data = self._original_y_data
        
        # Extract constraint values from original dict format
        if original_y_data and len(original_y_data) > 0:
            device = y.device
            dtype = y.dtype
            
            for constraint_name in self._constraints_info.keys():
                constraint_values = []
                for y_dict in original_y_data:
                    if isinstance(y_dict, dict) and constraint_name in y_dict:
                        constraint_values.append(float(y_dict[constraint_name]))
                    else:
                        logger.warning(f"Constraint '{constraint_name}' missing from y_data entry")
                        constraint_values.append(0.0)  # Default value
                
                if constraint_values:
                    constraint_data[constraint_name] = torch.tensor(constraint_values, dtype=dtype, device=device)
                    logger.debug(f"Extracted {len(constraint_values)} constraint values for '{constraint_name}'")
        
        return constraint_data if constraint_data else None

    def _build_constrained_acquisition(self, base_acq, model, **kwargs):
        """Build constraint-aware acquisition function using feasibility weighting."""
        try:
            # For now, fall back to penalty method approach to avoid tensor issues
            logger.warning("Constraint-aware acquisition has tensor compatibility issues, falling back to penalty method")
            return base_acq
            
        except Exception as e:
            logger.error(f"Failed to create constraint-aware acquisition: {e}")
            return base_acq

    def _auto_generate_reference_point(self):
        """
        Auto-generate reference point from data statistics.
        Returns a reference point that is worse than any realistic objective values.
        """
        try:
            # Get current training data
            if not hasattr(self, 'rb') or len(self.rb) == 0:
                logger.debug("No training data available for auto-generating reference point")
                return None
            
            training_data = self.rb[:]
            if 'y' not in training_data or training_data['y'] is None:
                logger.debug("No y data available for auto-generating reference point")
                return None
            
            Y_observed = training_data['y']
            objectives_info = getattr(self, '_objectives_info', {})
            
            if not objectives_info or not hasattr(self, '_y_scheme'):
                logger.debug("No objectives info available for auto-generating reference point")
                return None
            
            # Calculate statistics for each objective
            objective_names = [name for name in self._y_scheme.model_fields.keys() if name in objectives_info]
            ref_point = []
            
            logger.debug(f"Auto-generating reference point from {Y_observed.shape[0]} observations")
            
            for i, obj_name in enumerate(objective_names):
                if i < Y_observed.shape[1]:
                    # Get values for this objective (already transformed to maximization)
                    obj_values = Y_observed[:, i]
                    
                    # Calculate statistics
                    min_val = torch.min(obj_values).item()
                    max_val = torch.max(obj_values).item()
                    mean_val = torch.mean(obj_values).item()
                    std_val = torch.std(obj_values).item() if len(obj_values) > 1 else abs(mean_val) * 0.1
                    
                    # For hypervolume computation to work, reference point must be dominated by some observations
                    # Use a more aggressive approach: slightly worse than the worst observation
                    margin = max(abs(min_val) * 0.05, abs(std_val) * 0.1, 0.01)  # At least 1% margin
                    conservative_point = min_val - margin
                    
                    # Special handling for constraint-heavy scenarios where all points might be penalized
                    # Check if all values are very negative (indicating heavy constraint penalties)
                    if max_val < -10:  # Heavily penalized data
                        logger.debug(f"Detected heavily penalized data for {obj_name} (max={max_val:.3f})")
                        # Use a reference point based on the penalty structure
                        conservative_point = min_val - abs(min_val) * 0.1
                    
                    # Ensure minimum separation for numerical stability
                    range_val = max_val - min_val
                    if range_val < 1e-6:  # Very small range
                        conservative_point = min_val - 0.1
                    
                    ref_point.append(conservative_point)
                    
                    logger.debug(f"Objective {obj_name}: range=[{min_val:.4f}, {max_val:.4f}], "
                               f"mean={mean_val:.4f}, ref_point={conservative_point:.4f}")
                else:
                    logger.warning(f"Objective {obj_name} not found in Y_observed tensor")
                    ref_point.append(-1.0)  # Conservative fallback
            
            # Validate that the reference point makes sense for hypervolume computation
            ref_point_tensor = torch.tensor(ref_point, dtype=Y_observed.dtype, device=Y_observed.device)
            dominated_points = torch.all(Y_observed >= ref_point_tensor, dim=1).sum().item()
            
            if dominated_points == 0:
                logger.warning(f"Reference point {ref_point} is not dominated by any observations")
                # Adjust reference point to ensure at least one point dominates it
                for i in range(len(ref_point)):
                    ref_point[i] = Y_observed[:, i].min().item() - 0.01
                logger.info(f"Adjusted reference point to: {ref_point}")
            else:
                logger.debug(f"Reference point is dominated by {dominated_points}/{Y_observed.shape[0]} observations")
            
            return ref_point
            
        except Exception as e:
            logger.error(f"Failed to auto-generate reference point: {e}")
            return None
