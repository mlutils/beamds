import copy
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

from ..base import BeamBase
from ..logging import beam_logger as logger
from beam import resource

from .config import BayesianHPOServiceConfig, BayesianConfig
from .hp_scheme import BaseParameters
from .core import BayesianBeam
from .experiment_schemas import (
    ExperimentSuggestionSchema, 
    ExperimentResultSchema, 
    ExperimentSummarySchema,
    OptimizationConfigSchema
)


@dataclass
class ProblemScheme:
    """
    Scheme for defining a problem in Hyperparameter Optimization (HPO).
    This class is used to define the input and configuration schemes for HPO problems.
    """

    solver: BayesianBeam
    x_scheme: BaseParameters = None
    c_scheme: BaseParameters = None
    y_scheme: BaseParameters = None  # Output schema for multi-objective optimization
    embedding_keys: list[str] = None
    config_kwargs: BayesianConfig = None



class HPOService(BeamBase):
    """
    Base class for Hyperparameter Optimization (HPO) services.
    This class provides a common interface for HPO services.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, _config_scheme=BayesianHPOServiceConfig, **kwargs)
        self._problems: dict[str, ProblemScheme] = {}
        self._embedding_model = None
        self._embedding_size = None
        
        # Initialize database logging if configured
        self._dataset = None
        self._db_enabled = False
        self._db_tables_initialized = False
        if self.hparams.dataset is not None:
            try:
                self._dataset = resource(self.hparams.dataset)
                self._db_enabled = True
                logger.info(f"Database logging enabled: {self.hparams.dataset}")
                
                # Initialize database tables on first use
                self._initialize_db_tables()
                
            except Exception as e:
                logger.error(f"Failed to initialize database logging: {e}")
                logger.warning("Continuing without database logging")
                self._db_enabled = False

    @property
    def embedding_model(self):
        """
        Get the embedding model used for HPO.
        """
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.hparams.embedding_model,
                                        trust_remote_code=True,
                                        truncate_dim=self.hparams.truncate_dim)
            self._embedding_model = model
        return self._embedding_model

    @property
    def embedding_size(self):
        """
        Get the size of the embedding used for HPO.
        """
        if self._embedding_size is None:
            self._embedding_size = len(self.embedding_model.encode("test 1 2 3"))
        return self._embedding_size

    def _initialize_db_tables(self):
        """Initialize database tables for experiment logging if they don't exist."""
        if not self._db_enabled or self._db_tables_initialized:
            return
        
        try:
            # Get the dataset level (parent of table)
            if self._dataset.level == 'table':
                dataset_level = self._dataset.parent
            else:
                dataset_level = self._dataset
            
            # Create tables if they don't exist
            table_schemas = {
                'suggestions': ExperimentSuggestionSchema,
                'results': ExperimentResultSchema,
                'summaries': ExperimentSummarySchema,
                'configs': OptimizationConfigSchema
            }
            
            for table_name, schema_class in table_schemas.items():
                table_path = dataset_level / table_name
                if not table_path.exists():
                    logger.info(f"Creating database table: {table_path}")
                    dataset_level.create_table_from_schema(schema_class, table_name)
                else:
                    logger.debug(f"Database table exists: {table_path}")
            
            self._db_tables_initialized = True
            logger.info("Database tables initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database tables: {e}")
            self._db_enabled = False

    def _log_suggestion(self, problem_name: str, suggestion_data: Dict[str, Any], 
                       acquisition_value: float = None, is_initial: bool = False):
        """Log a parameter suggestion to the database."""
        if not self._db_enabled or not self.hparams.log_suggestions:
            return
        
        try:
            suggestion_id = str(uuid.uuid4())
            solver = self._problems[problem_name].solver
            
            record = {
                'suggestion_id': suggestion_id,
                'experiment_name': self.hparams.experiment_name,
                'timestamp': datetime.now(),
                'problem_name': problem_name,
                'iteration': len(solver.rb) + 1,  # Approximate iteration number
                'acquisition_function': solver.hparams.get('acquisition_function', 'unknown'),
                'acquisition_value': acquisition_value or 0.0,
                'parameters': json.dumps(suggestion_data),
                'context': json.dumps({}),  # Could be filled with context data
                'model_type': solver.hparams.get('gp_model', 'SingleTaskGP'),
                'n_observations': len(solver.rb),
                'is_initial': is_initial,
                'initialization_method': solver.hparams.get('initialization_method', '') if is_initial else '',
                'batch_index': 0,  # Could be enhanced for batch optimization
                'batch_size': 1,
            }
            
            # Write to suggestions table
            suggestions_table = self._get_table('suggestions')
            suggestions_table.append_row(record)
            logger.debug(f"Logged suggestion {suggestion_id} for problem '{problem_name}'")
            
            return suggestion_id
            
        except Exception as e:
            logger.error(f"Failed to log suggestion: {e}")
            return None

    def _log_result(self, problem_name: str, suggestion_id: str, parameters: Dict[str, Any],
                   objectives: Any, execution_time: float = None, success: bool = True, 
                   error_message: str = None):
        """Log experiment results to the database."""
        if not self._db_enabled or not self.hparams.log_results:
            return
        
        try:
            result_id = str(uuid.uuid4())
            solver = self._problems[problem_name].solver
            
            # Process objectives data
            if isinstance(objectives, dict):
                objectives_json = json.dumps(objectives)
                is_multi_objective = True
                n_objectives = len([k for k, v in objectives.items() 
                                  if k in getattr(solver, '_objectives_info', {})])
                # Extract constraints
                constraints_info = getattr(solver, '_constraints_info', {})
                constraints = {k: v for k, v in objectives.items() if k in constraints_info}
                constraints_json = json.dumps(constraints)
                constraint_violations = sum(1 for k, v in constraints.items() 
                                          if self._violates_constraint(k, v, constraints_info.get(k, '')))
            elif isinstance(objectives, list):
                objectives_json = json.dumps(objectives)
                is_multi_objective = len(objectives) > 1
                n_objectives = len(objectives)
                constraints_json = json.dumps({})
                constraint_violations = 0
            else:
                objectives_json = json.dumps([objectives])
                is_multi_objective = False
                n_objectives = 1
                constraints_json = json.dumps({})
                constraint_violations = 0
            
            # Determine if this is the best result so far
            best_so_far = self._is_best_result(solver, objectives)
            
            record = {
                'result_id': result_id,
                'suggestion_id': suggestion_id or '',
                'experiment_name': self.hparams.experiment_name,
                'timestamp': datetime.now(),
                'problem_name': problem_name,
                'parameters': json.dumps(parameters),
                'objectives': objectives_json,
                'constraints': constraints_json,
                'metadata': json.dumps({}),
                'execution_time': execution_time or 0.0,
                'success': success,
                'error_message': error_message or '',
                'is_multi_objective': is_multi_objective,
                'n_objectives': n_objectives,
                'constraint_violations': constraint_violations,
                'best_so_far': best_so_far,
            }
            
            # Write to results table
            results_table = self._get_table('results')
            results_table.append_row(record)
            logger.debug(f"Logged result {result_id} for problem '{problem_name}'")
            
            return result_id
            
        except Exception as e:
            logger.error(f"Failed to log result: {e}")
            return None

    def _get_table(self, table_name: str):
        """Get a table handle for database operations."""
        if self._dataset.level == 'table':
            # If dataset points to a specific table, navigate to the correct one
            dataset_level = self._dataset.parent
            return dataset_level / table_name
        else:
            return self._dataset / table_name

    def _violates_constraint(self, constraint_name: str, value: float, constraint_expr: str) -> bool:
        """Check if a value violates a constraint."""
        try:
            if '<=' in constraint_expr:
                limit = float(constraint_expr.split('<=')[1].strip())
                return value > limit
            elif '>=' in constraint_expr:
                limit = float(constraint_expr.split('>=')[1].strip())
                return value < limit
            return False
        except (ValueError, IndexError):
            return False

    def _is_best_result(self, solver, objectives) -> bool:
        """Determine if this is the best result so far (simplified heuristic)."""
        try:
            if hasattr(solver, 'best_f') and solver.best_f is not None:
                if isinstance(objectives, (int, float)):
                    return objectives >= solver.best_f
                elif isinstance(objectives, list) and len(objectives) == 1:
                    return objectives[0] >= solver.best_f
            return False
        except Exception:
            return False

    def _apply_constraint_penalties(self, problem_name: str, solver, y_data: list):
        """Apply penalty method for constraint violations."""
        if solver.hparams.constraint_method != "penalty":
            return y_data
        
        constraints_info = getattr(solver, '_constraints_info', {})
        if not constraints_info:
            # No constraints defined, return as-is
            return y_data
        
        penalty_weight = solver.hparams.penalty_weight
        penalized_y_data = []
        total_penalties_applied = 0
        
        logger.debug(f"Applying constraint penalties with weight {penalty_weight}")
        
        for i, y_sample in enumerate(y_data):
            # Calculate total constraint violation
            total_violation = 0.0
            violations = []
            
            if isinstance(y_sample, dict):
                # Multi-objective case: check each constraint
                for constraint_name, constraint_expr in constraints_info.items():
                    if constraint_name in y_sample:
                        value = y_sample[constraint_name]
                        violation = self._calculate_violation(value, constraint_expr)
                        if violation > 0:
                            violations.append(f"{constraint_name}={value:.3f} violates {constraint_expr}")
                            total_violation += violation
                
                if total_violation > 0:
                    # Apply penalty to the first objective found
                    objectives_info = getattr(solver, '_objectives_info', {})
                    if objectives_info:
                        # Find the first objective to penalize
                        primary_objective = next(iter(objectives_info.keys()))
                        if primary_objective in y_sample:
                            original_value = y_sample[primary_objective]
                            penalty = penalty_weight * total_violation
                            y_sample_copy = y_sample.copy()
                            y_sample_copy[primary_objective] = original_value - penalty
                            
                            logger.debug(f"Sample {i+1}: {primary_objective} {original_value:.3f} -> {y_sample_copy[primary_objective]:.3f} "
                                       f"(penalty: {penalty:.3f}, violations: {', '.join(violations)})")
                            
                            penalized_y_data.append(y_sample_copy)
                            total_penalties_applied += 1
                        else:
                            penalized_y_data.append(y_sample)
                    else:
                        # No objectives found, return as-is
                        penalized_y_data.append(y_sample)
                else:
                    # No violations, return as-is  
                    penalized_y_data.append(y_sample)
                    
            else:
                # Single-objective case: y_sample is a scalar
                # Note: For single objective, constraints must be tracked separately
                # This is a limitation - constraints need to be in y_data dict format
                penalized_y_data.append(y_sample)
        
        if total_penalties_applied > 0:
            logger.info(f"Applied constraint penalties to {total_penalties_applied}/{len(y_data)} samples")
        else:
            logger.debug("No constraint penalties applied - all samples feasible")
        
        return penalized_y_data

    def _calculate_violation(self, value: float, constraint_expr: str) -> float:
        """Calculate the amount of constraint violation."""
        try:
            if '<=' in constraint_expr:
                limit = float(constraint_expr.split('<=')[1].strip())
                return max(0.0, value - limit)  # Positive if violation
            elif '>=' in constraint_expr:
                limit = float(constraint_expr.split('>=')[1].strip())
                return max(0.0, limit - value)  # Positive if violation
            elif '==' in constraint_expr:
                limit = float(constraint_expr.split('==')[1].strip())
                return abs(value - limit)  # Always positive for equality
            else:
                logger.warning(f"Unknown constraint format: {constraint_expr}")
                return 0.0
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing constraint '{constraint_expr}': {e}")
            return 0.0

    def register(self, name: str, x_scheme: dict, c_scheme: dict = None,
                 y_scheme: dict = None, config_kwargs: dict = None, **kwargs):
        """
        Register a new HPO problem.

        :param name: Name of the problem.
        :param x_scheme: The scheme for the input space.
        :param c_scheme: The scheme for the configuration space (optional).
        :param y_scheme: The scheme for the output space (optional, for multi-objective optimization).
        :param config_kwargs: Additional keyword arguments for configuration (optional).
        :param kwargs: Additional keyword arguments.
        """

        embedding_keys = None
        if c_scheme is not None:
            embedding_keys = []
            for k, v in c_scheme['properties'].items():
                if isinstance(v, dict) and v.get('type') == 'string':
                    embedding_keys.append(k)
                    c_scheme['properties'][k] = {
                        'type': 'array',
                        'items': {'type': 'number'},
                        'title': v['title'],
                        'maxItems': self.embedding_size,
                        'minItems': self.embedding_size,
                    }

        # Parse y_scheme to extract objectives and constraints  
        objectives_info = {}
        constraints_info = {}
        is_multi_objective = False
        y_scheme_params = None
        
        if y_scheme is not None:
            logger.info(f"Processing y_scheme for problem '{name}' with {len(y_scheme.get('properties', {}))} outputs")
            
            # Create BaseParameters from y_scheme (just like x_scheme and c_scheme)
            y_scheme_params = BaseParameters.from_json_schema(y_scheme)
            
            for field_name, field_info in y_scheme.get('properties', {}).items():
                if 'objective' in field_info:
                    objectives_info[field_name] = field_info['objective']  # 'maximize' or 'minimize'
                    is_multi_objective = True
                elif 'constraint' in field_info:
                    constraints_info[field_name] = field_info['constraint']  # e.g., '<= 2000'
            
            if len(objectives_info) > 1:
                logger.info(f"Multi-objective optimization detected: {list(objectives_info.keys())}")
            
            if constraints_info:
                logger.info(f"Output constraints detected: {list(constraints_info.keys())}")

        config_kwargs = config_kwargs or {}
        local_config = copy.copy(self.hparams.dict())
        local_config.update(config_kwargs)
        
        # Auto-configure acquisition function for multi-objective problems
        if is_multi_objective and len(objectives_info) > 1:
            current_acq = local_config.get('acquisition_function', 'LogExpectedImprovement')
            if current_acq in ['ExpectedImprovement', 'LogExpectedImprovement']:
                local_config['acquisition_function'] = 'qLogEHVI'
                logger.info(f"Auto-selected qLogEHVI acquisition function for multi-objective problem (was: {current_acq})")
            elif current_acq == 'qEHVI':
                local_config['acquisition_function'] = 'qLogEHVI'
                logger.info(f"Auto-upgraded to qLogEHVI acquisition function for better numerical stability (was: {current_acq})")
            
            # Set default reference point if not provided
            if 'acquisition_kwargs' not in local_config:
                local_config['acquisition_kwargs'] = {}
            
            if 'ref_point' not in local_config['acquisition_kwargs']:
                # Create a conservative reference point based on objectives
                ref_point = []
                for field_name, direction in objectives_info.items():
                    if direction == 'maximize':
                        ref_point.append(0.0)  # Assume worst case is 0
                    else:  # minimize
                        ref_point.append(1000.0)  # Assume worst case is large number
                
                local_config['acquisition_kwargs']['ref_point'] = ref_point
                logger.info(f"Auto-generated reference point for qEHVI: {ref_point}")

        hparams = BayesianConfig(**local_config)

        x_scheme = BaseParameters.from_json_schema(x_scheme)
        c_scheme = BaseParameters.from_json_schema(c_scheme) if c_scheme else None

        embedding_keys = embedding_keys or []
        solver = BayesianBeam(
            x_scheme=x_scheme,
            c_scheme=c_scheme,
            hparams=hparams,
            **kwargs
        )

        problem_scheme = ProblemScheme(
            solver=solver,
            x_scheme=x_scheme,
            c_scheme=c_scheme,
            y_scheme=y_scheme_params,  # Store the BaseParameters instance
            embedding_keys=embedding_keys,
            config_kwargs=hparams
        )
        self._problems[name] = problem_scheme

        # Store objectives and constraints info in the solver for easy access
        solver._objectives_info = objectives_info
        solver._constraints_info = constraints_info
        solver._is_multi_objective = is_multi_objective
        solver._y_scheme = y_scheme_params  # Store BaseParameters for encoding/decoding

        if y_scheme is not None:
            logger.info(f"Registered HPO problem '{name}' with x_scheme: {x_scheme}, "
                       f"y_scheme: {len(objectives_info)} objectives, {len(constraints_info)} constraints")
        else:
            logger.info(f"Registered HPO problem '{name}' with x_scheme: {x_scheme}")

        return {
            'name': name, 
            'x_scheme': x_scheme.model_json_schema(),
                'c_scheme': c_scheme.model_json_schema() if c_scheme is not None else None,
            'y_scheme': y_scheme_params.model_json_schema() if y_scheme_params is not None else None,
            'objectives': objectives_info,
            'constraints': constraints_info,
                'message': f"Problem '{name}' registered successfully.",
            'embedding_keys': embedding_keys,
        }

    def add(self, name, x: list[dict] | dict, y: list | float | dict | list[dict], c: list[dict] | dict = None, **kwargs):
        """
        Add training data to the HPO service.

        :param name: Name of the problem.
        :param x: Input data for the problem.
        :param y: Target data - can be scalar, list of scalars (single-objective), 
                 dict, or list of dicts (multi-objective with constraints).
        :param c: Configuration data for the problem (optional).
        :param kwargs: Additional keyword arguments.
        """
        if name not in self._problems:
            logger.error(f"Problem '{name}' is not registered. Please register it first.")
            return {'message': f"Problem '{name}' is not registered."}

        problem_scheme = self._problems[name]
        embedding_keys = problem_scheme.embedding_keys
        solver = problem_scheme.solver
        
        # Normalize inputs
        if isinstance(x, dict):
            x = [x]
        if isinstance(c, dict):
            c = [c]
        
        # Handle multi-objective y_data
        if isinstance(y, dict):
            # Single sample with multiple objectives
            y = [y]
        elif not isinstance(y, list):
            # Single scalar value
            y = [y]

        # Apply constraint handling if enabled
        if solver.hparams.constraint_method == "penalty":
            y = self._apply_constraint_penalties(name, solver, y)
        
        # Validate and process multi-objective data
        if problem_scheme.y_scheme is not None and isinstance(y[0], dict):
            logger.info(f"Processing multi-objective y_data for problem '{name}': {len(y)} samples with keys {list(y[0].keys())}")
            
            # Validate constraint violations and log warnings (after penalty application)
            constraints_info = getattr(solver, '_constraints_info', {})
            if constraints_info:
                logger.debug(f"Checking {len(constraints_info)} constraints: {list(constraints_info.keys())}")
                violation_count = 0
                
                for i, y_dict in enumerate(y):
                    violations = []
                    for constraint_name, constraint_expr in constraints_info.items():
                        if constraint_name in y_dict:
                            value = y_dict[constraint_name]
                            # Parse constraint (e.g., "<= 2000")
                            if '<=' in constraint_expr:
                                limit = float(constraint_expr.split('<=')[1].strip())
                                if value > limit:
                                    violations.append(f"{constraint_name}={value:.1f} > {limit}")
                            elif '>=' in constraint_expr:
                                limit = float(constraint_expr.split('>=')[1].strip())
                                if value < limit:
                                    violations.append(f"{constraint_name}={value:.1f} < {limit}")
                    
                    if violations:
                        violation_count += 1
                        logger.warning(f"Sample {i+1} violates constraints: {', '.join(violations)}")
                
                if violation_count > 0:
                    logger.info(f"Constraint summary: {violation_count}/{len(y)} samples violate constraints")
                else:
                    logger.info(f"All {len(y)} samples satisfy constraints")
            
            logger.info(f"Added {len(y)} multi-objective samples to problem '{name}'")
        elif not isinstance(y[0], (int, float)):
            # Convert other types to scalars if possible
            try:
                y = [float(yi) for yi in y]
            except (ValueError, TypeError):
                logger.error(f"Unable to convert y_data to numeric format: {y}")
                return {'message': 'Invalid y_data format'}
        
        # Process embeddings
        if c is not None:
            c = copy.deepcopy(c)
            logger.info(f"Converting keys {embedding_keys} to embeddings for problem '{name}'")
            for ci in c:
                for k in embedding_keys:
                    ci[k] = self.embedding_model.encode(ci[k], convert_to_tensor=True)

        # Log results to database if enabled
        suggestion_ids = []
        if self._db_enabled and self.hparams.log_results:
            logger.debug(f"Logging {len(x)} results to database for problem '{name}'")
            for i, (xi, yi) in enumerate(zip(x, y)):
                try:
                    # For now, we don't have suggestion_id from the add call
                    # In a production system, you might want to track this
                    suggestion_id = None  
                    result_id = self._log_result(
                        problem_name=name,
                        suggestion_id=suggestion_id,
                        parameters=xi,
                        objectives=yi,
                        success=True
                    )
                    suggestion_ids.append(result_id)
                except Exception as e:
                    logger.warning(f"Failed to log result {i+1}: {e}")

        status = solver.train(x=x, y=y, c=c, **kwargs)
        return {
            'name': name,
            'method': 'add',
            'message': status.message,
            'logged_results': len(suggestion_ids) if self._db_enabled else 0
        }



    def sample(self, name, c: list[dict] | dict = None, n_samples: int = 1, **kwargs):
        """
        Query the HPO service for suggested hyperparameters.
        Automatically handles initialization if replay buffer doesn't have enough samples.
        
        :param name: Name of the problem.
        :param c: Configuration data for the problem (optional).
        :param n_samples: Number of samples to query (default is 1).
        :param kwargs: Additional keyword arguments.
        """
        if name not in self._problems:
            logger.error(f"Problem '{name}' is not registered. Please register it first.")
            return {'message': f"Problem '{name}' is not registered."}

        problem_scheme = self._problems[name]
        embedding_keys = problem_scheme.embedding_keys
        solver = problem_scheme.solver
        
        # Check if we need initialization
        current_samples = len(solver.rb)
        min_samples = solver.hparams.get('start_fitting_after_n_points', 10)
        
        logger.debug(f"Sampling check for problem '{name}': {current_samples} current samples, {min_samples} needed")
        
        if current_samples < min_samples:
            # Need to generate initial samples
            needed_samples = min_samples - current_samples
            method = solver.hparams.get('initialization_method', 'sobol')
            
            logger.info(f"Insufficient samples ({current_samples}/{min_samples}) for problem '{name}'. "
                       f"Generating {needed_samples} initial samples using {method} method.")
            
            try:
                initial_samples = solver.generate_initial_samples(needed_samples, method=method)
                logger.info(f"Generated {len(initial_samples)} initial samples for problem '{name}'")
                
                # Log initial suggestions to database if enabled
                suggestion_ids = []
                if self._db_enabled and self.hparams.log_suggestions:
                    logger.debug(f"Logging {len(initial_samples)} initial suggestions to database")
                    for sample in initial_samples:
                        try:
                            suggestion_id = self._log_suggestion(
                                problem_name=name,
                                suggestion_data=sample,
                                is_initial=True
                            )
                            suggestion_ids.append(suggestion_id)
                        except Exception as e:
                            logger.warning(f"Failed to log initial suggestion: {e}")
                
                # Return initial samples directly - user should evaluate and add them via add()
                return {
                    'name': name,
                    'method': 'initialize',
                    'initialization_method': method,
                    'message': f"Generated {len(initial_samples)} initial samples using {method}. "
                             f"Please evaluate and add them using add() before requesting optimized samples.",
                    'samples': initial_samples,
                    'logged_suggestions': len(suggestion_ids) if self._db_enabled else 0
                }
            except Exception as e:
                logger.error(f"Failed to generate initial samples for problem '{name}': {e}")
                return {
                    'name': name,
                    'method': 'initialize_error',
                    'message': f"Failed to generate initial samples: {e}",
                    'samples': [],
                }
        
        # We have enough samples, proceed with normal optimization
        logger.info(f"Starting optimization sampling for problem '{name}': requesting {n_samples} samples")
        
        if isinstance(c, dict):
            c = [c]
        c = copy.deepcopy(c)
        if c is not None:
            logger.debug(f"Converting {len(embedding_keys)} embedding keys to vectors for problem '{name}'")
            for ci in c:
                for k in embedding_keys:
                    ci[k] = self.embedding_model.encode(ci[k], convert_to_tensor=True)

        status = solver.sample(c=c, n_samples=n_samples, **kwargs)
        
        num_candidates = len(status.candidates) if status.candidates else 0
        logger.info(f"Optimization completed for problem '{name}': generated {num_candidates} candidates")
        
        # Log optimized suggestions to database if enabled
        suggestion_ids = []
        if self._db_enabled and self.hparams.log_suggestions and status.candidates:
            logger.debug(f"Logging {num_candidates} optimized suggestions to database")
            
            # Extract acquisition value if available
            acquisition_value = None
            if hasattr(status, 'debug') and status.debug and 'acq_val' in status.debug:
                acq_val = status.debug['acq_val']
                if hasattr(acq_val, 'item'):
                    acquisition_value = acq_val.item()
                elif isinstance(acq_val, (int, float)):
                    acquisition_value = acq_val
            
            for i, candidate in enumerate(status.candidates):
                try:
                    suggestion_id = self._log_suggestion(
                        problem_name=name,
                        suggestion_data=dict(candidate),
                        acquisition_value=acquisition_value,
                        is_initial=False
                    )
                    suggestion_ids.append(suggestion_id)
                except Exception as e:
                    logger.warning(f"Failed to log optimized suggestion {i+1}: {e}")
        
        return {
            'name': name,
            'method': 'optimize',
            'message': status.message,
            'samples': [dict(xi) for xi in status.candidates] if status.candidates else [],
            'logged_suggestions': len(suggestion_ids) if self._db_enabled else 0
        }

    def log_result(self, name: str, parameters: Dict[str, Any], objectives: Any, 
                  suggestion_id: str = None, execution_time: float = None, 
                  success: bool = True, error_message: str = None, **metadata):
        """
        Explicitly log experiment results to the database.
        
        This method allows users to log results with additional timing and metadata information
        that might not be captured in the automatic logging from add().
        
        :param name: Name of the problem.
        :param parameters: Dictionary of parameters that were evaluated.
        :param objectives: Objective value(s) - scalar, list, or dict.
        :param suggestion_id: Optional suggestion ID to link with a previous suggestion.
        :param execution_time: Time taken to evaluate the parameters (in seconds).
        :param success: Whether the evaluation was successful.
        :param error_message: Error message if evaluation failed.
        :param metadata: Additional metadata to log.
        """
        if not self._db_enabled:
            logger.warning("Database logging is not enabled")
            return None
        
        if name not in self._problems:
            logger.error(f"Problem '{name}' is not registered")
            return None
        
        try:
            result_id = self._log_result(
                problem_name=name,
                suggestion_id=suggestion_id,
                parameters=parameters,
                objectives=objectives,
                execution_time=execution_time,
                success=success,
                error_message=error_message
            )
            
            logger.info(f"Logged result {result_id} for problem '{name}'")
            return {
                'result_id': result_id,
                'message': 'Result logged successfully'
            }
            
        except Exception as e:
            logger.error(f"Failed to log result: {e}")
            return {
                'result_id': None,
                'message': f'Failed to log result: {e}'
            }

    def get_experiment_summary(self, name: str = None):
        """
        Get experiment statistics from the database.
        
        :param name: Optional problem name to filter by. If None, returns stats for all problems.
        :return: Dictionary with experiment statistics.
        """
        if not self._db_enabled:
            return {'message': 'Database logging is not enabled'}
        
        try:
            results_table = self._get_table('results')
            suggestions_table = self._get_table('suggestions')
            
            # Build query filters
            filters = {}
            if name:
                filters['problem_name'] = name
            if self.hparams.experiment_name:
                filters['experiment_name'] = self.hparams.experiment_name
            
            # Get basic statistics
            stats = {
                'total_suggestions': 0,
                'total_results': 0,
                'success_rate': 0.0,
                'problems': [],
            }
            
            # Query suggestions
            suggestions_query = suggestions_table.query_table
            if filters:
                for field, value in filters.items():
                    suggestions_query = suggestions_query.filter(getattr(suggestions_query, field) == value)
            
            suggestions_df = suggestions_query.as_df()
            stats['total_suggestions'] = len(suggestions_df)
            
            # Query results
            results_query = results_table.query_table
            if filters:
                for field, value in filters.items():
                    results_query = results_query.filter(getattr(results_query, field) == value)
            
            results_df = results_query.as_df()
            stats['total_results'] = len(results_df)
            
            if len(results_df) > 0:
                stats['success_rate'] = results_df['success'].mean()
                stats['problems'] = results_df['problem_name'].unique().tolist()
                
                # Add timing statistics
                if 'execution_time' in results_df.columns:
                    stats['avg_execution_time'] = results_df['execution_time'].mean()
                    stats['total_execution_time'] = results_df['execution_time'].sum()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get experiment summary: {e}")
            return {'message': f'Failed to get experiment summary: {e}'}
        



