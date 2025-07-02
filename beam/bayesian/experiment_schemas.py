"""
Database schemas for Bayesian optimization experiment logging.

These schemas define the structure for logging experiment suggestions and results
to a database using the BeamIbis system.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from beam.sql.schema import BeamIbisSchema


class ExperimentSuggestionSchema(BeamIbisSchema):
    """Schema for logging parameter suggestions from the Bayesian optimization process."""
    
    suggestion_id: str          # Unique identifier for this suggestion
    experiment_name: str        # Name of the experiment/problem
    timestamp: datetime         # When the suggestion was made
    problem_name: str           # Problem identifier
    iteration: int              # Optimization iteration number
    acquisition_function: str   # Which acquisition function was used
    acquisition_value: float    # The acquisition function value
    parameters: str             # JSON string of suggested parameters
    context: str                # JSON string of context features (if any)
    model_type: str             # Type of GP model used
    n_observations: int         # Number of observations when suggestion was made
    is_initial: bool            # Whether this was an initial/random suggestion
    initialization_method: str  # Method used for initial sampling (if applicable)
    batch_index: int            # Index within batch (for batch optimization)
    batch_size: int             # Total batch size


class ExperimentResultSchema(BeamIbisSchema):
    """Schema for logging experiment results and outcomes."""
    
    result_id: str              # Unique identifier for this result
    suggestion_id: str          # Links to the suggestion that produced this result
    experiment_name: str        # Name of the experiment/problem
    timestamp: datetime         # When the result was recorded
    problem_name: str           # Problem identifier
    parameters: str             # JSON string of parameters used
    objectives: str             # JSON string of objective values
    constraints: str            # JSON string of constraint values (if any)
    metadata: str               # JSON string of additional metadata
    execution_time: float       # Time taken to evaluate (seconds)
    success: bool               # Whether evaluation was successful
    error_message: str          # Error message if evaluation failed
    is_multi_objective: bool    # Whether this was multi-objective optimization
    n_objectives: int           # Number of objectives
    constraint_violations: int  # Number of constraint violations
    best_so_far: bool           # Whether this is the best result so far


class ExperimentSummarySchema(BeamIbisSchema):
    """Schema for logging experiment summaries and statistics."""
    
    summary_id: str             # Unique identifier for this summary
    experiment_name: str        # Name of the experiment/problem
    timestamp: datetime         # When the summary was created
    problem_name: str           # Problem identifier
    total_iterations: int       # Total number of iterations
    total_evaluations: int      # Total number of function evaluations
    best_objective: float       # Best objective value found
    best_parameters: str        # JSON string of best parameters
    convergence_iteration: int  # Iteration where best was found
    average_evaluation_time: float # Average time per evaluation
    total_experiment_time: float # Total experiment duration
    success_rate: float         # Fraction of successful evaluations
    model_stats: str            # JSON string of model statistics
    acquisition_stats: str     # JSON string of acquisition function stats
    status: str                 # Experiment status (running, completed, failed)


class OptimizationConfigSchema(BeamIbisSchema):
    """Schema for logging optimization configuration and hyperparameters."""
    
    config_id: str              # Unique identifier for this configuration
    experiment_name: str        # Name of the experiment/problem
    timestamp: datetime         # When the configuration was recorded
    problem_name: str           # Problem identifier
    config_hash: str            # Hash of configuration for deduplication
    hparams: str                # JSON string of all hyperparameters
    x_scheme: str               # JSON string of input parameter schema
    y_scheme: str               # JSON string of output parameter schema
    c_scheme: str               # JSON string of context parameter schema
    device: str                 # Device used (cpu, cuda, mps)
    backend: str                # Backend used (BoTorch, etc.)
    version: str                # Version of the optimization library 