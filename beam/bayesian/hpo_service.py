import copy
from dataclasses import dataclass

from ..base import BeamBase
from ..logging import beam_logger as logger

from .config import BayesianHPOServiceConfig, BayesianConfig
from .hp_scheme import BaseParameters
from .core import BayesianBeam


@dataclass
class ProblemScheme:
    """
    Scheme for defining a problem in Hyperparameter Optimization (HPO).
    This class is used to define the input and configuration schemes for HPO problems.
    """

    solver: BayesianBeam
    x_scheme: BaseParameters = None
    c_scheme: BaseParameters = None
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

    def register(self, name: str, x_scheme: dict, c_scheme: dict = None, embedding_keys: list[str] = None,
                 config_kwargs: dict = None, **kwargs):
        """
        Register a new HPO problem.

        :param x_scheme: The scheme for the input space.
        :param c_scheme: The scheme for the configuration space (optional).
        :param embedding_keys: Keys for embedding (optional).
        :param config_kwargs: Additional keyword arguments for configuration (optional).
        :param kwargs: Additional keyword arguments.
        """

        config_kwargs = config_kwargs or {}
        local_config = copy.copy(self.hparams.dict())
        local_config.update(config_kwargs)

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
            embedding_keys=embedding_keys,
            config_kwargs=hparams
        )
        self._problems[name] = problem_scheme

        logger.info(f"Registered HPO problem '{name}' with x_scheme: {x_scheme}, ")

        return {'name': name, 'x_scheme': x_scheme.model_json_schema(),
                'c_scheme': c_scheme.model_json_schema() if c_scheme is not None else None,
                'message': f"Problem '{name}' registered successfully.",
                'embedding_keys': embedding_keys,}

    def add(self, name, x: list[dict] | dict, y: list | float, c: list[dict] | dict = None, **kwargs):
        """
        Add a new problem to the HPO service.

        :param name: Name of the problem.
        :param x: Input data for the problem.
        :param c: Configuration data for the problem (optional).
        :param kwargs: Additional keyword arguments.
        """
        if name not in self._problems:
            logger.error(f"Problem '{name}' is not registered. Please register it first.")
            return {'message': f"Problem '{name}' is not registered."}

        problem_scheme = self._problems[name]
        embedding_keys = problem_scheme.embedding_keys
        if isinstance(x, dict):
            x = [x]
        if isinstance(c, dict):
            c = [c]
        if not isinstance(y, list):
            y = [y]

        if c is not None:
            logger.info(f"Converting keys {embedding_keys} to embeddings for problem '{name}'")
            for ci in c:
                for k in embedding_keys:
                    ci[k] = self.embedding_model.encode(ci[k], convert_to_tensor=True)

        solver = problem_scheme.solver
        status = solver.train(x=x,y=y,c=c,**kwargs)
        return {
            'name': name,
            'method': 'add',
            'message': status.message
        }

    def sample(self, name, c: list[dict] | dict = None, n_samples: int = 1, **kwargs):
        """
        Query the HPO service for suggested hyperparameters.
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
        if isinstance(c, dict):
            c = [c]
        if c is not None:
            logger.info(f"Converting keys {embedding_keys} to embeddings for problem '{name}'")
            for ci in c:
                for k in embedding_keys:
                    ci[k] = self.embedding_model.encode(ci[k], convert_to_tensor=True)

        solver = problem_scheme.solver
        suggestions = solver.sample(c=c, n_samples=n_samples, **kwargs)
        return {
            'name': name,
            'method': 'query',
            'suggestions': suggestions
        }



