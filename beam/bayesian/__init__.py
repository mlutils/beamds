# Explicit imports for IDE
if len([]):
    from .core import BayesianBeam
    from .config import BayesianConfig, BayesianHPOServiceConfig
    from .hp_scheme import BaseParameters
    from .hpo_service import HPOService, ProblemScheme

__all__ = [
    'BayesianBeam',
    'BayesianConfig',
    'BayesianHPOServiceConfig',
    'BaseParameters',
    'HPOService',
    'ProblemScheme'
]


def __getattr__(name):
    if name == 'BayesianBeam':
        from .core import BayesianBeam
        return BayesianBeam
    elif name == 'BayesianConfig':
        from .config import BayesianConfig
        return BayesianConfig
    elif name == 'BayesianHPOServiceConfig':
        from .config import BayesianHPOServiceConfig
        return BayesianHPOServiceConfig
    elif name == 'BaseParameters':
        from .hp_scheme import BaseParameters
        return BaseParameters
    elif name == 'HPOService':
        from .hpo_service import HPOService
        return HPOService
    elif name == 'ProblemScheme':
        from .hpo_service import ProblemScheme
        return ProblemScheme
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")