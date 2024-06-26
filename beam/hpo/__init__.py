# Explicit imports for IDE
if len([]):
    from .resource import beam_hpo
    from .params import HPOConfig
    from .ray import RayHPO
    from .optuna import OptunaHPO

__all__ = ['beam_hpo', 'HPOConfig', 'RayHPO', 'OptunaHPO']


def __getattr__(name):
    if name == 'beam_hpo':
        from .resource import beam_hpo
        return beam_hpo
    elif name == 'HPOConfig':
        from .params import HPOConfig
        return HPOConfig
    elif name == 'RayHPO':
        from .ray import RayHPO
        return RayHPO
    elif name == 'OptunaHPO':
        from .optuna import OptunaHPO
        return OptunaHPO
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
