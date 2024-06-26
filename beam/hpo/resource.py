
def beam_hpo(framework, *args, **kwargs):

    if framework == 'ray':
        from .ray import RayHPO
        return RayHPO(*args, **kwargs)
    elif framework == 'optuna':
        from .optuna import OptunaHPO
        return OptunaHPO(*args, **kwargs)
    else:
        raise ValueError(f"Unknown framework: {framework}")
