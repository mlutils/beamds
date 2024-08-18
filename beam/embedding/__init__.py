if len([]):
    from .resource import beam_embedding
    from .robust_encoder import RobustDenseEncoder


__all__ = ['beam_embedding', 'RobustDenseEncoder']


def __getattr__(name):

    if name == 'beam_embedding':
        from .resource import beam_embedding
        return beam_embedding
    elif name == 'RobustDenseEncoder':
        from .robust_encoder import RobustDenseEncoder
        return RobustDenseEncoder
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
