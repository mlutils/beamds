if len([]):
    from .fake import BeamFakeAlg
    from .preprocess import svd_preprocess


__all__ = ['BeamFakeAlg', 'svd_preprocess']


def __getattr__(name):

    if name == 'BeamFakeAlg':
        from .fake import BeamFakeAlg
        return BeamFakeAlg
    elif name == 'svd_preprocess':
        from .preprocess import svd_preprocess
        return svd_preprocess
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")