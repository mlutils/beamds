
if len([]):
    from .elastic import BeamElastic
    from .resource import beam_elastic


__all__ = ['BeamElastic', 'beam_elastic']


def __getattr__(name):
    if name == 'BeamElastic':
        from .elastic import BeamElastic
        return BeamElastic
    if name == 'beam_elastic':
        from .resource import beam_elastic
        return beam_elastic
    raise AttributeError(f"module {__name__} has no attribute {name}")