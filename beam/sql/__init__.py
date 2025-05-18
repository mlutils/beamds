
if len([]):
    from .core import BeamIbis
    from .resource import beam_ibis


__all__ = ['BeamIbis', 'beam_ibis']


def __getattr__(name):
    if name == 'BeamIbis':
        from .core import BeamIbis
        return BeamIbis
    elif name == 'beam_ibis':
        from .resource import beam_ibis
        return beam_ibis
    raise AttributeError(f"module {__name__} has no attribute {name}")
