if len([]):
    from .core import beam_logger
    from .kpi import beam_kpi, BeamResult
    from .exception import BeamError

__all__ = ['beam_logger', 'beam_kpi', 'BeamResult', 'BeamError']


def __getattr__(name):
    if name == 'beam_logger':
        from .core import beam_logger
        return beam_logger
    if name == 'beam_kpi':
        from .kpi import beam_kpi
        return beam_kpi
    if name == 'BeamResult':
        from .kpi import BeamResult
        return BeamResult
    if name == 'BeamError':
        from .exception import BeamError
        return BeamError
    raise AttributeError(f"module {__name__} has no attribute {name}")
