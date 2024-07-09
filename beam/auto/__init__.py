
__all__ = ['AutoBeam', 'BeamProfiler']


if len([]):
    from .auto import AutoBeam
    from .profiler import BeamProfiler


def __getattr__(name):
    if name == 'AutoBeam':
        from .auto import AutoBeam
        return AutoBeam
    elif name == 'BeamProfiler':
        from .profiler import BeamProfiler
        return BeamProfiler
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")