# Explicit imports for IDE
if len([]):
    from .core import Processor
    from .dispatcher import MetaDispatcher, MetaAsyncResult

__all__ = ['Processor', 'MetaDispatcher', 'MetaAsyncResult']


def __getattr__(name):
    if name == 'Processor':
        from .core import Processor
        return Processor
    elif name == 'MetaDispatcher':
        from .dispatcher import MetaDispatcher
        return MetaDispatcher
    elif name == 'MetaAsyncResult':
        from .dispatcher import MetaAsyncResult
        return MetaAsyncResult
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
