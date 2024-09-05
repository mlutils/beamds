# Explicit imports for IDE
if len([]):
    from .utils import parallel_copy_path, parallel, task
    from .tasks import BeamTask, TaskAsyncResult
    from .core import BeamParallel, BeamAsync

__all__ = ['BeamTask', 'TaskAsyncResult', 'BeamParallel', 'BeamAsync', 'parallel_copy_path', 'parallel', 'task']


def __getattr__(name):
    if name == 'BeamTask':
        from .tasks import BeamTask
        return BeamTask
    elif name == 'TaskResult':
        from .tasks import TaskAsyncResult
        return TaskAsyncResult
    elif name == 'BeamParallel':
        from .core import BeamParallel
        return BeamParallel
    elif name == 'BeamAsync':
        from .core import BeamAsync
        return BeamAsync
    elif name == 'parallel_copy_path':
        from .utils import parallel_copy_path
        return parallel_copy_path
    elif name == 'parallel':
        from .utils import parallel
        return parallel
    elif name == 'task':
        from .utils import task
        return task
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")