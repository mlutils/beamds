from ..path import beam_path
from .task import BeamTask
from .core import BeamParallel


def parallel_copy_path(src, dst, chunklen=10, **kwargs):

    src = beam_path(src)
    dst = beam_path(dst)

    def copy_file(p, f):
        s = p.joinpath(f)
        d = dst.joinpath(path.relative_to(src), f)
        d.parent.mkdir(parents=True, exist_ok=True)
        s.copy(d)

    def copy_files(tasks):
        for t in tasks:
            copy_file(*t)

    walk = list(src.walk())
    jobs = []
    chunk = []
    for path, _, files in walk:
        if len(files) == 0:
            dst.joinpath(path.relative_to(src)).mkdir(parents=True, exist_ok=True)
        for f in files:
            chunk.append((path, f))
            if len(chunk) == chunklen:
                jobs.append(task(copy_files)(chunk))
                chunk = []

    if len(chunk) > 0:
        jobs.append(task(copy_files)(chunk))

    parallel(jobs, **kwargs)


def parallel(tasks, n_workers=0, func=None, method='joblib', progressbar='beam', reduce=False, reduce_dim=0, **kwargs):
    bp = BeamParallel(func=func, n_workers=n_workers, method=method, progressbar=progressbar,
                      reduce=reduce, reduce_dim=reduce_dim, **kwargs)
    return bp(tasks)


# def task(func, name=None, silence=False):
#     return BeamTask(func, name=name, silence=silence)


def task(func=None, *, name=None, silence=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return BeamTask(func, *args, name=name, silence=silence, **kwargs)
        return wrapper

    # Allows usage as both @task and @task(...)
    if func is None:
        return decorator
    else:
        return decorator(func)