from typing import Union, Dict, List

from ..path import beam_path
from ..utils import retry
from .tasks import BeamTask
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


def parallel(tasks: Union[Dict, List], n_workers=0, func=None, method='threading', progressbar='beam', reduce=False, reduce_dim=0,
             use_dill=False, retries=1, sleep=1, **kwargs):

    if func is not None and retries > 1:
        from ..logging import beam_logger as logger
        func = retry(func, retries=retries, sleep=sleep, logger=logger)

    bp = BeamParallel(func=func, n_workers=n_workers, method=method, progressbar=progressbar,
                      reduce=reduce, reduce_dim=reduce_dim, use_dill=use_dill, **kwargs)
    return bp(tasks).values


def task(func=None, *, name=None, silent=False, silence=None, retries=1, sleep=1):
    silent = silent or silence
    def decorator(func):

        if func is not None and retries > 1:
            from ..logging import beam_logger as logger
            func = retry(func, retries=retries, sleep=sleep, logger=logger)

        def wrapper(*args, **kwargs):
            return BeamTask(func, *args, name=name, silent=silent, **kwargs)

        return wrapper

    # Allows usage as both @task and @task(...)
    if func is None:
        return decorator
    else:
        return decorator(func)