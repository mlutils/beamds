from .utils import tqdm_beam
from tqdm import tqdm
from .utils import divide_chunks, collate_chunks
import multiprocessing as mp
import inspect
from tqdm.contrib.concurrent import process_map, thread_map


def process_async(func, args, mp_context='spawn', num_workers=10):
    ctx = mp.get_context(mp_context)
    with ctx.Pool(num_workers) as pool:
        res = [pool.apply_async(func, (arg,)) for arg in args]
        results = []
        for r in tqdm_beam(res):
            results.append(r.get())

    return results


def parallelize(func, args_list, kwargs_list=None, constant_kwargs=None, chunksize=None,
                context='spawn', workers=10, method='apply_async', progressbar='beam',
                collate=True, dim=0):

    if progressbar == 'beam':
        progressbar = tqdm_beam
    elif progressbar == 'tqdm':
        progressbar = tqdm
    else:
        progressbar = lambda x: x

    if constant_kwargs is None:
        constant_kwargs = {}

    if kwargs_list is None:
        kwargs_list = [{}] * len(args_list)

    if workers == 0:

        results = []
        for args_i, kwargs_i in zip(args_list, kwargs_list):
            results.append(func(*args_i, **{**kwargs_i, **constant_kwargs}))

    else:

        ars = inspect.getfullargspec(func)
        if len(ars.args) == 1:
            if (type(args_list[0]) is tuple and len(args_list[0]) != len(ars.args)) or type(args_list[0]) is not tuple:
                args_list = [(ai,) for ai in args_list]

        if method == 'process_map' or method == 'thread_map':

            mp_method = thread_map if method == 'thread_map' else process_map
            args_list = list(zip(*args_list))

            if chunksize is None:
                chunksize = 1
            results = mp_method(func, *args_list, max_workers=workers, chunksize=chunksize)

        else:

            ctx = mp.get_context(context)

            with ctx.Pool(workers) as pool:

                if method == 'apply_async':

                    res = [pool.apply_async(func, tuple(args_i), {**kwargs_i, **constant_kwargs}) for args_i, kwargs_i in zip(args_list, kwargs_list)]
                    results = []
                    for r in progressbar(res):
                        results.append(r.get())

                # elif method == 'apply':
                #
                #     results = list(progressbar((pool.apply_async(func, tuple(args_i), {**kwargs_i, **constant_kwargs})
                #                             for args_i, kwargs_i in zip(args_list, kwargs_list)), total=len(args_list)))

                elif method in ['starmap', 'map']:
                    results = list(pool.starmap(func, args_list, chunksize=chunksize))

                # elif method == 'starmap_async':
                #
                #     res = pool.map_async(func, args_list, chunksize=chunksize)
                #     results = []
                #     for r in progressbar(res):
                #         results.append(r.get())

                else:
                    raise NotImplementedError

    if collate:
        results = collate_chunks(*results, dim=dim)

    return results