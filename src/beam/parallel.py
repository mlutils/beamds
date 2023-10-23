from .utils import tqdm_beam
from tqdm import tqdm
from .utils import divide_chunks, collate_chunks, retrieve_name, jupyter_like_traceback
import inspect
from .logger import Timer
from .logger import beam_logger as logger
from .path import beam_path
import random

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


def task(func, name=None, silence=False):
    return BeamTask(func, name=name, silence=silence)


class TaskResult:
    def __init__(self, async_result):

        from celery.result import AsyncResult as CeleryAsyncResult
        from multiprocessing.pool import AsyncResult as MultiprocessingAsyncResult

        self.async_result = async_result
        if isinstance(async_result, CeleryAsyncResult):
            self.method = 'celery'
        elif isinstance(async_result, MultiprocessingAsyncResult):
            self.method = 'apply_async'
        else:
            raise ValueError(
                "Invalid async_result type. It must be either CeleryAsyncResult or MultiprocessingAsyncResult.")

    @property
    def done(self):
        if self.method == 'celery':
            return self.async_result.ready()
        else:  # method == 'apply_async'
            return self.async_result.ready()

    @property
    def result(self):

        if self.method == 'celery':
            return self.async_result.result if self.done else None
        else:  # method == 'apply_async'
            return self.async_result.get() if self.done else None


class BeamAsync(object):
    def __init__(self, method='apply_async', name=None, silence=False, context='spawn', n_workers=1,
                 backend='redis://localhost', broker='pyamqp://guest@localhost//', local_celery=False):
        self.method = method
        self._name = name
        self.silence = silence

        if method == 'apply_async':

            import multiprocessing as mp
            ctx = mp.get_context(context)
            self.pool = ctx.Pool(n_workers)

        elif method == 'celery':

            import celery
            import threading
            # Creating a celery instance with redis as the result backend
            celery_app = celery.Celery('tasks', broker=broker, backend=backend)

            if local_celery:
                worker = celery.bin.worker.worker(app=celery_app)
                options = {
                    'pool': 'solo',
                    'concurrency': n_workers,
                    'loglevel': 'INFO',
                }
                threading.Thread(target=worker.run, kwargs=options).start()

            @celery_app.task
            def celery_task(func, *args, **kwargs):
                return func(*args, **kwargs)

            self.celery_task = celery_task
        else:
            raise ValueError('method must be one of {apply_async, celery}')

    @property
    def name(self):
        if self._name is None:
            self._name = retrieve_name(self)
        return self._name

    def set_silent(self, silence):
        self.silence = silence

    def set_name(self, name):
        self._name = name

    def run(self, func, *args, **kwargs):

        if self.method == 'apply_async':
            async_result = self.pool.apply_async(func, args, kwargs)
        else:  # self.method == 'celery'
            async_result = self.celery_task.delay(func, *args, **kwargs)

        return TaskResult(async_result)


class BeamTask(object):

    def __init__(self, func, *args, name=None, silence=False, **kwargs):

        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._name = name
        self.pid = None
        self.is_pending = True
        self.result = None
        self.exception = None
        self.queue_id = -1
        self.silence = silence

    @property
    def name(self):
        if self._name is None:
            self._name = retrieve_name(self)
        return self._name

    def set_silent(self, silence):
        self.silence = silence

    def set_name(self, name):
        self._name = name

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self

    def run(self):

        if not self.silence:
            logger.info(f"Starting task: {self.name}")
        try:
            with Timer(silence=True) as t:
                res = self.func(*self.args, **self.kwargs)
                self.result = res
                if not self.silence:
                    logger.info(f"Finished task: {self.name}. Elapsed time: {t.elapsed}")
        except Exception as e:
            self.exception = e
            logger.error(f"Task {self.name} failed with exception: {e}")
            res = jupyter_like_traceback()
        finally:
            self.is_pending = False

        return res


class BeamParallel(object):

    def __init__(self, n_workers=0, func=None, method='joblib', progressbar='beam',
                 reduce=False, reduce_dim=0, name=None, shuffle=False,
                 **kwargs):

        self.func = func
        self.n_workers = n_workers
        self.method = method
        self.reduce = reduce
        self.shuffle = shuffle
        self.reduce_dim = reduce_dim
        self.queue = []
        self.kwargs = kwargs
        self._name = name

        if progressbar == 'beam':
            self.progressbar = tqdm_beam
        elif progressbar == 'tqdm':
            self.progressbar = tqdm
        else:
            self.progressbar = lambda x: x

        # TODO: add support for other methods: apply, apply_async, starmap_async, dask, ray

    def set_name(self, name):
        self._name = name

    @property
    def name(self):
        if self._name is None:
            self._name = retrieve_name(self)
        return self._name

    def __len__(self):
        return len(self.queue)

    def add(self, *args, **kwargs):

        args_list = []
        args_dict = {}
        for a in args:
            if isinstance(args[0], BeamTask):
                if a.name is None:
                    a.set_name(len(self.queue))
                self.queue.append(a)
            else:
                args_list.append(a)

        for k, v in kwargs.items():
            if isinstance(v, BeamTask):
                self.queue.append(v)
                v.set_name(k)
            else:
                args_dict[k] = v

        if len(args_list) > 0 or len(args_dict) > 0:
            self.add_task(*args_list, **args_dict)

    def add_task(self, *args, name=None, **kwargs):

        if self.func is None:
            func = args[0]
            args = args[1:]
        else:
            func = self.func

        if name is None:
            name = len(self.queue)

        t = BeamTask(func, *args, name=name, **kwargs)

        self.queue.append(t)
        return t

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        if len(self.queue):
            self.run()

    def _run_joblib(self, n_workers=None):

        if n_workers is None:
            n_workers = self.n_workers

        from joblib import Parallel, delayed

        if 'verbose' not in self.kwargs:
            self.kwargs['verbose'] = 10

        # for task in self.queue:
        #     task.set_silent(True)

        results = Parallel(n_jobs=n_workers, **self.kwargs)(delayed(t.run)() for t in self.queue)
        return results

    def _run_process_map(self, n_workers=None):

        if n_workers is None:
            n_workers = self.n_workers

        from tqdm.contrib.concurrent import process_map

        if 'chunksize' not in self.kwargs.keys() or self.kwargs['chunksize'] is None:
            self.kwargs['chunksize'] = 1

        if self.func is None:
            func = self.queue[0].func
        else:
            func = self.func

        results = process_map(func, *list(zip(*[t.args for t in self.queue])), max_workers=n_workers,
                              **self.kwargs)

        return results

    def _run_thread_map(self, n_workers=None):

        if n_workers is None:
            n_workers = self.n_workers

        from tqdm.contrib.concurrent import thread_map

        if 'chunksize' not in self.kwargs.keys() or self.kwargs['chunksize'] is None:
            self.kwargs['chunksize'] = 1

        if self.func is None:
            func = self.queue[0].func
        else:
            func = self.func

        results = thread_map(func, *list(zip(*[t.args for t in self.queue])), max_workers=n_workers, **self.kwargs)

        return results

    def _run_threading(self, n_workers=None):

        if n_workers is None:
            n_workers = self.n_workers

        import threading
        from queue import Queue

        def worker(task, semaphore, results_queue):
            with semaphore:
                res = task.run()
                results_queue.put(res)

        results_queue = Queue()
        semaphore = threading.Semaphore(n_workers)
        threads = []

        for task in self.queue:
            thread = threading.Thread(target=worker, args=(task, semaphore, results_queue))
            thread.start()
            threads.append(thread)

        # Optionally, you can wait for all threads to complete
        for thread in threads:
            thread.join()

        results = [results_queue.get() for _ in range(len(self.queue))]
        return results

    def _run_starmap(self, n_workers=None):

        if n_workers is None:
            n_workers = self.n_workers

        import multiprocessing as mp

        if 'chunksize' not in self.kwargs.keys() or self.kwargs['chunksize'] is None:
            self.kwargs['chunksize'] = 1

        if 'context' in self.kwargs:
            context = self.kwargs['context']
        else:
            context = 'spawn'

        if self.func is None:
            func = self.queue[0].func
        else:
            func = self.func

        ctx = mp.get_context(context)

        with ctx.Pool(n_workers) as pool:
            results = list(pool.starmap(func, *[t.args for t in self.queue], **self.kwargs))

        return results

    def _run_apply_async(self, n_workers=None):

        if n_workers is None:
            n_workers = self.n_workers

        import multiprocessing as mp

        if 'context' in self.kwargs:
            context = self.kwargs['context']
        else:
            context = 'spawn'

        ctx = mp.get_context(context)

        with ctx.Pool(n_workers) as pool:

            tasks = [pool.apply_async(t.run) for t in self.queue]
            results = []
            for res in self.progressbar(tasks):
                results.append(res.get())

        return results

    def _run_apply(self, n_workers=None):

        if n_workers is None:
            n_workers = self.n_workers

        import multiprocessing as mp

        if 'context' in self.kwargs:
            context = self.kwargs['context']
        else:
            context = 'spawn'

        ctx = mp.get_context(context)

        with ctx.Pool(n_workers) as pool:

            results = []
            for t in self.queue:
                results.append(pool.apply(t.run))

        return results

    def _run_ray(self, n_workers=None):

        if n_workers is None:
            n_workers = self.n_workers

        import ray

        def func_wrapper(t):
            return t.run()

        if 'runtime_env' in self.kwargs:
            runtime_env = self.kwargs.pop('runtime_env')
        else:
            runtime_env = {}

        if 'dashboard_port' in self.kwargs:
            dashboard_port = self.kwargs.pop('dashboard_port')
        else:
            dashboard_port = None

        if 'include_dashboard' in self.kwargs:
            include_dashboard = self.kwargs.pop('include_dashboard')
        else:
            include_dashboard = True

        ray.init(runtime_env=runtime_env, dashboard_port=dashboard_port,
                 include_dashboard=include_dashboard, dashboard_host="0.0.0.0", ignore_reinit_error=True)

        tasks = [ray.remote(num_cpus=n_workers)(func_wrapper).remote(t) for t in self.queue]
        results = ray.get(tasks)

        return results

    def _run_dask(self, n_workers=None):

        if n_workers is None:
            n_workers = self.n_workers

        # see https://docs.dask.org/en/latest/scheduler-overview.html#configuring-the-schedulers
        # for more info on dask scheduler options
        # set scheduler='single-threaded' for debugging

        if 'context' in self.kwargs:
            context = self.kwargs['context']
        else:
            context = 'spawn'

        import dask

        with dask.config.set({"multiprocessing.context": context, **self.kwargs}):
            tasks = [dask.delayed(t.run)() for t in self.queue]
            results = dask.compute(tasks, num_workers=n_workers)

        return results[0]

    def run(self, n_workers=None, method=None, shuffle=None):

        if shuffle is None:
            shuffle = self.shuffle

        if n_workers is None:
            n_workers = self.n_workers

        n_workers = min(n_workers, len(self.queue))
        if method is None:
            method = self.method

        if len(self.queue) == 0:
            logger.info(f"Queue {self.name} is empty, returning empty list.")
            return []

        logger.info(f"Start running queue: {self.name}: {len(self.queue)} tasks with {n_workers} workers,"
                    f" method: {method}")

        if shuffle:
            random.shuffle(self.queue)

        if n_workers <= 1 or len(self.queue) == 1:
            results = [t.run() for t in self.progressbar(self.queue)]
        elif method == 'joblib':
            results = self._run_joblib(n_workers=n_workers)
        elif method == 'process_map':
            results = self._run_process_map(n_workers=n_workers)
        elif method == 'apply_async':
            results = self._run_apply_async(n_workers=n_workers)
        elif method == 'thread_map':
            results = self._run_thread_map(n_workers=n_workers)
        elif method in ['starmap', 'map']:
            results = self._run_starmap(n_workers=n_workers)
        elif method == 'ray':
            results = self._run_ray(n_workers=n_workers)
        elif method == 'threading':
            results = self._run_threading(n_workers=n_workers)
        elif method == 'dask':
            results = self._run_dask(n_workers=n_workers)

        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Finish running queue: {self.name}.")
        return results

    def __call__(self, tasks=None, func=None, args_list=None, n_workers=None, method=None,
                 kwargs_list=None, name_list=None):

        if n_workers is None:
            n_workers = self.n_workers
        if method is None:
            method = self.method

        if tasks is not None:
            if isinstance(tasks, list):
                self.add(*tasks)
            elif isinstance(tasks, dict):
                self.add(**tasks)
            else:
                raise ValueError("tasks must be a list or a dict")

        if func is None:
            func = self.func

        if (args_list is not None) or (kwargs_list is not None):
            if args_list is None:
                args_list = [()] * len(kwargs_list)
            if kwargs_list is None:
                kwargs_list = [{}] * len(args_list)
            if name_list is None:
                name_list = [None] * len(args_list)

            for args, kwargs, name in zip(args_list, kwargs_list, name_list):
                self.add_task(func, *args, **kwargs)

        results = self.run(method=method, n_workers=n_workers)

        if self.reduce:
            results = self._reduce(results)

        return results

    def _reduce(self, results):
        results = collate_chunks(*results, dim=self.reduce_dim)
        return results
