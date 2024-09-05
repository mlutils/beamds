from tqdm import tqdm
import random

from ..utils import tqdm_beam
from ..utils import collate_chunks
from ..meta import BeamName
from ..logging import beam_logger as logger
from .tasks import BeamTask, TaskAsyncResult, SyncedResults


class BeamAsync(BeamName):
    def __init__(self, method='apply_async', name=None, silent=False, context='spawn', n_workers=1,
                 backend='redis://localhost', broker='pyamqp://guest@localhost//', local_celery=False):

        super().__init__(name=name)
        self.method = method
        self._name = name
        self.silent = silent

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

    def set_silent(self, silent):
        self.silent = silent

    def set_name(self, name):
        self._name = name

    def run(self, func, *args, **kwargs):

        if self.method == 'apply_async':
            async_result = self.pool.apply_async(func, args, kwargs)
        else:  # self.method == 'celery'
            async_result = self.celery_task.delay(func, *args, **kwargs)

        return TaskAsyncResult(async_result)


class BeamParallel(BeamName):

    def __init__(self, n_workers=0, func=None, method='joblib', progressbar='beam',
                 reduce=False, reduce_dim=0, name=None, shuffle=False, use_dill=False,
                 **kwargs):

        super().__init__(name=name)

        self.func = func
        self.n_workers = n_workers
        self.method = method
        self.reduce = reduce
        self.shuffle = shuffle
        self.reduce_dim = reduce_dim
        self.use_dill = use_dill
        self.queue = []
        self.kwargs = kwargs

        if progressbar == 'beam':
            self.progressbar = tqdm_beam
        elif progressbar == 'tqdm':
            self.progressbar = tqdm
        else:
            self.progressbar = lambda x: x

        # TODO: add support for other methods: apply, apply_async, starmap_async, dask, ray

    def __len__(self):
        return len(self.queue)

    def reset(self):
        self.queue = []

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

        from joblib.externals.loky import set_loky_pickler
        if self.use_dill:
            import dill
            set_loky_pickler('dill')
        else:
            set_loky_pickler('pickle')

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
        for thread in self.progressbar(threads):
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

        from ..distributed import RayClient, RayDispatcher

        if n_workers is None:
            n_workers = self.n_workers

        address = self.kwargs.get('address', None)
        ray_kwargs = self.kwargs.get('ray_kwargs', {})
        ray_kwargs['num_cpus'] = n_workers
        remote_kwargs = self.kwargs.get('remote_kwargs', {})

        RayClient(address=address, ray_kwargs=ray_kwargs)

        tasks = [RayDispatcher(t, remote_kwargs=remote_kwargs, asynchronous=False) for t in self.queue]
        results = [t.run() for t in self.progressbar(tasks)]

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

    def run(self, n_workers=None, method=None, shuffle=None) -> SyncedResults:

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

        if shuffle:
            random.shuffle(self.queue)

        if n_workers <= 1 or len(self.queue) == 1:
            results = [t.run() for t in self.progressbar(self.queue)]
            logger.info(f"Running queue (length={len(self.queue)}) on the main thread: {self.name} with 1 worker")
        else:
            logger.info(f"Start running queue: {self.name}: {len(self.queue)} tasks with {n_workers} workers,"
                        f" method: {method}")
            if method == 'joblib':
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
        return SyncedResults(results)

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
