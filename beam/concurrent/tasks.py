from collections import namedtuple

from ..meta import BeamName
from ..utils import Timer, jupyter_like_traceback, dict_to_list, cached_property
from ..logging import beam_logger as logger


TaskSyncedResult = namedtuple('TaskSyncedResult', ['name', 'result', 'exception'])


class TaskAsyncResult:
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


class SyncedResults:

    def __init__(self, results):
        # results is a list of dicts with keys: name, result, exception
        self.results = results

    @cached_property
    def results_map(self):
        return {r.name: r for r in self.results}

    @cached_property
    def failed(self):
        failed = {r.name: r.exception for r in self.results if r.exception is not None}
        return dict_to_list(failed, convert_str=False)

    @cached_property
    def succeeded(self):
        succeeded = {r.name: r.result for r in self.results if r.exception is None}
        return dict_to_list(succeeded, convert_str=False)

    @cached_property
    def values(self):
        vals = {r.name: r.result if r.exception is None else r for r in self.results}
        return dict_to_list(vals, convert_str=False)

    @cached_property
    def exceptions(self):
        vals = {r.name: {'exception': r.exception, 'traceback': r.result}
                for r in self.results if r.exception is not None}
        return dict_to_list(vals, convert_str=False)


class BeamTask(BeamName):

    def __init__(self, func, *args, name=None, silent=False, metadata=None, **kwargs):

        super().__init__(name=name, dynamic_name=False)

        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.pid = None
        self.is_pending = True
        self.result = None
        self.exception = None
        self.metadata = metadata
        self.queue_id = -1
        self.silent = silent

    def set_silent(self, silent):
        self.silent = silent

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self

    def run(self) -> TaskSyncedResult:

        metadata = f"({self.metadata})" if self.metadata is not None else ""

        if not self.silent:
            logger.info(f"Starting task: {self.name} {metadata}")
        try:
            with Timer(logger, silent=True) as t:
                res = self.func(*self.args, **self.kwargs)
                self.result = res
                if not self.silent:
                    logger.info(f"Finished task: {self.name} {metadata}. Elapsed time: {t.elapsed}")
        except Exception as e:
            self.exception = e
            logger.error(f"Task {self.name}{metadata} failed with exception: {e} (set --debug to see full traceback or check in log file)")
            res = jupyter_like_traceback()
            logger.debug(res)
        finally:
            self.is_pending = False

        # return {'name': self.name, 'result': res, 'exception': self.exception}
        return TaskSyncedResult(self.name, res, self.exception)
