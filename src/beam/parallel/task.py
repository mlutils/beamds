from ..utils import divide_chunks, collate_chunks, retrieve_name, jupyter_like_traceback, lazy_property, dict_to_list
from ..logger import Timer
from ..logger import beam_logger as logger


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

        return {'name': self.name, 'result': res, 'exception': self.exception}