import asyncio
import uuid
import inspect
from functools import partial, cached_property
import datetime
from typing import Any, Dict, Optional, Callable

from ..processor import MetaAsyncResult, MetaDispatcher


class AsyncioAsyncResult(MetaAsyncResult):
    """Asyncio implementation of MetaAsyncResult."""

    def __init__(self, task_obj):
        super().__init__(task_obj)
        self._value = None
        self._is_ready = None
        self._start_time = datetime.datetime.now()
        self._end_time = None

    @classmethod
    def from_str(cls, value, app=None):
        """Create an AsyncioAsyncResult from a task ID string."""
        if app is None:
            raise ValueError("Asyncio dispatcher app is required")
        return cls(app.get_task(value))

    @property
    def value(self):
        """Get the result value."""
        if self._value is None and self.is_ready:
            self._value = self.obj.result
        return self._value

    def wait(self, timeout=None):
        """Wait for the task to complete."""
        if not self.is_ready:
            try:
                # For asyncio tasks, we need to run the event loop if not already running
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If already in an event loop, use wait_for
                    future = asyncio.wait_for(self.obj.get_future(), timeout)
                    if inspect.iscoroutine(future):
                        self._value = loop.run_until_complete(future)
                    else:
                        self._value = future
                else:
                    # Otherwise run the task to completion
                    self._value = loop.run_until_complete(
                        asyncio.wait_for(self.obj.get_future(), timeout)
                    )
                self._is_ready = True
                self._end_time = datetime.datetime.now()
            except asyncio.TimeoutError:
                return None
        return self._value

    @property
    def hex(self):
        """Get the task ID."""
        return self.obj.task_id

    @property
    def is_ready(self):
        """Check if the task is ready."""
        if self._is_ready is None:
            self._is_ready = self.obj.done()
            if self._is_ready and self._end_time is None:
                self._end_time = datetime.datetime.now()
        return self._is_ready

    @property
    def is_success(self):
        """Check if the task completed successfully."""
        return self.is_ready and not self.obj.exception()

    @property
    def state(self):
        """Get the current state of the task."""
        if self.obj.cancelled():
            return "REVOKED"
        elif self.obj.exception():
            return "FAILURE"
        elif self.obj.done():
            return "SUCCESS"
        else:
            return "PENDING"

    @property
    def args(self):
        """Get the arguments the task was called with."""
        return self.obj.args

    @property
    def kwargs(self):
        """Get the keyword arguments the task was called with."""
        return self.obj.kwargs

    @property
    def runtime(self):
        """Get the task runtime in seconds."""
        if self._end_time:
            return (self._end_time - self._start_time).total_seconds()
        return None

    def __repr__(self):
        return f"AsyncioAsyncResult({self.hex}, is_ready={self.is_ready}, is_success={self.is_success})"


class AsyncioTask:
    """Represents an asyncio task with metadata."""

    def __init__(self, coro, task_id, args=None, kwargs=None, parent_id=None):
        self.task_id = task_id
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.parent_id = parent_id
        self.exception_info = None
        self.date_done = None
        self.retries = 0
        self.children = []

        # Create the actual asyncio task
        self.future = asyncio.ensure_future(coro)
        self.future.add_done_callback(self._on_complete)

    def _on_complete(self, _):
        """Called when the task completes."""
        self.date_done = datetime.datetime.now()

    def done(self):
        """Check if the task is done."""
        return self.future.done()

    def cancelled(self):
        """Check if the task was cancelled."""
        return self.future.cancelled()

    def exception(self):
        """Get the exception if the task failed."""
        if self.done() and not self.cancelled():
            exception = self.future.exception()
            if exception:
                self.exception_info = str(exception)
            return exception
        return None

    def get_future(self):
        """Get the underlying future."""
        return self.future

    @property
    def result(self):
        """Get the task result."""
        if self.done() and not self.cancelled() and not self.exception():
            return self.future.result()
        return None


class AsyncioDispatcher(MetaDispatcher):
    """Asyncio implementation of MetaDispatcher."""

    def __init__(self, obj=None, *args, name=None, asynchronous=True, log_level='INFO', **kwargs):
        super().__init__(obj, *args, name=name or "asyncio_tasks", asynchronous=asynchronous, **kwargs)
        self.log_level = log_level
        self._tasks = {}
        self._functions = {}

    @cached_property
    def broker(self):
        """Return self as the broker since we manage tasks internally."""
        return self

    def register_function(self, name, func):
        """Register a function to be available for dispatching."""
        self._functions[name] = func
        return func

    def get_task(self, task_id):
        """Get a task by ID."""
        return self._tasks.get(task_id)

    async def _execute_task(self, func, *args, **kwargs):
        """Execute a task function."""
        return await func(*args, **kwargs) if inspect.iscoroutinefunction(func) else func(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Dispatch the default 'function' task."""
        return self.dispatch('function', *args, **kwargs)

    def poll(self, task_id, timeout=0):
        """Poll for task result."""
        async_res = AsyncioAsyncResult.from_str(task_id, app=self)
        return async_res.wait(timeout=timeout)

    def metadata(self, task_id, *args, **kwargs):
        """Get task metadata."""
        task = self.get_task(task_id)
        if not task:
            return {'task_id': task_id, 'state': 'UNKNOWN'}

        result = task.result
        exception = task.exception()

        return {
            'task_id': task_id,
            'state': 'SUCCESS' if task.done() and not exception else 'FAILURE' if exception else 'PENDING',
            'result': result,
            'traceback': str(exception) if exception else None,
            'status': 'SUCCESS' if task.done() and not exception else 'FAILURE' if exception else 'PENDING',
            'children': task.children,
            'retries': task.retries,
            'parent_id': task.parent_id,
            'exception': str(exception) if exception else None,
            'date_done': task.date_done,
            'runtime': task.runtime if hasattr(task, 'runtime') else None,
        }

    def dispatch(self, attribute, *args, **kwargs):
        """Dispatch a task for execution."""
        func = self._functions.get(attribute)
        if func is None:
            if hasattr(self.obj, attribute):
                func = getattr(self.obj, attribute)
            else:
                raise AttributeError(f"No such task: {attribute}")

        task_id = str(uuid.uuid4())

        # Create a coroutine to execute the function
        coro = self._execute_task(func, *args, **kwargs)

        # Create and store the task
        task = AsyncioTask(coro, task_id, args, kwargs)
        self._tasks[task_id] = task

        # Create the result object
        result = AsyncioAsyncResult(task)

        if self.asynchronous:
            return result
        else:
            return result.wait()

    def getattr(self, item):
        """Get a partial function for dispatching a specific task."""
        return partial(self.dispatch, item)

