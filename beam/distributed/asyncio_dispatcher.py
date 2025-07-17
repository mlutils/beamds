import asyncio
import uuid
from functools import partial, cached_property
from typing import Dict, Any, Optional, Callable
import threading
import time

from ..processor import MetaAsyncResult, MetaDispatcher


class AsyncioResult(MetaAsyncResult):
    """Async result wrapper for asyncio tasks."""
    
    def __init__(self, task_id: str, task_future: Optional[asyncio.Future] = None, 
                 result_value: Any = None, exception: Optional[Exception] = None):
        super().__init__(None)  # obj is not used in asyncio
        self.task_id = task_id
        self._task_future = task_future
        self._result_value = result_value
        self._exception = exception
        self._start_time = time.time()
        self._end_time = None

    @classmethod
    def from_str(cls, task_id: str, **kwargs):
        """Create result from task ID string."""
        return cls(task_id)

    @property
    def value(self):
        """Get the result value, blocking if necessary."""
        if self._value is None and self.is_ready:
            if self._exception:
                raise self._exception
            self._value = self._result_value
        return self._value

    def wait(self, timeout=None):
        """Wait for task completion with optional timeout."""
        if self.is_ready:
            return self.value
        
        if self._task_future:
            try:
                # Convert timeout to seconds if it's not None
                timeout_sec = timeout if timeout is not None else None
                loop = asyncio.get_event_loop()
                
                # If we're in the same thread as the event loop, we can't use asyncio.wait_for
                # So we'll use a different approach
                if loop.is_running():
                    # We're in the event loop thread, so we need to wait differently
                    while not self.is_ready:
                        if timeout_sec and (time.time() - self._start_time) > timeout_sec:
                            return None
                        time.sleep(0.01)  # Small sleep to avoid busy waiting
                else:
                    # We can use asyncio.wait_for
                    try:
                        result = loop.run_until_complete(
                            asyncio.wait_for(self._task_future, timeout=timeout_sec)
                        )
                        return result
                    except asyncio.TimeoutError:
                        return None
            except Exception as e:
                self._exception = e
                raise e
        
        return None

    @property
    def hex(self):
        """Get task ID as hex string."""
        return self.task_id

    @property
    def is_ready(self):
        """Check if task is completed."""
        if self._is_ready is None:
            if self._task_future:
                self._is_ready = self._task_future.done()
            else:
                self._is_ready = self._result_value is not None or self._exception is not None
        return self._is_ready

    @property
    def state(self):
        """Get task state."""
        if not self.is_ready:
            return 'PENDING'
        elif self._exception:
            return 'FAILURE'
        else:
            return 'SUCCESS'

    @property
    def args(self):
        """Get task arguments (not available in asyncio)."""
        return None

    @property
    def kwargs(self):
        """Get task keyword arguments (not available in asyncio)."""
        return None

    def kill(self):
        """Cancel the task."""
        if self._task_future and not self._task_future.done():
            self._task_future.cancel()

    def __repr__(self):
        return f"AsyncioAsyncResult({self.hex}, is_ready={self.is_ready}, is_success={self.is_success})"


class AsyncioDispatcher(MetaDispatcher):
    """Asyncio-based dispatcher for executing tasks asynchronously."""

    def __init__(self, obj, *args, name=None, loop=None, max_workers=None,
                 asynchronous=True, log_level='INFO', **kwargs):
        """
        Initialize asyncio dispatcher.
        
        Args:
            name: Dispatcher name
            loop: Event loop to use (defaults to current loop)
            max_workers: Maximum number of concurrent workers
            asynchronous: Whether to return async results
            log_level: Logging level
        """
        # In asyncio, obj is not used in the same way as celery
        super().__init__(obj, *args, name=name, asynchronous=asynchronous, **kwargs)
        
        self.loop = loop or asyncio.get_event_loop()
        self.max_workers = max_workers
        self.log_level = log_level
        
        # Task storage
        self._tasks: Dict[str, AsyncioResult] = {}
        self._task_counter = 0
        
        # Semaphore for limiting concurrent tasks
        if max_workers:
            self._semaphore = asyncio.Semaphore(max_workers)
        else:
            self._semaphore = None

    @cached_property
    def broker(self):
        """Get the event loop as the 'broker' (for compatibility)."""
        return self.loop

    def __call__(self, *args, **kwargs):
        """Call the dispatcher with function name."""
        return self.dispatch('function', *args, **kwargs)

    def poll(self, task_id, timeout=0):
        """Poll for task completion."""
        if task_id in self._tasks:
            async_res = self._tasks[task_id]
            return async_res.wait(timeout=timeout)
        return None

    def metadata(self, task_id, *args, **kwargs):
        """Get task metadata."""
        if task_id not in self._tasks:
            return {'task_id': task_id, 'state': 'UNKNOWN', 'error': 'Task not found'}
        
        task = self._tasks[task_id]
        return {
            'task_id': task_id,
            'state': task.state,
            'result': task.value if task.is_ready else None,
            'traceback': str(task._exception) if task._exception else None,
            'status': task.state,
            'children': None,  # asyncio doesn't have task children
            'retries': 0,  # asyncio doesn't have retries
            'parent_id': None,  # asyncio doesn't have parent tasks
            'exception': str(task._exception) if task._exception else None,
            'date_done': task._end_time,
            'runtime': (task._end_time - task._start_time) if task._end_time else None
        }

    def dispatch(self, attribute, *args, **kwargs):
        """Dispatch a task to be executed asynchronously."""
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Create the task function
        async def execute_task():
            """Execute the task with semaphore if needed."""
            if self._semaphore:
                async with self._semaphore:
                    return await self._execute_attribute(attribute, *args, **kwargs)
            else:
                return await self._execute_attribute(attribute, *args, **kwargs)
        
        # Create future and schedule task
        future = asyncio.ensure_future(execute_task(), loop=self.loop)
        
        # Create async result
        async_result = AsyncioResult(task_id, future)
        self._tasks[task_id] = async_result
        
        # Add callback to update result when done
        def task_done(fut):
            try:
                result = fut.result()
                async_result._result_value = result
                async_result._end_time = time.time()
            except Exception as e:
                async_result._exception = e
                async_result._end_time = time.time()
            finally:
                async_result._is_ready = True
        
        future.add_done_callback(task_done)
        
        if self.asynchronous:
            return async_result
        else:
            return async_result.value

    async def _execute_attribute(self, attribute, *args, **kwargs):
        """Execute the actual attribute call."""
        if hasattr(self.obj, attribute):
            func = getattr(self.obj, attribute)
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                # Run synchronous function in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, *args, **kwargs)
        else:
            raise AttributeError(f"Attribute '{attribute}' not found on object")

    def getattr(self, item):
        """Get attribute as a callable."""
        return partial(self.dispatch, item)

    def cleanup(self):
        """Clean up completed tasks."""
        completed_tasks = [task_id for task_id, task in self._tasks.items() 
                         if task.is_ready]
        for task_id in completed_tasks:
            del self._tasks[task_id]

    def get_running_tasks(self):
        """Get list of running task IDs."""
        return [task_id for task_id, task in self._tasks.items() 
                if not task.is_ready]

    def cancel_all(self):
        """Cancel all running tasks."""
        for task_id, task in self._tasks.items():
            if not task.is_ready:
                task.kill()

