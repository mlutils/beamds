from concurrent.futures import ThreadPoolExecutor, Future

class ThreadAsyncResult:
    def __init__(self, future: Future):
        self.future = future

    @property
    def value(self):
        return self.future.result()

    def wait(self, timeout=None):
        return self.future.result(timeout=timeout)

    @property
    def is_ready(self):
        return self.future.done()

    @property
    def is_success(self):
        if not self.is_ready:
            return None
        try:
            _ = self.value
            return True
        except Exception:
            return False

    def __str__(self):
        return self.str

    def __repr__(self):
        return f"AsyncResult({self.str}, is_ready={self.is_ready}, is_success={self.is_success})"


class ThreadedCluster:
    def __init__(self, max_workers=None):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn, *args, **kwargs):
        future = self.executor.submit(fn, *args, **kwargs)
        return ThreadAsyncResult(future)

    def map(self, fn, *iterables):
        return self.executor.map(fn, *iterables)

    def shutdown(self, wait=True):
        self.executor.shutdown(wait=wait)


class ThreadedRemoteClass:
    def __init__(self, target_class, asynchronous=False):
        self.target_class = target_class
        self.asynchronous = asynchronous
        self.executor = ThreadPoolExecutor()

    def method_wrapper(self, method):
        def wrapper(*args, **kwargs):
            future = self.executor.submit(method, *args, **kwargs)
            if self.asynchronous:
                return ThreadAsyncResult(future)
            else:
                return future.result()
        return wrapper

    def __getattr__(self, item):
        attr = getattr(self.target_class, item)
        if callable(attr):
            return self.method_wrapper(attr)
        return attr

    def __call__(self, *args, **kwargs):
        return self.method_wrapper(self.target_class.__call__)(*args, **kwargs)
