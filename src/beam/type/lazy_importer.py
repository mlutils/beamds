from functools import wraps, lru_cache
import threading
import importlib
import sys


def singleton(cls):
    instances = {}
    lock = threading.Lock()

    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper


@singleton
class LazyImporter:

    def __init__(self):
        self._modules_cache = {}

    @lru_cache(maxsize=None)
    def has(self, module_name):
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False

    def is_loaded(self, module_name):
        # Check if the module is already loaded (globally)
        return module_name in sys.modules

    def __getattr__(self, module_name):
        if module_name not in self._modules_cache:
            self._modules_cache[module_name] = LazyModule(module_name)
        return self._modules_cache[module_name]

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self._modules_cache = {}


class LazyModule:
    def __init__(self, module_name):
        self._module_name = module_name
        self._module = None
        self._lock = threading.Lock()

    def __getattr__(self, name):
        if self._module is None:
            with self._lock:
                if self._module is None:
                    self._module = importlib.import_module(self._module_name)
        return getattr(self._module, name)

    def __getstate__(self):
        return self._module_name

    def __setstate__(self, state):
        self._module_name = state
        self._module = None
        self._lock = threading.Lock()


lazy_importer = LazyImporter()
