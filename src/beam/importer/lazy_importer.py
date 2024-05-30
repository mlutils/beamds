from functools import lru_cache
import importlib
import sys


class SafeLazyImporter:

    def __init__(self):
        self._modules_cache = {}

    @lru_cache(maxsize=None)
    def has(self, module_name):
        try:
            self._modules_cache[module_name] = importlib.import_module(module_name)
            return True
        except ImportError:
            self._modules_cache[module_name] = None
            return False

    @lru_cache(maxsize=None)
    def is_loaded(self, module_name):
        # Check if the module is already loaded (globally)
        return module_name in sys.modules

    @property
    def torch(self):
        return self._modules_cache['torch']

    @property
    def polars(self):
        return self._modules_cache['polars']

    @property
    def cudf(self):
        return self._modules_cache['cudf']

    @property
    def scipy(self):
        return self._modules_cache['scipy']

    def __getattr__(self, module_name):
        if module_name not in self._modules_cache:
            self._modules_cache[module_name] = importlib.import_module(module_name)
        return self._modules_cache[module_name]

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self._modules_cache = {}


lazy_importer = SafeLazyImporter()
