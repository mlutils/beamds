from functools import lru_cache, cached_property
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
        except Exception as e:
            print(f"Error in importing {module_name}: {e}, skipping...")
            self._modules_cache[module_name] = None
            return False

    @staticmethod
    def is_loaded(module_name):
        # Check if the module is already loaded (globally)
        return module_name in sys.modules

    @cached_property
    def torch(self):
        return self._getattr('torch')

    @cached_property
    def polars(self):
        return self._getattr('polars')

    @cached_property
    def cudf(self):
        return self._getattr('cudf')

    @cached_property
    def scipy(self):
        return self._getattr('scipy')

    @cached_property
    def optuna(self):
        return self._getattr('optuna')

    @cached_property
    def PIL(self):
        return self._getattr('PIL')

    @cached_property
    def pil_image(self):
        return self._getattr('PIL.Image')

    def _getattr(self, module_name):
        if module_name not in self._modules_cache:
            self.has(module_name)
        return self._modules_cache[module_name]

    def __getattr__(self, module_name):
        return self._getattr(module_name)

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self._modules_cache = {}


lazy_importer = SafeLazyImporter()
