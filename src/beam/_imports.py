import importlib

class BeamImporter:
    def __init__(self):
        self.modules = {}
        self.callables = {}
        self.aliases = {
            'pd': 'pandas',
            'np': 'numpy',
            'F': 'torch.nn.functional',
            'Path': 'pathlib.Path',
            'plt': 'matplotlib.pyplot',
            'sns': 'seaborn',
            'torch': 'torch',
            'nn': 'torch.nn',
            'optim': 'torch.optim',
            'distributions': 'torch.distributions',
            'os': 'os',
            'sys': 'sys',
            'inspect': 'inspect',
            'time': 'time',
            'timedelta': 'datetime.timedelta',
            'random': 'random',
            're': 're',
            'glob': 'glob',
            'pickle': 'pickle',
            'json': 'json',
            'datetime': 'datetime.datetime',
            'date': 'datetime.date',
            'tqdm': 'tqdm.notebook.tqdm',
            'beam': 'beam',

        }
        self.__all__ = list(self.aliases.keys())  # Iterable of attribute names
        self.__file__ = None
        self._initialize_aliases()

    def _initialize_aliases(self):
        pass
        # # Get installed packages
        # installed_packages = {dist.metadata['Name'] for dist in distributions()}
        # # Add installed packages to aliases with their name as alias
        # for pkg in installed_packages:
        #     if pkg not in self.aliases:  # Don't overwrite explicitly stated aliases
        #         self.aliases[pkg] = pkg

    def __getattr__(self, name):
        if name in self.aliases:
            actual_name = self.aliases[name]
        else:
            actual_name = name

        try:
            imported_object = importlib.import_module(actual_name)
        except:
            module_name, object_name = actual_name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            imported_object = getattr(module, object_name)

        return imported_object




# try:
#     from importlib.metadata import distributions
# except ImportError:
#     from importlib_metadata import distributions
#
# import sys
# import importlib
# from types import ModuleType
# import importlib
# import inspect
#
# def get_imported_object_type(import_statement):
#     module_name, object_name = import_statement.rsplit('.', 1)
#     module = importlib.import_module(module_name)
#     imported_object = getattr(module, object_name)
#
#     if inspect.ismodule(imported_object):
#         return 'module'
#     elif callable(imported_object):
#         return 'callable'
#     else:
#         return 'unknown'
#
# class LazyImport:
#     def __init__(self, module_name):
#         self.module_name = module_name
#         self.module = None
#         self._loaded_attributes = {}
#
#     def _import_module(self):
#         module = importlib.import_module(self.module_name)
#         assert isinstance(module, ModuleType), f"{self.module_name} is not a module"
#         return module
#
#     def __getattr__(self, name):
#
#         if self.module is None:
#             self.module = self._import_module()
#
#         if name not in self._loaded_attributes:
#             attribute = getattr(self.module, name)
#
#             # If the attribute is a callable, we can just use it directly
#             if callable(attribute):
#                 self._loaded_attributes[name] = attribute
#             # Otherwise, if it's another module, we make a new LazyImport for it
#             elif isinstance(attribute, type(self.module)):
#                 self._loaded_attributes[name] = LazyImport(f"{self.module_name}.{name}")
#             # Other non-callable attributes can be used directly too
#             else:
#                 self._loaded_attributes[name] = attribute
#
#         return self._loaded_attributes[name]
#
# class LazyCallableImport:
#     def __init__(self, module_name, callable_name):
#         self.module_name = module_name
#         self.callable_name = callable_name
#         self.callable = None
#
#     def __call__(self, *args, **kwargs):
#         if self.callable is None:
#             module = __import__(self.module_name, fromlist=[self.callable_name])
#             self.callable = getattr(module, self.callable_name)
#         return self.callable(*args, **kwargs)
#
#
# class LazyImporter:
#     def __init__(self):
#         self.modules = {}
#         self.callables = {}
#         self.aliases = {
#             'pd': 'pandas',
#             'np': 'numpy',
#             'F': 'torch.nn.functional',
#             'Path': 'pathlib.Path'
#         }
#         self.__all__ = list(self.aliases.keys())  # Iterable of attribute names
#         self.__file__ = None
#         self._initialize_aliases()
#
#     def _initialize_aliases(self):
#         # Get installed packages
#         installed_packages = {dist.metadata['Name'] for dist in distributions()}
#         # Add installed packages to aliases with their name as alias
#         for pkg in installed_packages:
#             if pkg not in self.aliases:  # Don't overwrite explicitly stated aliases
#                 self.aliases[pkg] = pkg
#
#     def __getattr__(self, name):
#         if name in self.aliases:
#             actual_name = self.aliases[name]
#         else:
#             actual_name = name
#
#         if '.' not in actual_name:
#             import_type = 'module'
#         else:
#             try:
#                 import_type = get_imported_object_type(actual_name)
#             except:
#                 import_type = 'module'
#
#         if import_type == 'module':
#             if name not in self.modules:
#                 self.modules[name] = LazyImport(actual_name)
#             return self.modules[name]
#         elif import_type == 'callable':
#             if name not in self.callables:
#                 module_name, callable_name = actual_name.rsplit('.', 1)
#                 self.callables[name] = LazyCallableImport(module_name, callable_name)
#             return self.callables[name]
#         else:
#             raise ImportError(f"Could not import {actual_name}")
#
# sys.modules[__name__] = LazyImporter()