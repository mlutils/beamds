import inspect
import ast
import os
import sys

from .utils import lazy_property
from .path import beam_path

import pkg_resources
import os
import importlib
from pathlib import Path
import warnings
from collections import defaultdict
import pkgutil
from .logger import beam_logger as logger
from uuid import uuid4 as uuid


class ImportCollector(ast.NodeVisitor):
    def __init__(self):
        self.import_nodes = []

    def visit_Import(self, node):
        self.import_nodes.append(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.import_nodes.append(node)
        self.generic_visit(node)


def get_origin(module_name_or_spec):

    if type(module_name_or_spec) is str:
        module_name = module_name_or_spec

        try:
            spec = importlib.util.find_spec(module_name)
        except:
            return None

        if spec is None:
            return None

    else:
        spec = module_name_or_spec

    if hasattr(spec, 'origin') and spec.origin is not None:
        if spec.origin == 'built-in':
            return spec.origin
        origin = str(beam_path(spec.origin).resolve())
    else:
        try:
            origin = str(beam_path(spec.submodule_search_locations[0]).resolve())
        except:
            origin = None

    return origin


def get_module_paths(spec):

    if spec is None:
        return []

    paths = []

    if hasattr(spec, 'origin') and spec.origin is not None:
        origin = beam_path(spec.origin).resolve()
        if origin.is_file() and origin.parent.joinpath('__init__.py').is_file():
            origin = origin.parent

        paths.append(str(origin))

    if hasattr(spec, 'submodule_search_locations') and spec.submodule_search_locations is not None:
        for path in spec.submodule_search_locations:
            path = beam_path(path).resolve()
            if path.is_file():
                path = path.parent
            paths.append(str(path))

    return list(set(paths))


def classify_module(module_name):

    origin = get_origin(module_name)

    if origin == 'built-in':
        return "stdlib"

    if origin is None:
        return None

    # Get the standard library path using base_exec_prefix
    std_lib_path = beam_path(sys.base_exec_prefix).joinpath('lib')

    # Check for standard library
    if beam_path(origin).parts[:len(std_lib_path.parts)] == std_lib_path.parts:
        return "stdlib"

    # Check for installed packages in site-packages or dist-packages
    elif "site-packages" in origin or "dist-packages" in origin:
        return "installed"

    # Otherwise, it's a custom module
    else:
        return "custom"


def is_std_lib(module_name):
    return classify_module(module_name) == 'stdlib'


def is_installed_package(module_name):
    return classify_module(module_name) == 'installed'


class AutoBeam:

    def __init__(self, obj):
        self._top_levels = None
        self._private_modules_walk = None
        self._module_spec = None
        self._module_dependencies = None
        self._requirements = None
        self._private_modules = None
        self._visited_modules = None
        self.obj = obj

    @lazy_property
    def self_path(self):
        return beam_path(inspect.getfile(AutoBeam)).resolve()

    @lazy_property
    def loaded_modules(self):
        modules = list(sys.modules.keys())
        root_modules = [m.split('.')[0] for m in modules]
        return set(root_modules).union(set(modules))


    @property
    def visited_modules(self):
        if self._visited_modules is None:
            self._visited_modules = set()
        return self._visited_modules

    @property
    def private_modules(self):
        if self._private_modules is None:
            self._private_modules = [self.module_spec]
            _ = self.module_dependencies
        return self._private_modules

    def add_private_module_spec(self, module_name):
        if self._private_modules is None:
            self._private_modules = [self.module_spec]

        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None:
            return
        self._private_modules.append(module_spec)

    @property
    def module_spec(self):
        if self._module_spec is None:
            module_spec = importlib.util.find_spec(type(self.obj).__module__)
            root_module = module_spec.name.split('.')[0]
            self._module_spec = importlib.util.find_spec(root_module)

        return self._module_spec

    @staticmethod
    def module_walk(root_path):

        root_path = beam_path(root_path).resolve()
        module_walk = {}
        # if root_path.is_file():
        #     module_walk = {'..': {root_path.name: root_path.read()}}
        #     return module_walk

        for r, dirs, files in root_path.walk():

            r_relative = r.relative_to(root_path)
            dir_files = {}
            for f in files:
                p = r.joinpath(f)
                if p.suffix == '.py':
                    dir_files[f] = p.read()
            if len(dir_files):
                module_walk[str(r_relative)] = dir_files

        return module_walk

    @property
    def private_modules_walk(self):
        if self._private_modules_walk is None:
            private_modules_walk = {}

            root_paths = set(sum([get_module_paths(m) for m in self.private_modules if m is not None], []))

            for root_path in root_paths:
                private_modules_walk[root_path] = self.module_walk(root_path)

            self._private_modules_walk = private_modules_walk
        return self._private_modules_walk

    def recursive_module_dependencies(self, module_path):

        if module_path is None:
            return set()
        module_path = beam_path(module_path).resolve()
        if str(module_path) in self.visited_modules:
            return set()
        else:
            self.visited_modules.add(str(module_path))

        try:
            content = module_path.read()
        except:
            logger.warning(f"Could not read module: {module_path}")
            return set()

        ast_tree = ast.parse(content)
        collector = ImportCollector()
        collector.visit(ast_tree)
        import_nodes = collector.import_nodes

        modules = set()
        for a in import_nodes:
            if type(a) is ast.Import:
                for ai in a.names:
                    root_name = ai.name.split('.')[0]

                    if is_installed_package(root_name) and not is_std_lib(root_name):
                        if root_name in self.loaded_modules:
                            modules.add(root_name)
                    elif not is_installed_package(root_name) and not is_std_lib(root_name):
                        if root_name in ['__main__']:
                            continue
                        try:
                            self.add_private_module_spec(root_name)
                            path = beam_path(get_origin(ai))
                            if path in [module_path, self.self_path, None]:
                                continue
                        except ValueError:
                            logger.warning(f"Could not find module: {root_name}")
                            continue
                        internal_modules = self.recursive_module_dependencies(path)
                        modules = modules.union(internal_modules)

            elif type(a) is ast.ImportFrom:

                root_name = a.module.split('.')[0]

                if a.level == 0 and (not is_std_lib(root_name)) and is_installed_package(root_name):
                    if root_name in self.loaded_modules:
                        modules.add(root_name)
                elif not is_installed_package(root_name) and not is_std_lib(root_name):
                    if a.level == 0:

                        self.add_private_module_spec(root_name)

                        path = beam_path(get_origin(a.module))
                        if path in [module_path, self.self_path, None]:
                            continue
                        internal_modules = self.recursive_module_dependencies(path)
                        modules = modules.union(internal_modules)

                    else:

                        path = module_path
                        for i in range(a.level):
                            path = path.parent

                        path = path.joinpath(f"{a.module.replace('.', os.sep)}")
                        if path.is_dir():
                            path = path.joinpath('__init__.py')
                        else:
                            path = path.with_suffix('.py')

                        internal_modules = self.recursive_module_dependencies(path)
                        modules = modules.union(internal_modules)

        return modules

    @property
    def module_dependencies(self):

        if self._module_dependencies is None:

            module_path = beam_path(inspect.getfile(type(self.obj))).resolve()
            modules = self.recursive_module_dependencies(module_path)
            self._module_dependencies = list(set(modules))

        return self._module_dependencies

    @property
    def top_levels(self):

        if self._top_levels is None:
            top_levels = {}
            for i, dist in enumerate(pkg_resources.working_set):
                egg_info = beam_path(dist.egg_info).resolve()
                tp_file = egg_info.joinpath('top_level.txt')
                module_name = None
                project_name = dist.project_name

                if egg_info.parent.joinpath(project_name).is_dir():
                    module_name = project_name
                elif egg_info.parent.joinpath(project_name.replace('-', '_')).is_dir():
                    module_name = project_name.replace('-', '_')
                elif egg_info.joinpath('RECORD').is_file():

                    record = egg_info.joinpath('RECORD').read(ext='.txt', readlines=True)
                    for line in record:
                        if '__init__.py' in line:
                            module_name = line.split('/')[0]
                            break
                if module_name is None and tp_file.is_file():
                    module_names = tp_file.read(ext='.txt', readlines=True)
                    module_names = list(filter(lambda x: len(x) >= 2 and (not x.startswith('_')), module_names))
                    if len(module_names):
                        module_name = module_names[0].strip()

                if module_name is None and egg_info.parent.joinpath(f"{project_name.replace('-', '_')}.py").is_file():
                    module_name = project_name.replace('-', '_')

                if module_name is None:
                    # warnings.warn(f"Could not find top level module for package: {project_name}")
                    top_levels[module_name] = project_name
                elif not (module_name):
                    warnings.warn(f"{project_name}: is empty")
                else:
                    if module_name in top_levels:
                        if type(top_levels[module_name]) is list:
                            v = top_levels[module_name]
                        else:
                            v = [top_levels[module_name]]
                            v.append(dist)
                        top_levels[module_name] = v
                    else:
                        top_levels[module_name] = dist

            self._top_levels = top_levels

        return self._top_levels

    @property
    def import_statement(self):
        module_name = type(self.obj).__module__
        # origin = beam_path(get_origin(module_name))
        # if origin.parent.joinpath('__init__.py').is_file():
        #     module_name = f"{origin.parent.name}.{module_name}"
        class_name = type(self.obj).__name__
        return f"from {module_name} import {class_name}"

    @property
    def metadata(self):
        return {'name': self.obj.name, 'type': type(self.obj).__name__, 'import_statement': self.import_statement}

    @staticmethod
    def to_bundle(obj, path=None):

        if path is None:
            path = beam_path('.')
            if hasattr(obj, 'name'):
                path = path.joinpath(obj.name)
            else:
                path = path.joinpath('beam_bundle')
        else:
            path = beam_path(path)

        path = path.resolve()

        ab = AutoBeam(obj)
        path.clean()
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving object's files to path {path}: [requirements.txt, modules.tar.gz, state.pt]")
        ab.write_requirements(path.joinpath('requirements.txt'))
        ab.modules_to_tar(path.joinpath('modules.tar.gz'))
        path.joinpath('metadata.json').write(ab.metadata)
        if hasattr(obj, 'save_state'):
            obj.save_state(path.joinpath('state.pt'))
        else:
            logger.warning(f"Object {obj} does not have a save_state method.")
        try:
            path.joinpath('skeleton.pkl').write(obj)
        except Exception as e:
            path.joinpath('skeleton.pkl').unlink()
            logger.warning(f"Could not pickle object: {obj} ({e}), saving only the state.")

        if not path.joinpath('skeleton.pkl').is_file() and \
                not path.joinpath('state.pt').is_file():
            logger.error(f"Could not save object {obj} to path {path}. "
                         f"Make sure the object has a save_state method and/or "
                         f"it is pickleable.")

    @classmethod
    def from_path(cls, path, cache_path=None):

        if cache_path is None:
            cache_path = beam_path('/tmp/autobeam').joinpath(uuid())
        else:
            cache_path = beam_path(cache_path)

        import tarfile
        path = beam_path(path).resolve()

        # 1. Check necessary files
        req_file = path.joinpath('requirements.txt')
        modules_tar = path.joinpath('modules.tar.gz')
        state_file = path.joinpath('state.pt')
        skeleton_file = path.joinpath('skeleton.pkl')
        metadata_file = path.joinpath('metadata.json')

        if not all([file.exists() for file in [req_file, modules_tar, metadata_file]]) or \
           not any([file.exists() for file in [state_file, skeleton_file]]):
            raise ValueError(f"Path {path} does not contain all necessary files for reconstruction.")

        def load_obj():

            # 3. Extract the Python modules
            cache_path.mkdir(parents=True, exist_ok=True)
            with tarfile.open(modules_tar, "r:gz") as tar:
                tar.extractall(str(cache_path))

            # 4. Add the directory containing the extracted Python modules to sys.path
            sys.path.insert(0, str(cache_path))

            # 5. Load metadata and import necessary modules
            metadata = metadata_file.read()

            import_statement = metadata['import_statement']
            imported_class = metadata['type']

            exec(import_statement, globals())

            # 6. Load the state of the object using the imported class type
            cls_obj = globals()[imported_class]

            # 7. Construct the object from its state using a hypothetical from_state method
            if skeleton_file.is_file():
                try:
                    obj = skeleton_file.read()
                    if state_file.exists:
                        obj.load_state(state_file)
                except:
                    obj = cls_obj.from_path(state_file)
            else:
                obj = cls_obj.from_path(state_file)

            return obj

        try:
            obj = load_obj()
        except ImportError:
            # 2. Install necessary packages
            os.system(f"pip install -r {req_file}")
            obj = load_obj()

        return obj

    def get_pip_package(self, module_name):

        if module_name not in self.top_levels:
            return None
        return self.top_levels[module_name]

    @property
    def requirements(self):
        if self._requirements is None:
            requirements = []
            for module_name in self.module_dependencies:
                pip_package = self.get_pip_package(module_name)
                if pip_package is not None:
                    requirements.append(f"{pip_package.project_name}>={pip_package.version}")
                else:
                    logger.warning(f"Could not find pip package for module: {module_name}")
            self._requirements = requirements

        return self._requirements

    def write_requirements(self, path):
        path = beam_path(path)
        content = '\n'.join(self.requirements)
        content = f"{content}\n"
        path.write(content, ext='.txt')

    def modules_to_tar(self, path):
        path = beam_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import tarfile
        with tarfile.open(str(path), "w:gz") as tar:
            for i, (root_path, sub_paths) in enumerate(self.private_modules_walk.items()):
                root_path = beam_path(root_path)
                for sub_path, files in sub_paths.items():
                    for file_name, _ in files.items():
                        local_name = root_path.joinpath(sub_path, file_name)
                        relative_name = local_name.relative_to(root_path.parent)
                        tar.add(str(local_name), arcname=str(relative_name))