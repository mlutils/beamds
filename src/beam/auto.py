import inspect
import importlib
import ast
import os
import sys

from .tabular import DeepTabularAlg
from .path import beam_path

import pkg_resources
import os
import importlib
from pathlib import Path
import warnings
from collections import defaultdict
import pkgutil
from .logger import beam_logger as logger


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
        origin = str(beam_path(spec.submodule_search_locations[0]).resolve())

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
        self.obj = obj

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

        module_path = beam_path(module_path).resolve()
        self_path = beam_path(inspect.getfile(AutoBeam)).resolve()

        try:
            content = module_path.read()
        except:
            logger.warning(f"Could not read module: {module_path}")
            return set()

        ast_tree = ast.parse(content)

        modules = set()
        for a in ast_tree.body:
            if type(a) is ast.Import:
                for ai in a.names:
                    root_name = ai.name.split('.')[0]

                    if is_installed_package(root_name) and not is_std_lib(root_name):
                        modules.add(root_name)
                    elif not is_installed_package(root_name) and not is_std_lib(root_name):
                        if root_name in ['__main__']:
                            continue
                        try:
                            path = beam_path(get_origin(root_name))
                            self.add_private_module_spec(root_name)
                            if path in [module_path, self_path]:
                                continue
                        except ValueError:
                            logger.warning(f"Could not find module: {root_name}")
                            continue
                        modules.union(self.recursive_module_dependencies(path))

            elif type(a) is ast.ImportFrom:

                root_name = a.module.split('.')[0]

                if a.level == 0 and (not is_std_lib(root_name)) and is_installed_package(root_name):
                    modules.add(root_name)
                elif not is_installed_package(root_name) and not is_std_lib(root_name):
                    if a.level == 0:

                        self.add_private_module_spec(root_name)
                        path = beam_path(get_origin(root_name))

                        if path in [module_path, self_path]:
                            continue
                        modules.union(self.recursive_module_dependencies(path))
                    else:

                        path = module_path
                        for i in range(a.level):
                            path = path.parent

                        path = path.joinpath(f"{a.module.replace('.', os.sep)}")
                        if path.is_dir():
                            path = path.joinpath('__init__.py')
                        else:
                            path = path.with_suffix('.py')

                        modules.union(self.recursive_module_dependencies(path))

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
        origin = beam_path(get_origin(module_name))
        if origin.parent.joinpath('__init__.py').is_file():
            module_name = f"{origin.parent.name}.{module_name}"
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
            path = beam_path(path)

        path = path.resolve()

        ab = AutoBeam(obj)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving object's files to path {path}: [requirements.txt, modules.tar.gz, state.pt]")
        ab.write_requirements(path.joinpath('requirements.txt'))
        ab.modules_to_tar(path.joinpath('modules.tar.gz'))
        path.joinpath('metadata.json').write(ab.metadata)
        obj.save_state(path.joinpath('state.pt'))
        path.joinpath('skeleton.pkl').write(obj)

    @classmethod
    def from_path(cls, path):

        import tarfile

        path = beam_path(path).resolve()

        # 1. Check necessary files
        req_file = path.joinpath('requirements.txt')
        modules_tar = path.joinpath('modules.tar.gz')
        state_file = path.joinpath('state.pt')
        skeleton_file = path.joinpath('skeleton.pkl')
        metadata_file = path.joinpath('metadata.json')

        if not all([file.exists() for file in [req_file, modules_tar, state_file, metadata_file]]):
            raise ValueError(f"Path {path} does not contain all necessary files for reconstruction.")

        # 2. Install necessary packages
        os.system(f"pip install -r {req_file}")

        # 3. Extract the Python modules
        extracted_path = path.joinpath('modules')
        extracted_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(modules_tar, "r:gz") as tar:
            tar.extractall(str(extracted_path))

        # 4. Add the directory containing the extracted Python modules to sys.path
        sys.path.insert(0, str(extracted_path))

        # 5. Load metadata and import necessary modules
        metadata = metadata_file.read()

        import_statement = metadata['import_statement']
        imported_class = metadata['type']

        exec(import_statement)

        # 6. Load the state of the object using the imported class type
        cls_obj = locals()[imported_class]

        # 7. Construct the object from its state using a hypothetical from_state method
        # obj = cls_obj.from_path(state_file)
        obj = skeleton_file.read()
        obj.load_state(state_file)

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