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


def is_std_lib(module_name):

    # Check if module is part of the standard library
    # if module_name in sys.builtin_module_names:
    #     return True

    try:
        spec = importlib.util.find_spec(module_name)
    except:
        return False

    if spec is None:
        return False

    if os.name == 'nt':
        # For windows
        if spec.origin.startswith(sys.base_prefix):
            return True
    else:
        # For unix-based systems
        if 'site-packages' not in spec.origin and 'dist-packages' not in spec.origin:
            return True

    return False


def is_installed_package(module_name):

    try:
        spec = importlib.util.find_spec(module_name)
    except:
        return False

    if spec is None:
        return False

    if 'site-packages' in spec.origin or 'dist-packages' in spec.origin:
        return True

    return False


def is_package_installed(module_name):
    # Check if the package is installed (whether std lib or external)
    return pkgutil.find_loader(module_name) is not None


class AutoBeam:

    def __init__(self, obj):
        self._top_levels = None
        self._module_walk = None
        self._module_spec = None
        self._module_dependencies = None
        self._requirements = None
        self._private_modules = None
        self.obj = obj

    @property
    def private_modules(self):
        if self._private_modules is None:
            self._private_modules = []
            _ = self.module_dependencies
        return self._private_modules

    def add_private_module(self, module_path):
        if self._private_modules is None:
            self._private_modules = []

        module_spec = importlib.util.find_spec(module_path)
        if module_spec is None:
            return
        root_module = module_spec.name.split('.')[0]
        module_spec = importlib.util.find_spec(root_module)
        self._private_modules.append(module_spec)

    @property
    def module_spec(self):
        if self._module_spec is None:
            module_spec = importlib.util.find_spec(type(self.obj).__module__)
            root_module = module_spec.name.split('.')[0]
            self._module_spec = importlib.util.find_spec(root_module)

        return self._module_spec

    @property
    def module_walk(self):
        if self._module_walk is None:
            module_walk = defaultdict(dict)

            for root_path in self.module_spec.submodule_search_locations:

                root_path = beam_path(root_path).resolve()
                if str(root_path) in module_walk:
                    continue

                for r, dirs, files in root_path.walk():

                    r_relative = r.relative_to(root_path)
                    dir_files = {}
                    for f in files:
                        p = r.joinpath(f)
                        if p.suffix == '.py':
                            dir_files[f] = p.read()
                    if len(dir_files):
                        module_walk[str(root_path)][str(r_relative)] = dir_files

                self._module_walk = module_walk
        return self._module_walk

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
                            path = beam_path(importlib.util.find_spec(root_name).origin).resolve()
                            self.add_private_module(root_name)
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
                    self.add_private_module(root_name)
                    if a.level == 0:
                        path = beam_path(importlib.util.find_spec(root_name).origin).resolve()

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
    def metadata(self):
        return {'name': self.obj.name, 'type': type(self.obj)}

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
        ab.module_to_tar(path.joinpath('modules'))
        path.joinpath('metadata.pkl').write(ab.metadata)
        obj.save_state(path.joinpath('state.pt'))

    @classmethod
    def from_path(cls, path):

        import tarfile

        path = beam_path(path).resolve()

        # 1. Check necessary files
        req_file = path.joinpath('requirements.txt')
        modules_tar = path.joinpath('modules')
        state_file = path.joinpath('state.pt')
        metadata_file = path.joinpath('metadata.pkl')

        if not all([file.exists() for file in [req_file, modules_tar, state_file, metadata_file]]):
            raise ValueError(f"Path {path} does not contain all necessary files for reconstruction.")

        # 2. Install necessary packages
        os.system(f"pip install -r {req_file}")

        # 3. Extract the Python modules
        extracted_path = path.joinpath('extracted_modules')
        for m in modules_tar:
            mr_extracted = extracted_path.joinpath(m.name.split('.')[0])
            mr_extracted.mkdir(parents=True, exist_ok=True)
            with tarfile.open(m, "r:gz") as tar:
                tar.extractall(extracted_path)

        # 4. Add the directory containing the extracted Python modules to sys.path
        sys.path.append(str(extracted_path))

        # 5. Load metadata and import necessary modules
        metadata = metadata_file.read()

        module_name = metadata.get('type').__module__
        class_name = metadata.get('type').__name__
        imported_module = importlib.import_module(module_name)
        imported_class = getattr(imported_module, class_name)

        # 6. Load the state of the object using the imported class type
        obj_state = state_file.read()

        # 7. Construct the object from its state using a hypothetical from_state method
        obj = imported_class.from_state(obj_state)

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

    def module_to_tar(self, path):
        path = beam_path(path)
        path.mkdir(parents=True, exist_ok=True)
        import tarfile
        for i, (root_path, sub_paths) in enumerate(self.module_walk.items()):
            root_path = beam_path(root_path)
            with tarfile.open(str(path.joinpath(f"{i}.tar.gz")), "w:gz") as tar:
                for sub_path, files in sub_paths.items():
                    for file_name, _ in files.items():
                        local_name = root_path.joinpath(sub_path, file_name)
                        relative_name = local_name.relative_to(root_path)
                        tar.add(str(local_name), arcname=str(relative_name))