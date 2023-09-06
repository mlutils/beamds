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


def is_std_lib(module_name):
    # Check if module is part of the standard library
    if module_name in sys.builtin_module_names:
        return True

    spec = importlib.util.find_spec(module_name)
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
    spec = importlib.util.find_spec(module_name)

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
        self.obj = obj

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
                        module_walk[str(root_path)][r_relative] = dir_files

                self._module_walk = module_walk
        return self._module_walk

    def recursive_module_dependencies(self, module_name=None, module_origin=None):

        if module_name is not None:
            module_spec = importlib.util.find_spec(module_name)
            module_origin = module_spec.origin

        content = beam_path(module_origin).read()
        ast_tree = ast.parse(content)

        modules = set()
        for a in ast_tree.body:
            if type(a) is ast.Import:
                for ai in a.names:
                    root_name = ai.name.split('.')[0]
                    if root_name != module_name and not is_std_lib(root_name):
                        modules.add(root_name)
                    elif not is_installed_package(root_name):
                        modules.union(self.recursive_module_dependencies(ai.name))

            elif type(a) is ast.ImportFrom:

                root_name = a.module.split('.')[0]
                if a.level == 0 and not is_std_lib(root_name):
                    modules.add(root_name)
                elif not is_installed_package(root_name):
                    if a.level == 0:
                        modules.union(self.recursive_module_dependencies(a.module))
                    else:
                        modules.union(self.recursive_module_dependencies('.'.join(module_name.split('.')[:-a.level])))

        return modules

    @property
    def module_dependencies(self):

        if self._module_dependencies is None:

            content = beam_path(inspect.getfile(type(self.obj))).read()
            ast_tree = ast.parse(content)
            module_name = self.module_spec.name

            modules = []
            for a in ast_tree.body:
                if type(a) is ast.Import:
                    for ai in a.names:
                        root_name = ai.name.split('.')[0]
                        if root_name != module_name:
                            modules.append(root_name)

                elif type(a) is ast.ImportFrom:

                    root_name = a.module.split('.')[0]
                    if root_name != module_name and a.level == 0:
                        modules.append(root_name)
            self._module_dependencies = list(set(modules))

        return self._module_dependencies

    @property
    def top_levels(self):

        if self._top_levels is None:
            top_levels = {}
            for i, dist in enumerate(pkg_resources.working_set):
                egg_info = beam_path(dist.egg_info)
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
                    warnings.warn(f"Could not find top level module for package: {project_name}")
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

    @classmethod
    def to_bundle(cls, module):
        return cls(module)

    def get_pip_package(self, module_name):
        return self.top_levels[module_name]
