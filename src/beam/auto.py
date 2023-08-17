import inspect
import importlib
import ast
import os

from .tabular import DeepTabularAlg
from .path import beam_path

import pkg_resources
import os
import importlib
from pathlib import Path
import warnings


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
            module_walk = {}

            root_path = beam_path(self.module_spec.origin).parent
            for r, dirs, files in root_path.walk():

                r_relative = r.relative_to(root_path)
                dir_files = {}
                for f in files:
                    p = r.joinpath(f)
                    if p.suffix == '.py':
                        dir_files[f] = p.read()
                if len(dir_files):
                    module_walk[r_relative] = dir_files

            self._module_walk = module_walk
        return self._module_walk

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
                    if root_name != module_name:
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
