import inspect
import ast
import sys

from ..core import Processor
from .utils import get_module_paths, ImportCollector, is_installed_package, is_std_lib, get_origin, is_module_installed
from ..utils import lazy_property
from ..path import beam_path

import importlib.metadata

import pkg_resources
import os
import importlib
import warnings

from ..logger import beam_logger as logger
from uuid import uuid4 as uuid


class AutoBeam(Processor):

    # Blacklisted pip packages (sklearn is a fake project that should be ignored, scikit-learn is the real one)
    blacklisted_pip_package = ['sklearn']

    def __init__(self, obj):
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

    @lazy_property
    def module_spec(self):
        module_spec = importlib.util.find_spec(type(self.obj).__module__)
        root_module = module_spec.name.split('.')[0]
        return  importlib.util.find_spec(root_module)

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

    @lazy_property
    def private_modules_walk(self):

        private_modules_walk = {}
        root_paths = set(sum([get_module_paths(m) for m in self.private_modules if m is not None], []))
        for root_path in root_paths:
            private_modules_walk[root_path] = self.module_walk(root_path)

        return private_modules_walk

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

    @lazy_property
    def module_dependencies(self):
        module_path = beam_path(inspect.getfile(type(self.obj))).resolve()
        modules = self.recursive_module_dependencies(module_path)
        return list(set(modules))


    @lazy_property
    def top_levels(self):

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

        return top_levels

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
        return {'name': self.obj.name, 'type': type(self.obj).__name__,
                'import_statement': self.import_statement, 'module_name': type(self.obj).__module__}

    @staticmethod
    def to_bundle(obj, path=None):

        if path is None:
            path = beam_path('..')
            if hasattr(obj, 'name'):
                path = path.joinpath(obj.name)
            else:
                path = path.joinpath('beam_bundle')
        else:
            path = beam_path(path)

        path = path.resolve()

        ab = AutoBeam(obj)
        path.clean()
        path.mkdir()
        logger.info(f"Saving object's files to path {path}: [requirements.json, modules.tar.gz, state, requierements.txt]")
        path.joinpath('requirements.json').write(ab.requirements)
        ab.write_requirements(ab.requirements, path.joinpath('requirements.txt'))
        ab.modules_to_tar(path.joinpath('modules.tar.gz'))
        path.joinpath('metadata.json').write(ab.metadata)
        if hasattr(obj, 'save_state'):
            obj.save_state(path.joinpath('state'))
        else:
            logger.warning(f"Object {obj} does not have a save_state method.")
        try:
            path.joinpath('skeleton.pkl').write(obj)
        except Exception as e:
            path.joinpath('skeleton.pkl').unlink()
            logger.warning(f"Could not pickle object: {obj} ({e}), saving only the state.")

        if not path.joinpath('skeleton.pkl').is_file() and \
                not path.joinpath('state').exists():
            logger.error(f"Could not save object {obj} to path {path}. "
                         f"Make sure the object has a save_state method and/or "
                         f"it is pickleable.")

        return path

    @classmethod
    def from_bundle(cls, path, cache_path=None):

        logger.info(f"Loading object from path {path}")
        if cache_path is None:
            cache_path = beam_path('/tmp/autobeam').joinpath(uuid())
        else:
            cache_path = beam_path(cache_path)

        import tarfile
        path = beam_path(path).resolve()

        # 1. Check necessary files
        req_file = path.joinpath('requirements.json')
        modules_tar = path.joinpath('modules.tar.gz')
        state_file = path.joinpath('state')
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

            imported_class = metadata['type']
            module = importlib.import_module(metadata['module_name'])
            cls_obj = getattr(module, imported_class)

            # import_statement = metadata['import_statement']
            # exec(import_statement, globals())
            # cls_obj = globals()[imported_class]

            # 7. Construct the object from its state using a hypothetical from_state method
            if skeleton_file.is_file():
                try:
                    obj = skeleton_file.read()
                    if state_file.exists:
                        obj.load_state(state_file)
                except:
                    obj = cls_obj.from_state_path(state_file)
            else:
                obj = cls_obj.from_state_path(state_file)

            return obj

        try:
            obj = load_obj()
        except ImportError:
            logger.error(f"ImportError, some of the packages are not installed. "
                         f"Trying to install only the missing requirements.")
            # 2. Install necessary packages
            requirements = req_file.read()
            for r in requirements:
                if not is_module_installed(r['module_name']):
                    os.system(f"pip install {r['pip_package']}=={r['version']}")
            try:
                obj = load_obj()
            except Exception as e:
                logger.error(f"Exception: {e}. Trying to install all requirements.")
                all_reqs = ' '.join([f"{r['pip_package']}=={r['version']}" for r in requirements])
                os.system(f"pip install {all_reqs}")
                obj = load_obj()

        return obj

    def get_pip_package(self, module_name):

        if module_name not in self.top_levels:
            return None
        return self.top_levels[module_name]

    @lazy_property
    def requirements(self):
        requirements = []
        for module_name in self.module_dependencies:
            pip_package = self.get_pip_package(module_name)
            if pip_package is not None:
                if type(pip_package) is not list:
                    pip_package = [pip_package]
                for pp in pip_package:
                    if pp.project_name is AutoBeam.blacklisted_pip_package:
                        continue
                    requirements.append({'pip_package': pp.project_name, 'module_name': module_name,
                                         'version': pp.version
                                         })
            else:
                logger.warning(f"Could not find pip package for module: {module_name}")
        return requirements

    @staticmethod
    def write_requirements(requirements, path, relation='~=', sim_type='major'):
        '''

        @param requirements:
        @param path:
        @param relation: can be '~=', '==' or '>=' or 'all'
        @return:
        '''

        path = beam_path(path)
        if relation == 'all':
            content = '\n'.join([f"{r['pip_package']}" for r in requirements])
        elif relation in ['==', '>=']:
            content = '\n'.join([f"{r['pip_package']}{relation}{r['version']}" for r in requirements])
        elif relation == '~=':
            if sim_type == 'major':
                content = '\n'.join([f"{r['pip_package']}{relation}{'.'.join(r['version'].split('.'))[:2]}" for r in requirements])
            elif sim_type == 'minor':
                content = '\n'.join([f"{r['pip_package']}{relation}{'.'.join(r['version'].split('.'))[:3]}" for r in requirements])
            else:
                raise ValueError(f"sim_type can be 'major' or 'minor'")
        else:
            raise ValueError(f"relation can be '~=', '==' or '>=' or 'all'")

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

    @staticmethod
    def _build_image(image_name, base_image, entry_point, path):
        path = beam_path(path)
        import docker
        from docker.errors import BuildError
        # Initialize Docker client
        client = docker.from_env()

        # Define build arguments
        build_args = {
            'BASE_IMAGE': base_image,
            'REQUIREMENTS_FILE': path.joinpath('requirements.txt').str,
            'ALGORITHM_DIR': path.str,
            'ENTRYPOINT_SCRIPT': entry_point
        }

        # Path to the directory containing the Dockerfile
        path_to_dockerfile = beam_path(__file__).parent.joinpath('Dockerfile').str

        try:
            # Build the image
            image, build_logs = client.images.build(path=path_to_dockerfile, buildargs=build_args,
                                                    tag=image_name)

            # Print build logs (optional)
            for line in build_logs:
                if 'stream' in line:
                    print(line['stream'].strip())

        except BuildError as e:
            print("Error building Docker image:", e)
        except Exception as e:
            print("Error:", e)
        finally:
            client.close()
