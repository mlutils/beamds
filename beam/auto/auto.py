import inspect
import ast
import sys
from collections import defaultdict

from packaging import version

from ..data import BeamData
from ..base import BeamBase, base_paths
from .utils import get_module_paths, ImportCollector, is_installed_package, is_std_lib, get_origin, is_module_installed
from ..path import beam_path, local_copy

import importlib.metadata

import os
import importlib
import warnings

from ..logging import beam_logger as logger
from ..type.utils import is_class_instance, is_function
from ..utils import cached_property
from uuid import uuid4 as uuid


class AutoBeam(BeamBase):

    # Blacklisted pip packages (sklearn is a fake project that should be ignored, scikit-learn is the real one)
    blacklisted_pip_package = ['sklearn']

    def __init__(self, obj, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._private_modules = None
        self._visited_modules = None
        self.obj = obj

    @cached_property
    def self_path(self):
        return beam_path(inspect.getfile(AutoBeam)).resolve()

    @cached_property
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
        if module_spec is not None and module_spec.origin != 'frozen':
            self._private_modules.append(module_spec)

    # @cached_property
    # def module_spec(self):
    #     try:
    #         module_spec = importlib.util.find_spec(type(self.obj).__module__)
    #     except ValueError:
    #         module_spec = None
    #
    #     root_module = module_spec.name.split('.')[0]
    #     return importlib.util.find_spec(root_module)

    @property
    def in_main_script(self):
        return type(self.obj).__module__ == '__main__'

    @cached_property
    def module_name(self):
        module_name = self._module_name
        if module_name == '__main__':
            # The object is defined in the __main__ script
            main_script_path = os.path.abspath(sys.argv[0])

            cwd = beam_path(os.getcwd())
            name = beam_path(main_script_path).relative_to(cwd)
            module_name = name.str.removesuffix('.py').replace(os.sep, '.')

        return module_name

    @property
    def _module_name(self):
        if is_class_instance(self.obj):
            return type(self.obj).__module__
        elif is_function(self.obj):
            return self.obj.__module__
        else:
            raise ValueError(f"Object type not supported: {type(self.obj)}")

    @cached_property
    def module_spec(self):

        module_name = self.module_name
        if module_name == '__main__':
            # The object is defined in the __main__ script
            main_script_path = os.path.abspath(sys.argv[0])

            module_spec = importlib.util.spec_from_file_location(self.module_name, main_script_path)
            return module_spec
        else:
            # The object is defined in a regular module
            module_spec = importlib.util.find_spec(module_name)
            root_module = module_spec.name.split('.')[0]
            return importlib.util.find_spec(root_module)

    @staticmethod
    def module_walk(root_path):

        root_path = beam_path(root_path).resolve()
        module_walk = {}

        for r, dirs, files in root_path.walk():

            r_relative = r.relative_to(root_path)
            dir_files = {}
            for f in files:
                p = r.joinpath(f)

                # TODO: better filter of undesired files
                if p.suffix not in ['.pyc', '.pyo', '.pyd', '.so']:
                    dir_files[f] = p.read()

            if len(dir_files):
                module_walk[str(r_relative)] = dir_files

        return module_walk
    # TODO: we need all files to be exist in the manager, else the docker builds will fail, not only .py files

    @cached_property
    def private_modules_walk(self):

        private_modules_walk = {}
        root_paths = set(sum([get_module_paths(m) for m in self.private_modules if m is not None], []))
        for root_path in root_paths:
            # todo: fix it
            root_path = beam_path(root_path)
            if root_path.is_file():
                private_modules_walk[root_path.str] = root_path.read()
            else:
                private_modules_walk[root_path.str] = self.module_walk(root_path.str)

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

                root_name = a.module.split('.')[0] if a.module else ''

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

                        a_module = a.module if a.module else ''
                        path = path.joinpath(f"{a_module.replace('.', os.sep)}")
                        if path.is_dir():
                            path = path.joinpath('__init__.py')
                        else:
                            path = path.with_suffix('.py')

                        internal_modules = self.recursive_module_dependencies(path)
                        modules = modules.union(internal_modules)

        return modules

    @cached_property
    def module_dependencies(self):
        if is_class_instance(self.obj):
            module_path = beam_path(inspect.getfile(type(self.obj))).resolve()
        elif is_function(self.obj):
            module_path = beam_path(inspect.getfile(self.obj)).resolve()
        else:
            raise ValueError(f"Object type not supported: {type(self.obj)}")

        modules = self.recursive_module_dependencies(module_path)
        return list(set(modules))

    @cached_property
    def top_levels(self):

        top_levels = {}
        for i, dist in enumerate(importlib.metadata.distributions()):

            egg_info = dist._path
            project_name = dist.metadata.get('Name', None)

            if project_name is None:
                logger.warning(f"Could not find project name for distribution: {egg_info}, skipping.")
                continue

            if egg_info is None:
                logger.warning(f"Could not find egg info for package: {project_name}, skipping.")
                continue

            egg_info = beam_path(egg_info).resolve()
            tp_file = egg_info.joinpath('top_level.txt')
            module_name = None

            if egg_info.parent.joinpath(project_name).is_dir():
                module_name = project_name
            elif egg_info.parent.joinpath(f"{project_name.replace('-', '_')}.py").is_file():
                module_name = project_name.replace('-', '_')
            elif egg_info.parent.joinpath(project_name.replace('-', '_')).is_dir():
                module_name = project_name.replace('-', '_')
            elif egg_info.joinpath('RECORD').is_file():

                record = egg_info.joinpath('RECORD').read(ext='.txt', readlines=True)
                module_name = []
                for line in record:
                    if '__init__.py' in line:
                        module_name.append(line.split('/')[0])
                if not len(module_name):
                    module_name = None

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
            elif not module_name:
                warnings.warn(f"{project_name}: is empty")
            else:
                if type(module_name) is not list:
                    module_name = [module_name]

                for module_name_i in module_name:
                    if module_name_i in top_levels:
                        if type(top_levels[module_name_i]) is list:
                            v = top_levels[module_name_i]
                        else:
                            v = [top_levels[module_name_i]]
                            v.append(dist)
                        top_levels[module_name_i] = v
                    else:
                        top_levels[module_name_i] = dist

        return top_levels

    @property
    def import_statement(self):
        return f"from {self.module_name} import {self.canonical_name()}"

    def canonical_name(self, look_for_property=False):
        if is_class_instance(self.obj):
            if look_for_property and hasattr(self.obj, 'name'):
                obj_name = self.obj.name
            else:
                obj_name = type(self.obj).__name__
        elif is_function(self.obj):
            obj_name = self.obj.__name__
        else:
            raise ValueError(f"Object type not supported: {type(self.obj)}")
        return obj_name

    @property
    def metadata(self):

        name = self.canonical_name(look_for_property=True)

        # # in case the object is defined in the __main__ script
        # # get all import statements from the script
        main_import_statements = None
        if self.in_main_script:
            main_script_path = os.path.abspath(sys.argv[0])
            main_script_path = beam_path(main_script_path)
            content = main_script_path.read()
            ast_tree = ast.parse(content)
            collector = ImportCollector()
            collector.visit(ast_tree)
            import_nodes = collector.import_nodes
            main_import_statements = '\n'.join([ast.unparse(a) for a in import_nodes])

        return {'name': name, 'type': type(self.obj).__name__,
                'import_statement': self.import_statement, 'module_name': self.module_name,
                'main_import_statements': main_import_statements}

    @staticmethod
    def static_bundle(requirements_file=None, root_path=None):
        return AutoBeam.to_bundle(obj=None, path=None, requirements_file=requirements_file, root_path=root_path,
                                    add_metadata=False, add_state=False)

    @staticmethod
    def to_bundle(obj=None, path=None, blacklist=None, requirements_file=None,
                  root_path=None, add_metadata=True, add_state=True):

        if path is None:
            path = beam_path('.')
            if hasattr(obj, 'name'):
                path = path.joinpath(obj.name)
            else:
                path = path.joinpath('beam_bundle')
        else:
            path = beam_path(path)

        path = path.resolve()

        ab = AutoBeam(registry_project_name=None, registry_url=None, obj=obj)
        path.clean()
        path.mkdir()
        logger.info(f"Saving object's files to path {path}: [requirements.json, modules.tar.gz, state, requierements.txt]")
        if requirements_file is None:
            path.joinpath('requirements.json').write(ab.requirements)
            ab.write_requirements(ab.requirements, path.joinpath('requirements.txt'), blacklist=blacklist)
        else:
            beam_path(requirements_file).copy(path.joinpath('requirements.json'))
        if root_path is None:
            ab.modules_to_tar(path.joinpath('modules.tar.gz'))
        else:
            ab.root_path_to_tar(root_path, path.joinpath('modules.tar.gz'))

        if add_metadata:
            path.joinpath('metadata.json').write(ab.metadata)
            logger.info(f"Contents of {path}/requirements.txt: {open(f'{path}/requirements.txt').read()}")

        if add_state:
            blacklist_priority = None
            if ab.in_main_script:
                blacklist_priority = ['.pkl']
            BeamData.write_object(obj, path.joinpath('state'), blacklist_priority=blacklist_priority)

        return path

    @classmethod
    def from_bundle(cls, path, cache_path=None):

        logger.info(f"Loading object from path {path}")
        if cache_path is None:
            cache_path = beam_path(base_paths.autobeam_cache).joinpath(uuid())
        else:
            cache_path = beam_path(cache_path)

        import tarfile
        path = beam_path(path).resolve()

        # 1. Check necessary files
        req_file = path.joinpath('requirements.json')
        modules_tar = path.joinpath('modules.tar.gz')
        metadata_file = path.joinpath('metadata.json')

        def load_obj():

            # 3. Extract the Python modules
            cache_path.mkdir(parents=True, exist_ok=True)
            with tarfile.open(modules_tar, "r:gz") as tar:
                tar.extractall(str(cache_path))

            # 4. Add the directory containing the extracted Python modules to sys.path
            sys.path.insert(0, str(cache_path))

            # 5. Load metadata and import necessary modules
            metadata = metadata_file.read()

            if metadata['main_import_statements'] is not None:
                exec(metadata['main_import_statements'], globals())

            imported_class = metadata['type']
            module = importlib.import_module(metadata['module_name'])

            if imported_class != 'function':
                obj = getattr(module, imported_class)
            else:
                obj = module

            # import_statement = metadata['import_statement']
            # exec(import_statement, globals())
            # cls_obj = globals()[imported_class]

            # 7. Construct the object from its state using a hypothetical from_state method
            _obj = BeamData.read(path.joinpath('state'))

            return _obj

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

    @cached_property
    def requirements(self):

        versions = defaultdict(lambda: version.parse('0.0.0'))
        requirements = {}

        for module_name in self.module_dependencies:
            pip_package = self.get_pip_package(module_name)

            if pip_package is not None:
                if type(pip_package) is not list:
                    pip_package = [pip_package]
                for pp in pip_package:

                    name = pp.metadata['Name']
                    name = name.lower().replace('_', '-')

                    if name is AutoBeam.blacklisted_pip_package:
                        continue

                    v = version.parse(pp.version)
                    if v <= versions[name]:
                        continue
                    else:
                        versions[name] = v

                    requirements[name] = {'pip_package': name, 'module_name': module_name, 'version': str(v),
                                          'metadata': {k: str(v) for k, v in pp.metadata.items()}}
            else:
                logger.warning(f"Could not find pip package for module: {module_name}")

        requirements = [requirements[k] for k in sorted(requirements.keys())]

        return requirements

    @staticmethod
    def write_requirements(requirements, path, relation='~=', sim_type='major', blacklist=None):
        '''

        @param requirements:
        @param path:
        @param relation: can be '~=', '==' or '>=' or 'all'
        @return:
        '''

        if blacklist is None:
            blacklist = []

        path = beam_path(path)
        if relation == 'all':
            content = '\n'.join([f"{r['pip_package']}" for r in requirements])
        elif relation in ['==', '>=']:
            content = '\n'.join([f"{r['pip_package']}{relation}{r['version']}" for r in requirements
                                 if r['pip_package'] not in blacklist])
        elif relation == '~=':
            if sim_type == 'major':
                content = '\n'.join(
                    [f"{r['pip_package']}{relation}{'.'.join(r['version'].split('.')[:2])}" for r in requirements
                     if r['pip_package'] not in blacklist])
            elif sim_type == 'minor':
                content = '\n'.join(
                    [f"{r['pip_package']}{relation}{'.'.join(r['version'].split('.')[:3])}" for r in requirements
                     if r['pip_package'] not in blacklist])
            else:
                raise ValueError(f"sim_type can be 'major' or 'minor'")
        else:
            raise ValueError(f"relation can be '~=', '==' or '>=' or 'all'")

        content = f"{content}\n"
        path.write(content, ext='.txt')

    def modules_to_tar(self, path):

        """
        This method is used to create a tarball of all the private modules used by the object.

        Parameters:
        path (str): The path where the tarball will be created.

        Returns:
        None

        """

        path = beam_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import tarfile
        with local_copy(path, override=True) as local_path:
            with tarfile.open(local_path, "w:gz") as tar:
                for i, (root_path, sub_paths) in enumerate(self.private_modules_walk.items()):
                    root_path = beam_path(root_path)
                    if root_path.is_file():
                        tar.add(str(root_path), arcname=root_path.name)
                    else:
                        root_path = beam_path(root_path)
                        for sub_path, files in sub_paths.items():
                            for file_name, _ in files.items():
                                local_name = root_path.joinpath(sub_path, file_name)
                                relative_name = local_name.relative_to(root_path.parent)
                                tar.add(str(local_name), arcname=str(relative_name))

    @staticmethod
    def root_path_to_tar(root_path, path):

        """
        This method is used to create a tarball of the root path.

        Parameters:
        root_path (str): The root path to be archived.
        path (str): The path where the tarball will be created.

        Returns:
        None

        """
        path = beam_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        root_path = beam_path(root_path).resolve()
        if not root_path.is_dir():
            raise ValueError(f"Root path must be a directory: {root_path}")

        import tarfile
        with local_copy(path, override=True) as local_path:
            with tarfile.open(local_path, "w:gz") as tar:
                for _, dirs, files in root_path.walk():
                    for f in files:
                        file_path = root_path.joinpath(f)
                        if file_path.is_file():
                            relative_name = file_path.relative_to(root_path.parent)
                            tar.add(str(file_path), arcname=str(relative_name))
                        else:
                            logger.warning(f"Skipping non-file: {file_path}")

    @staticmethod
    def to_docker(obj=None, base_image=None, serve_config=None, bundle_path=None, image_name=None,
                  entrypoint='synchronous-server', beam_version='latest', beam_ds_path=None, dockerfile='simple-entrypoint',
                  registry_url=None, base_url=None, registry_project_name=None, path_to_state=None,
                  username=None, password=None, copy_bundle=False, requirements_blacklist=None,
                  **kwargs):

        if obj is not None:
            logger.info(f"Building an object bundle")
            bundle_path = AutoBeam.to_bundle(obj, path=path_to_state, blacklist=requirements_blacklist)

        logger.info(f"Building a Docker image with the requirements and the object bundle. Base image: {base_image}")
        full_image_name = (
            AutoBeam._build_image(bundle_path, base_image, config=serve_config, image_name=image_name,
                                  entrypoint=entrypoint, beam_version=beam_version, beam_ds_path=beam_ds_path, username=username,
                                  password=password, base_url=base_url, registry_project_name=registry_project_name,
                                  registry_url=registry_url, copy_bundle=copy_bundle, dockerfile=dockerfile))
        logger.info(f"full_image_name: {full_image_name}")
        return full_image_name

    @staticmethod
    def _build_image(bundle_path, base_image=None, config=None, image_name=None, entrypoint=None,
                     copy_bundle=False, registry_url=None, username=None, password=None, override_image=False,
                     beam_version='latest', beam_ds_path=None, base_url=None, registry_project_name=None, dockerfile=None):

        assert base_image is not None, "You must provide a base_image."
        if not bool(beam_version):
            beam_version = ''

        import docker
        from docker.errors import BuildError, ImageNotFound, APIError

        # client = docker.APIClient()
        # client = docker.APIClient(base_url='unix:///var/run/docker.sock')
        client = docker.APIClient(base_url=base_url)
        # client = docker.APIClient(base_url='unix:///home/beam/.docker/run/docker.sock')
        # client = docker.APIClient(base_url='unix:////home/beam/runtime/docker.sock')

        try:
            # Try to get the base image locally, if not found, pull from registry
            try:
                client.inspect_image(base_image)
            except ImageNotFound:
                print(f"Base image {base_image} not found locally. Attempting to pull...")
                client.pull(base_image)
                print(f"Base image {base_image} pulled successfully.")
        finally:
            client.close()

        entrypoint = entrypoint or 'synchronous-server'
        dockerfile = dockerfile or 'simple-entrypoint'

        bundle_path = beam_path(bundle_path)
        # current_dir = this_dir()
        current_dir = beam_path(__file__).parent

        if image_name is None:
            image_name = f"autobeam-{bundle_path.name}-{base_image}"

        with local_copy(bundle_path, as_beam_path=True, disable=not copy_bundle, override=True) as bundle_path:

            docker_dir = bundle_path.joinpath('.docker')
            docker_dir.clean()
            docker_dir.mkdir()

            config = dict(config) if config else {}
            docker_dir.joinpath('config.yaml').write(config)


            entrypoint = beam_path(entrypoint)
            if entrypoint.is_file() and not entrypoint.suffix:
                source_entrypoint = entrypoint.read()
            else:
                source_entrypoint = current_dir.joinpath(f'{entrypoint}.py').read()

            entrypoint = docker_dir.joinpath('entrypoint.py')
            entrypoint.write(source_entrypoint)

            dockerfile = beam_path(dockerfile)
            if dockerfile.is_file():
                source_dockerfile = dockerfile.read(ext='.txt')
            else:
                source_dockerfile = current_dir.joinpath(f"dockerfile-{dockerfile}").read(ext='.txt')

            docker_tools_dir = current_dir.joinpath('docker-tools')

            docker_dir.joinpath('dockerfile').write(source_dockerfile, ext='.txt')

            beam_ds_path = beam_path(beam_ds_path)
            target_beam_ds_path = None
            if beam_ds_path is None or not beam_ds_path.is_file():
                logger.warning(f"Beam-DS path is invalid or file not found: {beam_ds_path}, requires pip installation")
            else:
                # Copy the Beam-DS file into the .docker directory
                target_beam_ds_path = docker_tools_dir.joinpath(beam_ds_path.name)
                target_beam_ds_path.parent.mkdir(parents=True, exist_ok=True)
                target_beam_ds_path.write_bytes(beam_ds_path.read_bytes())

                logger.info(f"Beam-DS file copied to: {target_beam_ds_path}")

            # Log the current directory and its contents
            logger.info(f"Contents of the specified directory Before copy '{docker_tools_dir}':")
            for file_name in docker_tools_dir.iterdir():  # Utilize Path.iterdir() for simplicity
                if file_name.is_dir():
                    logger.info(f"Dir: {file_name.name}")
                elif file_name.is_file():
                    logger.info(f"File: {file_name.name}")

            docker_tools_dir.copy(docker_dir.joinpath('docker-tools'))

            # Log the contents of the new docker-tools directory
            logger.info(f"Contents of the copied directory '{docker_dir.joinpath('docker-tools')}':")
            for file_name in docker_dir.joinpath('docker-tools').iterdir():  # Use Path.iterdir()
                if file_name.is_dir():
                    logger.info(f"Dir: {file_name.name}")
                elif file_name.is_file():
                    logger.info(f"File: {file_name.name}")

            # Define build arguments
            build_args = {
                'BASE_IMAGE': base_image,
                'REQUIREMENTS_FILE': 'requirements.txt',
                'ALGORITHM_DIR': bundle_path.relative_to(bundle_path).str,
                'ENTRYPOINT_SCRIPT': entrypoint.relative_to(bundle_path).str,
                'CONFIG_FILE': '.docker/config.yaml',
                'BEAM_DS_VERSION': beam_version,
                'BEAM_DS_PATH': target_beam_ds_path,
                # 'BEAM_DS_PATH': beam_ds_path.name,
                'DOCKER_TOOLS_DIR': '.docker/docker-tools',
            }

            try:

                if not override_image:
                    from uuid import uuid4 as uuid
                    random_string = uuid().hex[:6]
                    image_name = f"{image_name}-{random_string}"

                client = docker.APIClient(base_url=base_url)
                logger.debug(f"Docker client version: {client.version()}")
                with local_copy(bundle_path) as local_path:
                    response = client.build(path=local_path, dockerfile='.docker/dockerfile',
                                            buildargs=build_args, tag=image_name, rm=True, decode=True)

                # Process and print each log entry
                for line in response:
                    if 'stream' in line:
                        print(line['stream'].strip())

                if registry_url is not None:
                    image_name = AutoBeam._push_image(image_name, registry_url, base_url=base_url,
                                                           registry_project_name=registry_project_name,
                                                           username=username, password=password)
                logger.info(f"Full image name: {image_name}")
                return image_name

            except BuildError as e:
                logger.error(f"Error building Docker image: {e}")
                raise e
            except Exception as e:
                logger.error(f"Error building Docker image: {e}")
                raise e
            finally:
                client.close()

    @staticmethod
    def _push_image(image_name, registry_url, username=None, password=None,
                    registry_project_name=None, dockercfg_path=None, base_url=None):
        import docker
        from docker.errors import APIError

        # Default docker configuration path
        if dockercfg_path is None:
            dockercfg_path = base_paths.docker_config_dir

        # Set up Docker client
        client = docker.APIClient(base_url=base_url)

        # Extract registry name from URL, remove the protocol for tagging purposes
        if '://' in registry_url:
            _, registry_name = registry_url.split('://', 1)
        else:
            registry_name = registry_url

        # Ensure the registry name does not end with a slash for consistency
        registry_name = registry_name.rstrip('/')

        if ':' not in image_name:
            image_name += ':latest'

        # Construct the full image name including the project_name
        # full_image_name = f"{registry_name}/{registry_project_name}/{image_name}"
        repository = f"{registry_name}/{registry_project_name}"
        full_image_name = f"{repository}/{image_name}"
        try:
            # Tag the image to include the registry path
            # if client.tag(image_name, full_image_name):
            if client.tag(image_name, full_image_name):
                logger.info(f"Successfully tagged {image_name} as {full_image_name}")

            # Log into the registry if credentials are provided
            # TODO: Add True/False flag use or not use credentials with the registry
            if username and password:
                login_response = client.login(username=username, password=password, registry=registry_url,
                                              dockercfg_path=dockercfg_path, reauth=True)
                logger.info(f"Login response: {login_response}")

            # Push the image to the registry
            # response = client.push(full_image_name, stream=True, decode=True, insecure_registry=insecure_registry)
            response = client.push(full_image_name, stream=True, decode=True)
            for line in response:
                if 'status' in line:
                    logger.debug(line['status'])
                elif 'error' in line:
                    logger.error(f"Error during push: {line['error']}")
                    raise APIError(line['error'])
                elif 'progress' in line:
                    logger.debug(line['progress'])
            return full_image_name  # Return the full image name on success

        except APIError as e:
            print(f"Error pushing Docker image: {e}")
            raise e
        finally:
            client.close()
