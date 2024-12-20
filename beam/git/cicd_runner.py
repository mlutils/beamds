from beam import logger, resource
from beam.auto import AutoBeam
from beam.git import ServeCICDConfig
from beam.orchestration import ServeClusterConfig
import importlib
import sys
import os

import importlib.util


def import_module_from_path(module_path):
    # Create a module specification from the file location
    name = resource(module_path).stem
    spec = importlib.util.spec_from_file_location(name, module_path)

    if not spec:
        raise ImportError(f"Cannot find module at path: {module_path}")

    # Create a new module based on the specification
    module = importlib.util.module_from_spec(spec)

    # Execute the module in its own namespace
    loader = spec.loader
    if loader and hasattr(loader, 'exec_module'):
        loader.exec_module(module)
    else:
        raise ImportError(f"Cannot load module from path: {module_path}")

    return module

# the working directory of the runner is the git repository root

def main():

    base_config = ServeCICDConfig()
    yaml_config = ServeCICDConfig(resource('/home/dayosupp/projects/beamds/examples/cicd_example_yolo.yaml').read())
    config = ServeCICDConfig(**{**base_config, **yaml_config})

    logger.info(f"Config: {config}")
    logger.info("Building python object from python function...")

    python_file = config.python_file
    python_function = config.python_function

    module = import_module_from_path(python_file)
    function = getattr(module, python_function)

    obj = function()

    logger.info("Building bundle from python object...")
    AutoBeam.to_bundle(obj, config.path_to_state)
    logger.info("Deploying bundle via manager...")
    config.update(hparams={'alg': config.path_to_state})
    # config.update(alg=config.path_to_state)

    manager = resource(config.manager_url)
    manager.launch_serve_cluster(config)

    logger.info('Done!')


if __name__ == '__main__':
    main()