import os
import shutil
from beam import logger, resource
from ..path import beam_path
from .git_dataclasses import GitFilesConfig
from .config import ServeCICDConfig
from ..auto import AutoBeam
from .cicd_client import BeamCICDClient
from .git_resource import deploy_cicd


# Main runner function
def main():

    base_config = ServeCICDConfig()
    yaml_config = ServeCICDConfig(resource('/home/dayosupp/projects/beamds/examples/cicd_example_pdf_extractor.yaml').read())
    config = ServeCICDConfig(**{**base_config, **yaml_config})
    logger.info(f"Config: {config}")
    # Step 3: Use manager.py to launch serve cluster



    launch_manager(config)

if __name__ == "__main__":
    main()


# Launch serve cluster using manager.py
def launch_manager(config):
    print("Launching serve cluster via manager...")
    try:
        manager = resource(config.manager_url)
        manager.launch_serve_cluster(config)
        print("Serve cluster launched successfully.")
    except Exception as e:
        print(f"Error launching serve cluster: {e}")
        raise
