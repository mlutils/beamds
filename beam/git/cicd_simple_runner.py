import os
import shutil
from beam import logger, resource
from ..path import beam_path
from .git_dataclasses import GitFilesConfig
from .config import ServeCICDConfig
from ..auto import AutoBeam
from .cicd import BeamCICD
from .git_resource import deploy_cicd


def copy_files_from_path(git_files, dest_dir):
    """
    Copies specified files to the destination directory. Logs a warning for missing files.

    :param git_files: List of file paths to copy.
    :param dest_dir: Destination directory to copy the files into.
    """
    cicd = BeamCICD()
    project = self.gitlab_client.projects.get(self.get_hparam('gitlab_project'))
    current_dir = beam_path(__file__).parent

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for f in git_files:
        if os.path.exists(f):
            shutil.copy2(f, dest_dir)
        else:
            logger.warning(f"File not found: {git_files}. Skipping.")



# Main runner function
def main():

    base_config = ServeCICDConfig()
    yaml_config = ServeCICDConfig(resource('/home/dayosupp/projects/beamds/examples/cicd_example.yaml').read())
    config = ServeCICDConfig(**{**base_config, **yaml_config})

    logger.info(f"Config: {config}")
    logger.info("Building python object from python function...")
    # # Replace these with dynamic inputs or CLI arguments as needed
    # entrypoint = "entrypoint.sh"
    # requirements = "requirements.txt"
    # base_image = "python:3.9-slim"
    # manager_url = "http://example.com/manager"

    conf = BeamCICDConfig(resource('/home/dayosupp/projects/beamds/examples/cicd_example.yaml').read())
    beam_cicd = BeamCICD(conf)

    # Step 1: Copy Git files to /app
    print("Copying Git files to /app...")
    copy_git_files(src_dir=".", dest_dir="/app")

    # Step 2: Build the Docker image
    image_name = build_docker_image(config)

    # Step 3: Use manager.py to launch serve cluster
    launch_manager(config)

if __name__ == "__main__":
    main()

# Build Docker image
def build_docker_image(config):
    print("Starting Docker build process...")
    try:
        image_name = AutoBeam.to_docker(
            base_image=config.base_image,
            serve_config={"entrypoint": config.entrypoint},
            bundle_path="/app",  # Assuming the copied files are in /app
            image_name="built-image",
            dockerfile="simple-entrypoint",
        )
        print(f"Successfully built Docker image: {image_name}")
        return image_name
    except Exception as e:
        print(f"Error during Docker build: {e}")
        raise

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
