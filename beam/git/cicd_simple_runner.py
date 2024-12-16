import os
import shutil
from pathlib import Path
from auto import AutoBeam  # Importing your existing AutoBeam class
from manager import resource

# Configuration class to store parameters
class Config:
    def __init__(self, entrypoint, requirements, base_image, manager_url):
        self.entrypoint = entrypoint
        self.requirements = requirements
        self.base_image = base_image
        self.manager_url = manager_url

# Copy Git files to /app directory
def copy_git_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dest_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

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

# Main runner function
def main():
    # Replace these with dynamic inputs or CLI arguments as needed
    entrypoint = "entrypoint.sh"
    requirements = "requirements.txt"
    base_image = "python:3.9-slim"
    manager_url = "http://example.com/manager"

    config = Config(entrypoint, requirements, base_image, manager_url)

    # Step 1: Copy Git files to /app
    print("Copying Git files to /app...")
    copy_git_files(src_dir=".", dest_dir="/app")

    # Step 2: Build the Docker image
    image_name = build_docker_image(config)

    # Step 3: Use manager.py to launch serve cluster
    launch_manager(config)

if __name__ == "__main__":
    main()
