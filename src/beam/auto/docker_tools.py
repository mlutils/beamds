import docker
from docker.errors import BuildError

# Initialize Docker client
client = docker.from_env()

# Define build arguments
build_args = {
    'BASE_IMAGE': 'python:3.8-slim',
    'REQUIREMENTS_FILE': './path/to/requirements.txt',
    'ALGORITHM_DIR': './path/to/algorithm',
    'ENTRYPOINT_SCRIPT': './path/to/script.py'
}

# Path to the directory containing the Dockerfile
path_to_dockerfile = '.'

try:
    # Build the image
    image, build_logs = client.images.build(path=path_to_dockerfile, buildargs=build_args, tag='your-image-name')

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