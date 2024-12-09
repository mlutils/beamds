# This is an example of how to use the BeamDeploy class to deploy a container to an OpenShift cluster.
from beam.orchestration import (BeamK8S, BeamPod, BeamDeploy, SecurityContextConfig, MemoryStorageConfig,
                                ServiceConfig, StorageConfig, RayPortsConfig, UserIdmConfig, CommandConfig)
from beam.logging import beam_logger as logger
import time
from beam.resources import resource, this_dir
from beam.orchestration.config import K8SConfig
import os
import re


script_dir = os.path.dirname(os.path.realpath(__file__))
conf_path = resource(os.path.join(script_dir, 'orchestration_beamdeploy.yaml')).str
config = K8SConfig(conf_path)


print('hello world')
print("API URL:", config['api_url'])
print("API Token:", config['api_token'])


# the order of the VARS is important!! (see BeamK8S class)
k8s = BeamK8S(
    api_url=config['api_url'],
    api_token=config['api_token'],
    project_name=config['project_name'],
    namespace=config['project_name'],
)

try:
    # Parameters for the ImageStream
    namespace = config['project_name']  # Use the project namespace from config
    container_image = "docker.io/library/nginx:latest"

    # Call the method to create the ImageStream
    image_stream = k8s.create_image_stream(namespace, container_image)

    # Log and print the result
    logger.info(f"Test Passed: ImageStream '{image_stream}'")

except Exception as e:
    # Log and print any errors that occur
    logger.error(f"Test Failed: {e}")
    print(f"Test Failed: {e}")


