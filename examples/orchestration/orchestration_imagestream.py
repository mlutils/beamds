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

# Test cases
test_cases = [
    {
        "description": "Valid container image with explicit registry (nginx)",
        "container_image": "docker.io/library/nginx:1.21",
        "overwrite_registry": None,
    },
    {
        "description": "Valid container image with overwrite to harbor (PostgreSQL)",
        "container_image": "docker.io/library/postgres:15.1",
        "overwrite_registry": "harbor.dt.local",
    },
    {
        "description": "Valid container image without registry with overwrite to harbor (Alpine)",
        "container_image": "alpine:3.18",
        "overwrite_registry": "harbor.dt.local",
    },
    {
        "description": "Container image without registry, default registry applied (Python)",
        "container_image": "python:3.11-slim",
        "overwrite_registry": None,
    },
    {
        "description": "Overwrite registry to OpenShift default (Grafana)",
        "container_image": "grafana/grafana:10.1.0",
        "overwrite_registry": "default-route-openshift-image-registry.apps.cluster.local",
    },
    {
        "description": "Invalid container image format (invalid format with extra colon)",
        "container_image": "nginx:latest:extra",
        "overwrite_registry": None,
    },
]

# Run test cases
for test_case in test_cases:
    print(f"\nRunning Test: {test_case['description']}")
    logger.info(f"Running Test: {test_case['description']}")
    try:
        namespace = config['project_name']
        container_image = test_case["container_image"]
        overwrite_registry = test_case["overwrite_registry"]

        # Call the method to create the ImageStream
        result = k8s.create_image_stream(namespace, container_image, overwrite_registry)

        # Log and print the result
        logger.info(f"Test Result: {result}")
        print(f"Test Result: {result}")
    except Exception as e:
        # Log and print any errors that occur
        logger.error(f"Test Failed: {test_case['description']} - {e}")
        print(f"Test Failed: {test_case['description']} - {e}")
