from beam.orchestration import BeamK8S, BeamPod, K8SConfig
from beam.resources import resource
from beam.logging import beam_logger as logger
import docker
import os

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


def commit_pod_to_image_and_push(new_image="harbor.dt.local/public/fake-alg-http-server:test1"):
    """
    Update the image of all containers in the specified deployment to the new image and push it to a Docker registry.

    Args:
        deployment_name (str): The name of the deployment to update.
        new_image (str): The new image name to use, including the registry path.
        registry_url (str): URL of the Docker registry.
        username (str): Username for Docker registry authentication.
        password (str): Password for Docker registry authentication.

    Returns:
        bool: True if the update and push were successful, False otherwise.
    """
    try:

        # Update the Kubernetes deployment
        deployment_name = config['deployment_name']
        deployment = k8s.apps_v1_api.read_namespaced_deployment(deployment_name, config['project_name'])
        for container in deployment.spec.template.spec.containers:
            container.image = new_image
        k8s.apps_v1_api.patch_namespaced_deployment(deployment_name, config['project_name'], deployment)
        logger.info(f"Deployment {deployment_name} in {config['project_name']} updated to image {new_image}")

        return True
    except Exception as e:
        logger.error(f"Failed to update deployment or push image: {e}")
        return False


commit_pod_to_image_and_push()
