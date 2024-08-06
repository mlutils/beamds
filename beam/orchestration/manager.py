import kopf
import yaml
from kubernetes.client import ApiClient, Configuration
from openshift.dynamic import DynamicClient
import os

# Custom class imports
from beam.orchestration import (BeamK8S, BeamDeploy)
from beam.orchestration.config import K8SConfig


def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


script_dir = os.path.dirname(os.path.realpath(__file__))
conf_path = os.path.join(script_dir, '/home/dayosupp/projects/beamds/examples/orchestration_beamdeploy.yaml')
config = K8SConfig(load_config(conf_path))


def setup_custom_kubernetes_client():
    configuration = Configuration()
    configuration.host = config['api_url']
    configuration.api_key = {'authorization': f"Bearer {config['api_token']}"}
    configuration.verify_ssl = False
    return ApiClient(configuration=configuration)


# Instantiate and configure the global ApiClient
api_client = setup_custom_kubernetes_client()
Configuration.set_default(api_client.configuration)


class BeamK8SConfigured(BeamK8S):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.namespace = config.get('namespace', 'default')


@kopf.on.startup()
def configure_operator(**_):
    print("Custom Kubernetes client configured using specified API settings.")


@kopf.on.create('mydomain.com', 'v1', 'beamdeployments')
def handle_create(spec, **kwargs):
    beam_k8s = BeamK8SConfigured(config)
    project_name = spec.get('projectName', config.get('project_name'))
    beam_k8s.create_project(project_name)
    print(f"Project {project_name} created or verified.")


@kopf.on.update('mydomain.com', 'v1', 'beamdeployments')
def handle_update(spec, **kwargs):
    beam_k8s = BeamK8SConfigured(config)
    print("Update handled.")


@kopf.on.delete('mydomain.com', 'v1', 'beamdeployments')
def handle_delete(spec, **kwargs):
    beam_k8s = BeamK8SConfigured(config)
    project_name = spec.get('projectName', config.get('project_name'))
    beam_k8s.delete_project(project_name)
    print(f"Project {project_name} cleanup initiated.")


kopf.run()
