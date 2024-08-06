import kopf
import yaml
from kubernetes.client import ApiClient, Configuration
from openshift.dynamic import DynamicClient
import os

# Custom class imports
from beam.orchestration import BeamK8S, BeamDeploy
from beam.orchestration.config import K8SConfig



def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

script_dir = os.path.dirname(os.path.realpath(__file__))
conf_path = os.path.join(script_dir, '/home/dayosupp/projects/beamds/examples/orchestration_beamdeploy.yaml')
config_data = load_config(conf_path)

beam_conf_path = os.path.join(script_dir, '/home/dayosupp/projects/beamds/examples/orchestration_beamdeploy.yaml')
beam_config = K8SConfig(load_config(beam_conf_path))

def setup_custom_kubernetes_client():
    configuration = Configuration()
    configuration.host = config_data['kopf_server']
    configuration.verify_ssl = False
    api_client = ApiClient(configuration=configuration)
    return DynamicClient(api_client)

# Instantiate and configure the global ApiClient
dynamic_client = setup_custom_kubernetes_client()

class BeamK8SConfigured(BeamK8S):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.namespace = config.get('namespace', 'default')

@kopf.on.startup()
def configure_operator(**_):
    print("Custom Kubernetes client configured using specified API settings.")
    @kopf.on.login()
    def login_fn(**_):
        return kopf.ConnectionInfo(
            server = config_data['api_url'],
            token = config_data['api_token'],
            insecure = True
        )

@kopf.on.create('mydomain.com', 'v1', 'beamdeployments')
def handle_create(spec, **kwargs):
    beam_k8s = BeamK8SConfigured(beam_config)
    project_name = spec.get('projectName', beam_config.get('project_name'))
    beam_k8s.create_project(project_name)
    print(f"Project {project_name} created or verified.")

@kopf.on.update('mydomain.com', 'v1', 'beamdeployments')
def handle_update(spec, **kwargs):
    beam_k8s = BeamK8SConfigured(beam_config)
    print("Update handled.")

@kopf.on.delete('mydomain.com', 'v1', 'beamdeployments')
def handle_delete(spec, **kwargs):
    beam_k8s = BeamK8SConfigured(beam_config)
    project_name = spec.get('projectName', beam_config.get('project_name'))
    beam_k8s.delete_project(project_name)
    print(f"Project {project_name} cleanup initiated.")

kopf.run()
