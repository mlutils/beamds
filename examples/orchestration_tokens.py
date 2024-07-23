# This is an example of how to use the BeamDeploy class to deploy a container to an OpenShift cluster.
from beam.orchestration import (BeamK8S, BeamPod, BeamDeploy, SecurityContextConfig, MemoryStorageConfig,
                                ServiceConfig, StorageConfig, RayPortsConfig, UserIdmConfig, CommandConfig)
from beam.logging import beam_logger as logger
import time
from beam.resources import resource
from beam.orchestration.config import K8SConfig
import os

# hparams = K8SConfig()

script_dir = os.path.dirname(os.path.realpath(__file__))
conf_path = resource(os.path.join(script_dir, 'orchestration_tokens.yaml')).str
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


# # Initialize your K8S objectd
# api_url = "https://api.kh-dev.dt.local:6443"
# api_token = "your_api_token_here"
# project_name = "example_namespace"
# k8s = K8S(api_url=api_url, api_token=api_token, project_name=project_name)

# Details for the service account
service_account_name = "svcadm"
namespace = config.project_name  # or another namespace where you want to deploy

# Create service account and manage roles
try:
    # This will attempt to create a service account, bind it to an admin role, and create a secret for it
    k8s.create_service_account(service_account_name, namespace)
    k8s.bind_service_account_to_role(service_account_name, namespace,
                                     role='admin')  # Optionally specify a different role

    # Retrieve or create a secret for a persistent token and get the token
    token = k8s.create_or_retrieve_service_account_token(service_account_name, namespace)

    print(f"Service account setup complete in namespace {namespace}.")
    print(f"Token: {token}")  # Output the token so it can be copied
except Exception as e:
    print(f"An error occurred: {str(e)}")




