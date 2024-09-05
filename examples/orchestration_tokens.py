from beam.orchestration import (BeamK8S)
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

# Details for the service account
service_account_name = "default"  # or another name for the service account
# namespace = config.project_name  # or another namespace where you want to deploy
namespace = 'kube-system' # to have service account in kube-system namespace as kubeadmin user

# # Create service account and manage roles
# try:
#     # # This will attempt to delete the service account if it exists, then create it
#     # if service_account_name in k8s.get_service_accounts(namespace):
#     #    # k8s.delete_service_account(service_account_name, namespace)
#     #    pass
#     # else:
#        ## This will attempt to create a service account, bind it to ש role, and create a secret for it
#        # k8s.create_service_account(service_account_name, namespace)
#        # k8s.bind_service_account_to_role(service_account_name, namespace,
#        #                                  role='cluster-admin')  # Optionally specify a different role
#        # k8s.create_cluster_role_binding_for_scc(service_account_name, namespace)
#        # k8s.add_scc_to_service_account(service_account_name, namespace, scc_name="anyuid")
#        # # Retrieve or create a secret for a persistent token and get the token
#
#     token = k8s.create_or_retrieve_service_account_token(service_account_name, namespace)
#
#     logger.info(f"Service account setup complete in namespace {namespace}.")
#     logger.info(f"Token: {token}")  # Output the token so it can be copied
# except Exception as e:
#     logger.info(f"An error occurred: {str(e)}")


try:
    token = k8s.retrieve_service_account_token(service_account_name, namespace)
    logger.info(f"Token retrieved: {token}")

    # Use the token in your logic
    # For example, print or log the token
    logger.info(f"Token: {token}")
except Exception as e:
    logger.error(f"An error occurred while retrieving the token: {e}")




