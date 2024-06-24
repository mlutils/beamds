# This is an example of how to use the BeamDeploy class to deploy a container to an OpenShift cluster.
from beam.orchestration import BeamK8S, BeamPod

api_url = "https://api.kh-dev.dt.local:6443"
api_token = "sha256~y4xvFJcBdK_MckDxoeMh48v4gdsQ1CQUpky2RTI2gHs"
project_name = "ben-guryon"
deployment_name = "bgu-4"

print('hello world')
print("API URL:", api_url)
print("API Token:", api_token)

# the order of the VARS is important!! (see BeamK8S class)
k8s = BeamK8S(
    api_url=api_url,
    api_token=api_token,
    project_name=project_name,
    namespace=project_name,
)

deployment = k8s.apps_v1_api.read_namespaced_deployment(name=deployment_name, namespace=project_name)

pod = BeamPod(
    deployment=deployment,  # Pass the deployment object instead of the deployment name
    api_url=api_url,
    api_token=api_token,
    project_name=project_name,
    namespace=project_name,
)

k8s.delete_service(deployment_name=deployment_name)

