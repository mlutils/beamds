from typing import List

from src.beam import beam_logger as logger
from src.beam.orchestration import BeamK8S
from kubernetes import client, config

api_url = "https://api.kh-dev.dt.local:6443"
api_token = "sha256~rcBAFKblEq83dkshTqhMeCa8Gl2kUSHP4QunM8ukxJI"
project_name = "kleiner"
image_name = "harbor.dt.local/public/beam:20231221"
ports: list[int] = [80, 22,]
labels = {"app": "beamds"}
deployment_name = "beamds"
namespace = "kleiner"
replicas = 1
args = ["63"]  # Container arguments
env = {"TEST": "test"}  # Container environment variables

print('hello world')

print("API URL:", api_url)
print("API Token:", api_token)

# the order of the VARS is important!!! (see BeamK8S class)
k8s = BeamK8S(
    api_url=api_url,
    api_token=api_token,
    project_name=project_name,
    namespace=namespace,
    replicas=1,  # Ensure this is an integer
    labels=labels,
    image_name=image_name,
    deployment_name=deployment_name,
    ports=ports,
    env=env
)
#k8s = k8s.create_deployment()
k8s = k8s.apply_deployment()

# k8s.get_pods(namespace="kleiner")
# for ns in k8s.namespaces.items:
#     logger.info(f"Namespace: {ns.metadata.name}")
#
# k8s.get_deployments(namespace="kleiner")




#k8s = BeamK8S(api_url="https://api.kh-dev.dt.local:6443", api_token="sha256~rcBAFKblEq83dkshTqhMeCa8Gl2kUSHP4QunM8ukxJI")

# k8s.get_pods(namespace="kleiner")
#
# for ns in k8s.namespaces.items:
#     logger.info(f"Namespace: {ns.metadata.name}")
#
# k8s.get_deployments(namespace="kleiner")
#print(deployment)

# api_instance = client.AppsV1Api()
#
# api_response = api_instance.create_namespaced_deployment(
#     body=deployment,
#     namespace=namespace
# )
#
# print("Deployment created. status='%s'" % str(api_response.status))


#k8s.create_deployment(namespace="kleiner", deployment_name="yoss", image="beam", replicas=1, port=80, cpu_request=0.1, cpu_limit=1)


# k8s.generate_yaml_config('/tmp/my_conf.yaml', namespace='kleiner', deployment_name='yoss', image='beam',
#                          replicas=1, port=80, cpu_request=0.1, cpu_limit=1,)
#
# k8s.generate_file_config('/tmp/my_conf.json', namespace='kleiner', deployment_name='yoss', image='beam',
#

#k8s.get_cpu_resources(namespace="kleiner", deployment_name="Yos")

