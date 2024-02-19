from typing import List
from src.beam import beam_logger as logger
from src.beam.orchestration import BeamK8S
from kubernetes import client, config
from src.beam.orchestration.k8s import BeamDeploy, ServiceConfig

api_url = "https://api.kh-dev.dt.local:6443"
api_token = "sha256~6fzKeFQhEO2I4skt5Bj07epEiYsZclZIvXsrD0TiwRk"
project_name = "ben-guryon"
image_name = "harbor.dt.local/public/beam:openshift-17.02.2"
# ports: list[int] = [22, 8888]
labels = {"app": "bgu"}
deployment_name = "bgu"
namespace = "ben-guryon"
replicas = 1
entrypoint_args = ["63"]  # Container arguments
entrypoint_envs = {"TEST": "test"}  # Container environment variables
use_scc = True  # Pass the SCC control parameter
# service_type = "NodePort" # Service types are: ClusterIP, NodePort, LoadBalancer, ExternalName
service_configs = [
    ServiceConfig(port=22, service_name="ssh", service_type="NodePort", port_name="ssh-port",
                  create_route=False, create_ingress=False, ingress_host="ssh.example.com"),
    ServiceConfig(port=88, service_name="jupyter", service_type="LoadBalancer", port_name="jupyter-port",
                  create_route=True, create_ingress=False, ingress_host="jupyter.example.com"),
]

print('hello world')
print("API URL:", api_url)
print("API Token:", api_token)

# the order of the VARS is important!!! (see BeamK8S class)
k8s = BeamK8S(
    api_url=api_url,
    api_token=api_token,
    project_name=project_name,
    namespace=namespace,
)

deployment = BeamDeploy(
    k8s=k8s,
    project_name=project_name,
    namespace=namespace,
    replicas=replicas,
    labels=labels,
    image_name=image_name,
    deployment_name=deployment_name,
    service_configs=service_configs,
    use_scc=use_scc,
    scc_name="anyuid",
    entrypoint_args=entrypoint_args,
    entrypoint_envs=entrypoint_envs,
)

deployment.launch(replicas=1)

print("Fetching external endpoints...")
internal_endpoints = k8s.get_internal_endpoints_with_nodeport(namespace=namespace)
for endpoint in internal_endpoints:
    print(f"Internal Access: {endpoint['node_ip']}:{endpoint['node_port']}")