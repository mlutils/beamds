from typing import List
from src.beam import beam_logger as logger
from src.beam.orchestration import BeamK8S
from kubernetes import client, config
from src.beam.orchestration.k8s import BeamDeploy, ServiceConfig, StorageConfig

api_url = "https://api.kh-dev.dt.local:6443"
api_token = "sha256~szW2nZ6g9cJlvN1gaXYuLouE50pgJDbfO9ty2PIzEzg"
project_name = "ben-guryon"
image_name = "harbor.dt.local/public/beam:openshift-20.02.1"
labels = {"app": "bgu"}
deployment_name = "bgu"
namespace = "ben-guryon"
replicas = 1
entrypoint_args = ["63"]  # Container arguments
entrypoint_envs = {"TEST": "test"}  # Container environment variablesץ
use_scc = True  # Pass the SCC control parameter
cpu_requests = "4"  # 0.5 CPU
cpu_limits = "4"       # 1 CPU
memory_requests = "24Gi"
memory_limits = "24Gi"
gpu_requests = "1"
gpu_limits = "1"
storage_configs = [
    StorageConfig(pvc_name="data-pvc", pvc_mount_path="/data-pvc",
                  pvc_size="500Gi", pvc_access_mode="ReadWriteMany", create_pvc=True),
]
service_configs = [
    ServiceConfig(port=22, service_name="ssh", service_type="NodePort", port_name="ssh-port",
                  create_route=False, create_ingress=False, ingress_host="ssh.example.com"),
    ServiceConfig(port=88, service_name="jupyter", service_type="ClusterIP", port_name="jupyter-port",
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
    cpu_requests=cpu_requests,
    cpu_limits=cpu_limits,
    memory_requests=memory_requests,
    memory_limits=memory_limits,
    gpu_requests=gpu_requests,
    gpu_limits=gpu_limits,
    service_configs=service_configs,
    storage_configs=storage_configs,
    entrypoint_args=entrypoint_args,
    entrypoint_envs=entrypoint_envs,
)

deployment.launch(replicas=1)

print("Fetching external endpoints...")
internal_endpoints = k8s.get_internal_endpoints_with_nodeport(namespace=namespace)
for endpoint in internal_endpoints:
    print(f"Internal Access: {endpoint['node_ip']}:{endpoint['node_port']}")