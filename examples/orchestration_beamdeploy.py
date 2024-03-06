# This is an example of how to use the BeamDeploy class to deploy a container to an OpenShift cluster.
from src.beam.orchestration import (BeamK8S, BeamDeploy, ServiceConfig,
                                    StorageConfig, UserIdmConfig, SecurityContextConfig)


api_url = "https://api.kh-dev.dt.local:6443"
api_token = "sha256~3yeKlg7oWUnhGjYPNObmhcp8bLVi8YSiIbWlSE-sjRQ"
project_name = "ben-guryon"
image_name = "harbor.dt.local/public/beam:openshift-20.02.1"
labels = {"app": "bgu"}
deployment_name = "bgu"
# namespace = "ben-guryon"
namespace = project_name
replicas = 1
entrypoint_args = ["63"]  # Container arguments
entrypoint_envs = {"TEST": "test"}  # Container environment variables
use_scc = True  # Pass the SCC control parameter
scc_name = "anyuid"  # privileged , restricted, anyuid, hostaccess, hostmount-anyuid, hostnetwork, node-exporter-scc
security_context_config = (
    SecurityContextConfig(add_capabilities=["SYS_CHROOT", "CAP_AUDIT_CONTROL",
                                            "CAP_AUDIT_WRITE"], enable_security_context=True))
cpu_requests = "2000m"  # 0.5 CPU
cpu_limits = "2000m"       # 1 CPU
memory_requests = "24Gi"
memory_limits = "24Gi"
gpu_requests = "1"
gpu_limits = "1"
storage_configs = [
    StorageConfig(pvc_name="data-pvc", pvc_mount_path="/data-pvc",
                  pvc_size="500Gi", pvc_access_mode="ReadWriteMany", create_pvc=True),
]
service_configs = [
    ServiceConfig(port=22, service_name="ssh", service_type="LoadBalancer", port_name="ssh-port",
                  create_route=False, create_ingress=False, ingress_host="ssh.example.com"),
    ServiceConfig(port=88, service_name="jupyter", service_type="LoadBalancer", port_name="jupyter-port",
                  create_route=True, create_ingress=False, ingress_host="jupyter.example.com"),
]
user_idm_configs = [
    UserIdmConfig(user_name="yos", role_name="admin", role_binding_name="yos",
                  create_role_binding=True, project_name="ben-guryon"),
    UserIdmConfig(user_name="asafe", role_name="admin", role_binding_name="asafe",
                  create_role_binding=True, project_name="ben-guryon"),
]


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

deployment = BeamDeploy(
    k8s=k8s,
    project_name=project_name,
    namespace=project_name,
    replicas=replicas,
    labels=labels,
    image_name=image_name,
    deployment_name=deployment_name,
    use_scc=use_scc,
    scc_name=scc_name,
    cpu_requests=cpu_requests,
    cpu_limits=cpu_limits,
    memory_requests=memory_requests,
    memory_limits=memory_limits,
    gpu_requests=gpu_requests,
    gpu_limits=gpu_limits,
    service_configs=service_configs,
    storage_configs=storage_configs,
    security_context_config=security_context_config,
    entrypoint_args=entrypoint_args,
    entrypoint_envs=entrypoint_envs,
    user_idm_configs=user_idm_configs,
)

# deployment.launch(replicas=1)

beam_pod_instance = deployment.launch(replicas=1)
available_resources = k8s.query_available_resources()
print("Available Resources:", available_resources)

print("Fetching external endpoints...")
internal_endpoints = k8s.get_internal_endpoints_with_nodeport(namespace=namespace)
for endpoint in internal_endpoints:
    print(f"Internal Access: {endpoint['node_ip']}:{endpoint['node_port']}")