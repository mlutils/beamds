# This is an example of how to use the BeamDeploy class to deploy a container to an OpenShift cluster.
from src.beam.orchestration import BeamDeploy, ServiceConfig, StorageConfig, UserIdmConfig
from src.beam.orchestration import BeamK8S


api_url = "https://api.kh-dev.dt.local:6443"
api_token = "sha256~8vcDV8ltdAG4nepbeUo6X7O9xxN7VjVL7T9cQmZTLLc"
project_name = "ben-guryon"
image_name = "harbor.dt.local/public/beam:openshift-20.02.1"
labels = {"app": "bgu"}
deployment_name = "bgu"
namespace = "ben-guryon"
replicas = 1
entrypoint_args = ["63"]  # Container arguments
entrypoint_envs = {"TEST": "test"}  # Container environment variables
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
    use_scc=use_scc,
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
    user_idm_configs=user_idm_configs,
)

deployment.launch(replicas=1)

print("Fetching external endpoints...")
internal_endpoints = k8s.get_internal_endpoints_with_nodeport(namespace=namespace)
for endpoint in internal_endpoints:
    print(f"Internal Access: {endpoint['node_ip']}:{endpoint['node_port']}")


