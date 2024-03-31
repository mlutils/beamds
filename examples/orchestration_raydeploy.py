# This is an example of how to use the BeamDeploy class to deploy a container to an OpenShift cluster.
from src.beam.orchestration import (BeamK8S, BeamPod, BeamDeploy, ServiceConfig, StorageConfig,
                                    RayPortsConfig, UserIdmConfig, MemoryStorageConfig, SecurityContextConfig)
import time

# Initial configuration for the head deployment
api_url = "https://api.kh-dev.dt.local:6443"
api_token = "sha256~y4xvFJcBdK_MckDxoeMh48v4gdsQ1CQUpky2RTI2gHs"
project_name = "kh-dev"
image_name = "harbor.dt.local/public/beam:openshift-20.02.6"
labels = {"app": "kh-ray-cluster"}
deployment_name = "kh-ray-head"
# namespace = "ben-guryon"
namespace = project_name
replicas = 1
entrypoint_args = ["63"]  # Container arguments
entrypoint_envs = {"TEST": "test"}  # Container environment variables
use_scc = True  # Pass the SCC control parameter
scc_name = "anyuid"  # privileged , restricted, anyuid, hostaccess, hostmount-anyuid, hostnetwork, node-exporter-scc
security_context_config = (
    SecurityContextConfig(add_capabilities=["SYS_CHROOT", "CAP_AUDIT_CONTROL",
                                            "CAP_AUDIT_WRITE"], enable_security_context=False))
# node_selector = {"gpu-type": "tesla-a100"} # Node selector in case of GPU scheduling
node_selector = None
cpu_requests = "4"  # 0.5 CPU
cpu_limits = "4"       # 1 CPU
memory_requests = "12"
memory_limits = "12"
# gpu_requests = "1"
# gpu_limits = "1"
storage_configs = [
    StorageConfig(pvc_name="data-pvc", pvc_mount_path="/data-pvc",
                  pvc_size="500", pvc_access_mode="ReadWriteMany", create_pvc=True),
]
memory_storage_configs = [
    MemoryStorageConfig(name="dshm", mount_path="/dev/shm", size_gb=8, enabled=True),
    # Other MemoryStorageConfig instances as needed
]
# beam_ports(initials=234)
# # returns service_configs: {'ssh': 23422, 'jupyter': 23488, 'mlflow': 23480, }
service_configs = [
    ServiceConfig(port=2222, service_name="ssh", service_type="NodePort", port_name="ssh-port",
                  create_route=True, create_ingress=False, ingress_host="ssh.example.com"),
    ServiceConfig(port=8888, service_name="jupyter", service_type="LoadBalancer",
                  port_name="jupyter-port", create_route=True, create_ingress=False,
                  ingress_host="jupyter.example.com"),
]
ray_ports_configs = [
    RayPortsConfig(ray_ports=[10001, 10002, 10003, 10004, 10005, 10006, 10007,
                              10008, 10009, 10010, 30000, 30001, 30002, 30003, 30004],)
    ]
# user_idm_configs = [
#     UserIdmConfig(user_name="yos", role_name="admin", role_binding_name="yos",
#                   create_role_binding=True, project_name="ben-guryon"),
#     UserIdmConfig(user_name="asafe", role_name="admin", role_binding_name="asafe",
#                   create_role_binding=True, project_name="ben-guryon"),
# ]
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
    node_selector=node_selector,
    scc_name=scc_name,
    cpu_requests=cpu_requests,
    cpu_limits=cpu_limits,
    memory_requests=memory_requests,
    memory_limits=memory_limits,
    # gpu_requests=gpu_requests,
    # gpu_limits=gpu_limits,
    service_configs=service_configs,
    storage_configs=storage_configs,
    ray_ports_configs=ray_ports_configs,
    memory_storage_configs=memory_storage_configs,
    security_context_config=security_context_config,
    entrypoint_args=entrypoint_args,
    entrypoint_envs=entrypoint_envs,
    # user_idm_configs=user_idm_configs,
)

# Launch deployment and obtain pod instances
worker_deployment = deployment.launch(replicas=1)

wait_time = 10  # Time to wait before executing commands
time.sleep(wait_time)
command = "ls /"  # Command as a regular shell command string

# Handle multiple pod instances
if isinstance(worker_deployment, list):
    for pod_instance in worker_deployment:
        # Print pod status for each instance
        print(f"Pod statuses: {pod_instance.get_pod_status()}")
        # Execute command on each pod instance
        response = pod_instance.execute(command)
        print(f"Response from pod {pod_instance.pod_infos[0].metadata.name}: {response}")

# Handle a single pod instance
elif isinstance(worker_deployment, BeamPod):
    # Print pod status for the single instance
    print(f"Pod status: {worker_deployment.get_pod_status()}")
    # Execute command on the single pod instance
    response = worker_deployment.execute(command)
    print(f"Response from pod {worker_deployment.pod_infos[0].metadata.name}: {response}")

# Initial configuration for the Worker deployment

project_name = "kh-dev"
image_name = "harbor.dt.local/public/beam:openshift-20.02.6"
labels = {"app": "kh-ray-cluster"}
deployment_name = "kh-ray-workers"
# namespace = "ben-guryon"
namespace = project_name
replicas = 1
entrypoint_args = ["63"]  # Container arguments
entrypoint_envs = {"TEST": "test"}  # Container environment variables
use_scc = True  # Pass the SCC control parameter
scc_name = "anyuid"  # privileged , restricted, anyuid, hostaccess, hostmount-anyuid, hostnetwork, node-exporter-scc
security_context_config = (
    SecurityContextConfig(add_capabilities=["SYS_CHROOT", "CAP_AUDIT_CONTROL",
                                            "CAP_AUDIT_WRITE"], enable_security_context=False))
# node_selector = {"gpu-type": "tesla-a100"} # Node selector in case of GPU scheduling
node_selector = None
cpu_requests = "4"  # 0.5 CPU
cpu_limits = "4"       # 1 CPU
memory_requests = "12"
memory_limits = "12"
gpu_requests = "1"
gpu_limits = "1"
storage_configs = [
    StorageConfig(pvc_name="data-pvc", pvc_mount_path="/data-pvc",
                  pvc_size="500", pvc_access_mode="ReadWriteMany", create_pvc=True),
]
memory_storage_configs = [
    MemoryStorageConfig(name="dshm", mount_path="/dev/shm", size_gb=8, enabled=True),
    # Other MemoryStorageConfig instances as needed
]
# beam_ports(initials=234)
# # returns service_configs: {'ssh': 23422, 'jupyter': 23488, 'mlflow': 23480, }
service_configs = [
    ServiceConfig(port=2222, service_name="ssh", service_type="NodePort", port_name="ssh-port",
                  create_route=True, create_ingress=False, ingress_host="ssh.example.com"),
    ServiceConfig(port=8888, service_name="jupyter", service_type="LoadBalancer",
                  port_name="jupyter-port", create_route=True, create_ingress=False,
                  ingress_host="jupyter.example.com"),
]
ray_ports_configs = [
    RayPortsConfig(ray_ports=[10001, 10002, 10003, 10004, 10005, 10006, 10007,
                              10008, 10009, 10010, 30000, 30001, 30002, 30003, 30004],)
    ]
# user_idm_configs = [
#     UserIdmConfig(user_name="yos", role_name="admin", role_binding_name="yos",
#                   create_role_binding=True, project_name="ben-guryon"),
#     UserIdmConfig(user_name="asafe", role_name="admin", role_binding_name="asafe",
#                   create_role_binding=True, project_name="ben-guryon"),
# ]
print('hello world')
print("API URL:", api_url)
print("API Token:", api_token)
# the order of the VARS is important!! (see BeamK8S class)

deployment = BeamDeploy(
    k8s=k8s,
    project_name=project_name,
    namespace=project_name,
    replicas=replicas,
    labels=labels,
    image_name=image_name,
    deployment_name=deployment_name,
    use_scc=use_scc,
    node_selector=node_selector,
    scc_name=scc_name,
    cpu_requests=cpu_requests,
    cpu_limits=cpu_limits,
    memory_requests=memory_requests,
    memory_limits=memory_limits,
    gpu_requests=gpu_requests,
    gpu_limits=gpu_limits,
    service_configs=service_configs,
    storage_configs=storage_configs,
    ray_ports_configs=ray_ports_configs,
    memory_storage_configs=memory_storage_configs,
    security_context_config=security_context_config,
    entrypoint_args=entrypoint_args,
    entrypoint_envs=entrypoint_envs,
    # user_idm_configs=user_idm_configs,
)

# Launch deployment and obtain pod instances
worker_deployment = deployment.launch(replicas=1)

wait_time = 10  # Time to wait before executing commands
time.sleep(wait_time)
command = "ls /"  # Command as a regular shell command string

# Handle multiple pod instances
if isinstance(worker_deployment, list):
    for pod_instance in worker_deployment:
        # Print pod status for each instance
        print(f"Pod statuses: {pod_instance.get_pod_status()}")
        # Execute command on each pod instance
        response = pod_instance.execute(command)
        print(f"Response from pod {pod_instance.pod_infos[0].metadata.name}: {response}")

# Handle a single pod instance
elif isinstance(worker_deployment, BeamPod):
    # Print pod status for the single instance
    print(f"Pod status: {worker_deployment.get_pod_status()}")
    # Execute command on the single pod instance
    response = worker_deployment.execute(command)
    print(f"Response from pod {worker_deployment.pod_infos[0].metadata.name}: {response}")


#
# # Handle case where no valid pod instances are available
# else:
#     print("No valid pod instances available for executing commands.")

# command = "ls /"  # Command as a regular shell command string
# specific_pod_name = "kh-69b46bc57c-hcrfk"
# response = beam_pod_instance.execute(command, pod_name=specific_pod_name)


# deployment.launch(replicas=1)
# available_resources = k8s.query_available_resources()
# print("Available Resources:", available_resources)
# beam_pod_instance = deployment.launch(replicas=1)
# # print("beam pod instance:", beam_pod_instance)
#
# if isinstance(beam_pod_instance, list):  # If there are multiple Pods
#     for pod in beam_pod_instance:
#         print(f"pod statuses' : '{pod.get_pod_status()}'")
#         # Other operations on each pod
# else:
#     print(beam_pod_instance.get_pod_status())
#     # Other operations on the single pod

# print("Pod Status:", beam_pod_instance.pod_status)
# print("Pod Info:", beam_pod_instance.pod_info)

# print("Fetching external endpoints...")
# internal_endpoints = k8s.get_internal_endpoints_with_nodeport(namespace=namespace)
# for endpoint in internal_endpoints:
#     print(f"Internal Access: {endpoint['node_ip']}:{endpoint['node_port']}")
#
# beam_pod = BeamPod(pod_infos=beam_pod_instance.pod_name, namespace=namespace, k8s=k8s)
# # print(beam_pod)
# command = ["ls /"] # Example command
# # command = ("ray start --head --node-ip-address=10.128.0.80 --port=${RAY_REDIS_PORT} "
# #            "--dashboard-port=${RAY_DASHBOARD_PORT} --dashboard-host=0.0.0.0")
# # command = "ray status"
# response = beam_pod.execute(command)
# print(response)

# pod_array = BeamPod(pod_name=beam_pod_instance.pod_name, namespace=namespace, k8s=k8s, replicas=10)
#
# pod_array[0].execute('ray head ...')
# for p in pod_array[1:]:
#     p.execute('ray worker ...')

