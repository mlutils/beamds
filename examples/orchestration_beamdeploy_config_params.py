# This is an example of how to use the BeamDeploy class to deploy a container to an OpenShift cluster.
from src.beam.orchestration import (BeamK8S, BeamPod, BeamDeploy, ServiceConfig, StorageConfig,
                                    RayPortsConfig, UserIdmConfig, MemoryStorageConfig, SecurityContextConfig)
import time

from src.beam.orchestration.config import K8SConfig

# api_token = get_token()
api_token: str = 'eyJhbGciOiJSUzI1NiIsImtpZCI6IkFOYndaWldZRXFjT0FXeURWOVl4aWVQUE9OZW14UGRGR0RxTi1pb2czMG8ifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJkZWZhdWx0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6InRva2Fuc3ZjLXRva2VuIiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQubmFtZSI6InRva2Fuc3ZjIiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQudWlkIjoiNzZkYTdhYmQtOTE2Ni00ODI3LTkzYjEtN2YzNGFjMzk2ODY4Iiwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OmRlZmF1bHQ6dG9rYW5zdmMifQ.BrC9QLxHSMimPonn23f_KEEyPkMmLLfG14eVRn-Tt-WkuBlvfzuT6iVZQd-1ZmkCR7Q7LAhFRQFjyG53E-mhcppIFYcaR0_k4zxI7i1bet93wkVG4NdOKK75eudQQIrtN8-vqPyaqzdRB3PgljrWSxQQMFznXE-f2-JUH7WqcijiraBanXPy0wmgrx_4v2u5H1hotz-g9qKqdKcolZps1FmGrn1NjrA6a9Qz_jpj1BLOnRqjiPu9Ee1ATSKoVSic3a1ibztZ1VdzfIrehZE-junIYJlRinduIh1dD7xS0o3SWPk0sZigK6bKn6nAhm95lnrJcXOH9Mgq6-VHLDEIMmjt6hCYKyOoJqm_4QDUGDkixi1QlEnmu8GeIm3RFrrOvTkY-tQijoUzvCHZhigCEwbKefwNPdxMoqJF9Q1u5xLqnf6sF_KnGcTPhVu4CzTVLGyWcv0yxgeJ4Z06F9uYb7z-mP59HXqNfWUIFm1AqFLtGkKXzI17drQ1BflrIOIlPaH8ABRbpzm4WHD0sUdw9a8j7iXdhVowiJ5LVSvamkkFPySbXgQrVf4EcFVVh8T9S58scDzlxt1Zv8M6NH_YizSJ3VWXpBPIjFXdVM8dUbrZ3_IV7HAGLcCruZriB-KjUQJe3M-n92lElHMhFB22kV4N38TpTz-Xjq85Wu_nvC8'
hparams = K8SConfig(K8S_API_KEY=api_token, api_url="https://api.dayo-ocp.dt.local:6443", check_project_exists=True,
                    os_namespace="moh", project_name="moh", create_service_account=True,
                    image_name="harbor.dt.local/public/beam:openshift-moh150424", labels={"app": "moh"},
                    deployment_name="moh", replicas=1, use_scc=True,
                    scc_name="anyuid", node_selector=None, cpu_requests="4", cpu_limits="4",
                    memory_requests="12", memory_limits="12", gpu_requests="1", gpu_limits="1",  enable_ray_ports=False)

# api_url = "https://api.dayo-ocp.dt.local:6443"
# api_token = "sha256~EBoYZ2e8ON8BnPGx7187T4viQ-lScg78zcDcbsXFdW0"
# check_project_exists = True
# project_name = "moh"
# create_service_account = True
# image_name = "harbor.dt.local/public/beam:openshift-10.04.24"
# labels = {"app": "moh"}
# deployment_name = "moh"
# namespace = "ben-guryon"
# namespace = project_name
# replicas = 1
entrypoint_args = ["63"]  # Container arguments
entrypoint_envs = {"TEST": "test"}  # Container environment variables
# use_scc = True  # Pass the SCC control parameter
# scc_name = "anyuid"  # privileged , restricted, anyuid, hostaccess, hostmount-anyuid, hostnetwork, node-exporter-scc
security_context_config = (
    SecurityContextConfig(add_capabilities=["SYS_CHROOT", "CAP_AUDIT_CONTROL",
                                            "CAP_AUDIT_WRITE"], enable_security_context=False))
# node_selector = {"gpu-type": "tesla-a100"} #  Node selector in case of GPU scheduling
# node_selector = None
# cpu_requests = "4"  # 0.5 CPU
# cpu_limits = "4"       # 1 CPU
# memory_requests = "12"
# memory_limits = "12"
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
                  create_route=False, create_ingress=False, ingress_host="ssh.example.com"),
    ServiceConfig(port=8888, service_name="jupyter", service_type="LoadBalancer",
                  port_name="jupyter-port", create_route=True, create_ingress=False,
                  ingress_host="jupyter.example.com"),
    ServiceConfig(port=8880, service_name="mlflow", service_type="LoadBalancer",
                  port_name="mlflow-port", create_route=True, create_ingress=False,
                  ingress_host="mlflow.example.com"),
    ServiceConfig(port=8265, service_name="ray-dashboard", service_type="LoadBalancer",
                  port_name="ray-dashboard-port", create_route=True,
                  ingress_host="ray-dashboard.example.com"),
    ServiceConfig(port=6379, service_name="ray-gcs", service_type="LoadBalancer",
                  port_name="ray-gcs-port", create_route=False,
                  ingress_host="ray-gcs.example.com"),

]
# enable_ray_ports=False
ray_ports_configs = [
    RayPortsConfig(ray_ports=[10001, 10002, 10003, 10004, 10005, 10006, 10007,
                              10008, 10009, 10010, 30000, 30001, 30002, 30003, 30004],)
    ]


user_idm_configs = [
    UserIdmConfig(user_name="yos", role_name="admin", role_binding_name="yos",
                  create_role_binding=False, project_name="ben-guryon"),
    UserIdmConfig(user_name="asafe", role_name="admin", role_binding_name="asafe",
                  create_role_binding=False, project_name="ben-guryon"),
]


print('hello world')
print("API URL:", hparams.api_url)
print("API Token:", hparams.K8S_API_KEY)

# the order of the VARS is important!! (see BeamK8S class)
k8s = BeamK8S(
    api_url=hparams.api_url,
    api_token=hparams.K8S_API_KEY,
    project_name=hparams.project_name,
    namespace=hparams.project_name,
)

# k8s = BeamK8S(hparams)

deployment = BeamDeploy(
    k8s=k8s,
    project_name=hparams.project_name,
    namespace=hparams.os_namespace,
    create_service_account=hparams.create_service_account,
    replicas=hparams.replicas,
    labels=hparams.labels,
    image_name=hparams.image_name,
    deployment_name=hparams.deployment_name,
    use_scc=hparams.use_scc,
    node_selector=hparams.node_selector,
    scc_name=hparams.scc_name,
    cpu_requests=hparams.cpu_requests,
    cpu_limits=hparams.cpu_limits,
    memory_requests=hparams.memory_requests,
    memory_limits=hparams.memory_limits,
    gpu_requests=hparams.gpu_requests,
    gpu_limits=hparams.gpu_limits,
    service_configs=service_configs,
    storage_configs=storage_configs,
    ray_ports_configs=ray_ports_configs,
    memory_storage_configs=memory_storage_configs,
    security_context_config=security_context_config,
    entrypoint_args=entrypoint_args,
    entrypoint_envs=entrypoint_envs,
    user_idm_configs=user_idm_configs,
)

# Launch deployment and obtain pod instances
# beam_pod_instance = deployment.launch(hparams.replicas)
# wait_time = 10  # Time to wait before executing commands
# time.sleep(wait_time)
# command = "ls /"  # Command as a regular shell command string


# Handle multiple pod instances
# if isinstance(beam_pod_instance, list):
#     for pod_instance in beam_pod_instance:
#         # Print pod status for each instance
#         print(f"Pod statuses: {pod_instance.get_pod_status()}")
#         # Execute command on each pod instance
#         response = pod_instance.execute(command)
#         print(f"Response from pod {pod_instance.pod_infos[0].metadata.name}: {response}")
#
# # Handle a single pod instance
# elif isinstance(beam_pod_instance, BeamPod):
#     # Print pod status for the single instance
#     print(f"Pod status: {beam_pod_instance.get_pod_status()}")
#     # Execute command on the single pod instance
#     response = beam_pod_instance.execute(command)
#     print(f"Response from pod {beam_pod_instance.pod_infos[0].metadata.name}: {response}")
#
# # Handle case where no valid pod instances are available
# else:
#     print("No valid pod instances available for executing commands.")

# command = "ls /"  # Command as a regular shell command string
# specific_pod_name = "kh-69b46bc57c-hcrfk"
# response = beam_pod_instance.execute(command, pod_name=specific_pod_name)


deployment.launch(replicas=1)
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
internal_endpoints = k8s.get_internal_endpoints_with_nodeport(namespace=hparams.os_namespace)
for endpoint in internal_endpoints:
    print(f"Internal Access: {endpoint['node_ip']}:{endpoint['node_port']}")
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

