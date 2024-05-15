# This is an example of how to use the BeamDeploy class to deploy a container to an OpenShift cluster.
from src.beam.orchestration import (BeamK8S, BeamPod, BeamDeploy, ServiceConfig, StorageConfig,
                                    RayPortsConfig, UserIdmConfig, MemoryStorageConfig, SecurityContextConfig,
                                    RayClusterConfig)
from src.beam import resource
import time
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
conf_path = resource(os.path.join(script_dir, 'ray_configuration.json')).str
config = RayClusterConfig(conf_path)
security_context_config = SecurityContextConfig(**config.get('security_context_config', {}))
memory_storage_configs = [MemoryStorageConfig(**v) for v in config.get('memory_storage_configs', [])]
# print([type(sc) for sc in memory_storage_configs]) 21
service_configs = [ServiceConfig(**v) for v in config.get('service_configs', [])]
# print([type(sc) for sc in service_configs])
storage_configs = [StorageConfig(**v) for v in config.get('storage_configs', [])]
# print([type(sc) for sc in storage_configs])
ray_ports_configs = [RayPortsConfig(**v) for v in config.get('ray_ports_configs', [])]
user_idm_configs = [UserIdmConfig(**v) for v in config.get('user_idm_configs', [])]

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
deployment = BeamDeploy(
    k8s=k8s,
    project_name=config['project_name'],
    check_project_exists=config['check_project_exists'],
    namespace=config['project_name'],
    replicas=config['replicas'],
    labels=config['labels'],
    image_name=config['image_name'],
    deployment_name=config['deployment_name'],
    create_service_account=config['create_service_account'],
    use_scc=config['use_scc'],
    node_selector=config['node_selector'],
    scc_name=config['scc_name'],
    cpu_requests=config['cpu_requests'],
    cpu_limits=config['cpu_limits'],
    memory_requests=config['memory_requests'],
    memory_limits=config['memory_limits'],
    gpu_requests=config['gpu_requests'],
    gpu_limits=config['gpu_limits'],
    service_configs=service_configs,
    storage_configs=storage_configs,
    ray_ports_configs=ray_ports_configs,
    memory_storage_configs=memory_storage_configs,
    security_context_config=security_context_config,
    entrypoint_args=config['entrypoint_args'],
    entrypoint_envs=config['entrypoint_envs'],
    user_idm_configs=user_idm_configs,
)
# Launch deployment and obtain pod instances
head_deployment = deployment.launch(replicas=1)

wait_time = 10  # Time to wait before executing commands
time.sleep(wait_time)

# Retrieving Head Pod IP
# Assuming head_deployment returns a single BeamPod instance or a list with one BeamPod instance for the head
head_pod_instance = head_deployment[0] if isinstance(head_deployment, list) else head_deployment

# Check if the head pod is running
head_pod_status = head_pod_instance.get_pod_status()

head_pod_name = head_pod_instance.pod_infos[0].metadata.name  # Retrieve the head pod name

if head_pod_status[0][1] == "Running":
    # Head pod is running, retrieve the IP address
    head_pod_ip = k8s.get_pod_ip(head_pod_name, )
    print(f"Head Pod IP: {head_pod_ip}")
else:
    # Head pod is not running, raise an error
    raise Exception(f"Head pod {head_pod_name} is not running unable to start head cluster. Current status: {head_pod_status[0][1]}")


command = "ray start --head --port=6379 --disable-usage-stats --dashboard-host=0.0.0.0"  # Command as a regular shell command string

# Handle multiple pod instances
if isinstance(head_deployment, list):
    for pod_instance in head_deployment:
        # Print pod status for each instance
        print(f"Pod statuses: {pod_instance.get_pod_status()}")
        # Execute command on each pod instance
        response = pod_instance.execute(command)
        print(f"Response from pod {pod_instance.pod_infos[0].metadata.name}: {response}")

# Handle a single pod instance
elif isinstance(head_deployment, BeamPod):
    # Print pod status for the single instance
    print(f"Pod status: {head_deployment.get_pod_status()}")
    # Execute command on the single pod instance
    response = head_deployment.execute(command)
    print(f"Response from pod {head_deployment.pod_infos[0].metadata.name}: {response}")

# Initial configuration for the Worker deployment

print('hello world')
print("API URL:", config['api_url'])
print("API Token:", config['api_token'])
# the order of the VARS is important!! (see BeamK8S class)

deployment = BeamDeploy(
    k8s=k8s,
    project_name=config['project_name'],
    namespace=config['project_name'],
    replicas=config['replicas'],
    labels= config['labels'],
    image_name=config['image_name'],
    deployment_name=config['deployment_name'],
    use_scc=config['use_scc'],
    node_selector=config['node_selector'],
    scc_name=config['scc_name'],
    cpu_requests=config['cpu_requests'],
    cpu_limits=config['cpu_limits'],
    memory_requests=config['memory_requests'],
    memory_limits=config['memory_limits'],
    gpu_requests=config['gpu_requests'],
    gpu_limits=config['gpu_limits'],
    service_configs=config['service_configs'],
    storage_configs=config['storage_configs'],
    ray_ports_configs=config['ray_ports_configs'],
    memory_storage_configs=config['memory_storage_configs'],
    security_context_config=config['security_context_config'],
    entrypoint_args=config['entrypoint_args'],
    entrypoint_envs=config['entrypoint_envs'],
    user_idm_configs=config['user_idm_configs'],
)

# Launch deployment and obtain pod instances
worker_deployment = deployment.launch(replicas=1)

wait_time = 10  # Time to wait before executing commands
time.sleep(wait_time)

worker_pod_instance = worker_deployment[0] if isinstance(worker_deployment, list) else worker_deployment

worker_pod_status = worker_pod_instance.get_pod_status()

worker_pod_name = worker_pod_instance.pod_infos[0].metadata.name  # Retrieve the head pod name

# command = "ray start --address={head_pod_ip}:6379"  # Command as a regular shell command string
command = "ray start --address={}:6379".format(head_pod_ip)
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



