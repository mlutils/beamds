# This is an example of how to use the BeamDeploy class to deploy a container to an OpenShift cluster.
from src.beam.orchestration import (BeamK8S, BeamPod, BeamDeploy, SecurityContextConfig, MemoryStorageConfig,
                                    ServiceConfig, StorageConfig, RayPortsConfig, UserIdmConfig, CommandConfig)
import time
from src.beam.resources import resource
from src.beam.orchestration.config import K8SConfig
import os

# hparams = K8SConfig()

script_dir = os.path.dirname(os.path.realpath(__file__))
conf_path = resource(os.path.join(script_dir, 'orchestration_configuration.json')).str
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

security_context_config = SecurityContextConfig(**config.get('security_context_config', {}))
memory_storage_configs = [MemoryStorageConfig(**v) for v in config.get('memory_storage_configs', [])]
service_configs = [ServiceConfig(**v) for v in config.get('service_configs', [])]
storage_configs = [StorageConfig(**v) for v in config.get('storage_configs', [])]
ray_ports_configs = [RayPortsConfig(**v) for v in config.get('ray_ports_configs', [])]
user_idm_configs = [UserIdmConfig(**v) for v in config.get('user_idm_configs', [])]
command = CommandConfig(**config.get('command', {}))

deployment = BeamDeploy(
    k8s=k8s,
    project_name=config['project_name'],
    check_project_exists=config['check_project_exists'],
    namespace=config['project_name'],
    replicas=config['replicas'],
    labels=config['labels'],
    image_name=config['image_name'],
    command=command,
    deployment_name=config['deployment_name'],
    create_service_account=config['create_service_account'],
    use_scc=config['use_scc'],
    use_node_selector=config['use_node_selector'],
    node_selector=config['node_selector'],
    scc_name=config['scc_name'],
    cpu_requests=config['cpu_requests'],
    cpu_limits=config['cpu_limits'],
    memory_requests=config['memory_requests'],
    memory_limits=config['memory_limits'],
    use_gpu=config['use_gpu'],
    gpu_requests=config['gpu_requests'],
    gpu_limits=config['gpu_limits'],
    service_configs=service_configs,
    storage_configs=storage_configs,
    ray_ports_configs=ray_ports_configs,
    n_pods=config['n_pods'],
    memory_storage_configs=memory_storage_configs,
    security_context_config=security_context_config,
    entrypoint_args=config['entrypoint_args'],
    entrypoint_envs=config['entrypoint_envs'],
    user_idm_configs=user_idm_configs,
    enable_ray_ports=False
)

# Launch deployment and obtain pod instances
deployment.launch(replicas=1)
