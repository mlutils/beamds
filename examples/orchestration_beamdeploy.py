# This is an example of how to use the BeamDeploy class to deploy a container to an OpenShift cluster.
from beam.orchestration import (BeamK8S, BeamPod, BeamDeploy, SecurityContextConfig, MemoryStorageConfig,
                                ServiceConfig, StorageConfig, RayPortsConfig, UserIdmConfig, CommandConfig)
from beam.logging import beam_logger as logger
import time
from beam.resources import resource
from beam.orchestration.config import K8SConfig
import os

# hparams = K8SConfig()

script_dir = os.path.dirname(os.path.realpath(__file__))
conf_path = resource(os.path.join(script_dir, 'orchestration_beamdeploy.yaml')).str
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

# security_context_config = SecurityContextConfig(**config.get('security_context_config', {}))
# memory_storage_configs = [MemoryStorageConfig(**v) for v in config.get('memory_storage_configs', [])]
# service_configs = [ServiceConfig(**v) for v in config.get('service_configs', [])]
# storage_configs = [StorageConfig(**v) for v in config.get('storage_configs', [])]
# ray_ports_configs = [RayPortsConfig(**v) for v in config.get('ray_ports_configs', [])]
# user_idm_configs = [UserIdmConfig(**v) for v in config.get('user_idm_configs', [])]
# command = CommandConfig(**config.get('command', {}))

# where the config should be? either:
# 1. first argument of the BeamDeploy class
# 2. hparams=config

# deployment = BeamDeploy(config, k8s, command=command,
#                         service_configs=service_configs,
#                         storage_configs=storage_configs,
#                         ray_ports_configs=ray_ports_configs,
#                         memory_storage_configs=memory_storage_configs,
#                         security_context_config=security_context_config,
#                         user_idm_configs=user_idm_configs,
#                         # example of overwriting an argument using kwargs:
#                         # base_url='xxxxx',
#                         # this would overwrite the base_url from the config
#                         enable_ray_ports=False)

deployment = BeamDeploy(config, k8s,
                        # example of overwriting an argument using kwargs:
                        # base_url='xxxxx',
                        # this would overwrite the base_url from the config
                        enable_ray_ports=False)


# Launch deployment and obtain pod instances
deployment.launch(replicas=1)
