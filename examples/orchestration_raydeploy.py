# This is an example of how to use the BeamDeploy class to deploy a container to an OpenShift cluster.
from src.beam.orchestration import (RayClusterConfig, RayCluster)
from src.beam import resource
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
conf_path = resource(os.path.join(script_dir, 'ray_configuration.json')).str
config = RayClusterConfig(conf_path)

print('hello world')
print("API URL:", config['api_url'])
print("API Token:", config['api_token'])
# Create an instance of RayCluster
ray_cluster = RayCluster(deployment=None, config=config)

# Deploy the Ray cluster
ray_cluster.deploy_cluster()

# run on daemon mode the monitor of the cluster
try:
    ray_cluster.monitor_cluster()
except KeyboardInterrupt:
    ray_cluster.stop_monitoring()
    print("Monitoring stopped.")
