from beam.orchestration import (RayClusterConfig, RayCluster)
from beam.resources import resource, this_dir
from beam.orchestration import BeamPod as Pod
import os
import sys

if sys.argv[1:]:
    conf_path = sys.argv[1]
else:
    conf_path = this_dir().joinpath('orchestration_raydeploy.yaml')


script_dir = os.path.dirname(os.path.realpath(__file__))
# conf_path = resource(os.path.join(script_dir, 'orchestration_raydeploy.yaml')).str
config = RayClusterConfig(conf_path)

print('hello world')
print("API URL:", config.api_url)
print("API Token:", config.api_token)

# Create an instance of RayCluster
ray_cluster = RayCluster(deployment=None, n_pods=config['n_pods'], config=config)

# n_pods = '2'
# Deploy the Ray cluster
ray_cluster.deploy_ray_cluster_s_deployment(n_pods=config['n_pods'], config=config)
replicas=1

# TODO: save routes to env vars as str

# print(ray_cluster.deployment.cluster_info)

# # run on daemon mode the monitor of the cluster
# try:
#     # ray_cluster.get_cluster_logs()
#     ray_cluster.monitor_cluster()
# except KeyboardInterrupt:
#     ray_cluster.stop_monitoring()
#     print("Monitoring stopped.")
