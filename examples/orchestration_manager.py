from beam.orchestration import (RayClusterConfig, RayCluster, RnDCluster, BeamManagerConfig,
                                RnDClusterConfig, ServeClusterConfig, ServeCluster, BeamManager)
from beam.resources import resource, this_dir
from beam.logging import beam_logger as logger
import os
import sys


def main():

    script_dir = os.path.dirname(os.path.realpath(__file__))
    conf_path = resource(os.path.join(script_dir, 'orchestration_manager.yaml')).str
    config = BeamManagerConfig(conf_path)

    logger.info(f"API URL: {config.api_url}")
    logger.info(f"API Token: {config.api_token}")
    logger.info("deploy manager with config:")
    logger.info(str(config))
    manager = BeamManager(config)


    # manager.launch_ray_cluster('/home/dayosupp/projects/beamds/examples/orchestration_raydeploy.yaml')
    # manager.launch_serve_cluster('/home/dayosupp/projects/beamds/examples/orchestration_beamdemo.yaml')
    # manager.launch_cron_job('/home/dayosupp/projects/beamds/examples/orchestration_beamdemo.yaml')
    # manager.launch_job('/home/dayosupp/projects/beamds/examples/orchestration_beamdemo.yaml')
    # manager.launch_rnd_cluster('/home/dayosupp/projects/beamds/examples/orchestration_beamdemo.yaml')
    # print(manager.info())
    # manager.monitor_thread()
    # manager.retrieve_cluster_logs('rnd_cluster_name')


if __name__ == '__main__':
    main()



# script_dir = os.path.dirname(os.path.realpath(__file__))
# conf_path = resource(os.path.join(script_dir, 'orchestration_manager.yaml')).str
# config = BeamManagerConfig(conf_path)

# logger.info(f"hello world")
# logger.info(f"API URL: {config.api_url}")
# logger.info(f"API Token: {config.api_token}")

# manager = BeamManager(deployment=None, clusters=config['clusters'], config=config)

# manager.monitor_thread()