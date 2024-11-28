from beam.orchestration import (RayClusterConfig, RayCluster, RnDCluster, BeamManagerConfig,
                                RnDClusterConfig, ServeClusterConfig, ServeCluster, BeamManager)
from beam.resources import resource, this_dir
from beam.logging import beam_logger as logger
import os
import sys


def main():
    config = ServeClusterConfig()

    logger.info(f"API URL: {config.api_url}")
    logger.info(f"API Token: {config.api_token}")
    logger.info("deploy manager with config:")
    config.update({'project_name': 'dev',
        'deployment_name': 'yolo',
        'labels': {'app': 'yolo'},
        'alg': '/tmp/yolo-bundle',
        'debug_sleep': False})
    logger.info(str(config))
    manager = BeamManager(config)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    conf_path = resource(os.path.join(script_dir, 'orchestration_beamdemo.yaml')).str
    manager.launch_job(conf_path)

if __name__ == '__main__':
    main()
