from beam.orchestration import (JobManager, RayCluster, RnDCluster, BeamManagerConfig,
                                RnDClusterConfig, ServeClusterConfig, ServeCluster, BeamManager, CronJobConfig)
from beam.resources import resource, this_dir
from beam.logging import beam_logger as logger
import os
import sys


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))

    config_path = resource(os.path.join(script_dir, 'orchestration_cron_job.yaml')).str
    config = CronJobConfig(config_path)

    logger.info(f"API URL: {config.api_url}")
    logger.info(f"API Token: {config.api_token}")
    logger.info(str(config))
    manager = BeamManager(config)

    manager.launch_cron_job(config)

if __name__ == '__main__':
    main()
