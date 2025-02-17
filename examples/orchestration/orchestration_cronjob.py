from beam.orchestration import (BeamManager, CronJobConfig)
from beam.resources import resource, this_dir
from beam.logging import beam_logger as logger
import os
import sys


def main():
    config = CronJobConfig('/home/dayosupp/projects/beamds/examples/orchestration/orchestration_cron_job.yaml')

    logger.info(f"API URL: {config.api_url}")
    logger.info(f"API Token: {config.api_token}")
    logger.info(str(config))
    manager = BeamManager(config)

    manager.launch_cron_job(config)

if __name__ == '__main__':
    main()
