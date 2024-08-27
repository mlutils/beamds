from beam.orchestration import (RayClusterConfig, RayCluster, RnDCluster, BeamManagerConfig,
                                RnDClusterConfig, ServeClusterConfig, ServeCluster, BeamManager)
from beam.resources import resource, this_dir
from beam.logging import beam_logger as logger
import os
import sys


script_dir = os.path.dirname(os.path.realpath(__file__))
conf_path = resource(os.path.join(script_dir, 'orchestration_manager.yaml')).str
config = BeamManagerConfig(conf_path)

logger.info(f"hello world")
logger.info(f"API URL: {config.api_url}")
logger.info(f"API Token: {config.api_token}")

manager = BeamManager(deployment=None, clusters=config['clusters'], config=config)

manager.monitor_thread()
