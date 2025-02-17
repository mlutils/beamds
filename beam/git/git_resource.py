from ..type import check_type, Types
from .cluster import ServeCluster
from ..resources import resource
from ..logging import beam_logger as logger


# DeployServer
def deploy_cicd(config):
    obj_type = check_type(obj)
    config_type = check_type(config)

    if config_type.is_path or config_type.is_str:
        config = resource(config).read()

    if (obj_type.is_str and resource(obj).exists()) or obj_type.is_path:
        logger.info(f"Resource {obj} exists, deploying from bundle...")
        return ServeCluster.deploy_from_bundle(obj, config)
    elif obj_type.is_str:
        logger.info(f"Resource {obj} does not exist or is treated as a string, deploying from image...")
         # TODO: handle case where obj when path used as image name - should change to image name from config to avoid pod deployment failure
        return ServeCluster.deploy_from_image(obj, config)
    else:
        return ServeCluster.deploy_from_algorithm(obj, config)
