from ..type import check_type, Types
from .cluster import ServeCluster
from .jobs import BeamCronJob, BeamJob
from ..resources import resource
from ..logging import beam_logger as logger


# DeployServer
def deploy_server(obj, config):
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


def deploy_job(config):
    # obj_type = check_type(obj)
    # config_type = check_type(config)

    # if config_type.is_path or config_type.is_str:
    #     config = resource(config).read()

    if config.job_schedule:
        logger.info(f"Resource {config.job_schedule} CronJob, deploying CronJob...")
        return BeamCronJob.deploy_cron_job(config)
    # elif obj_type.is_str:
    #     logger.info(f"Resource {obj} does not exist or is treated as a string, deploying from image...")
