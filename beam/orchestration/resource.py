from ..type import check_type, Types
from .cluster import HTTPServeCluster
from ..resources import resource


def deploy_server(obj, config):
    obj_type = check_type(obj)
    config_type = check_type(config)

    if config_type.is_path or config_type.is_str:
        config = resource(config).read()

    if (obj_type.is_str and resource(obj).exists()) or obj_type.is_path:
        return HTTPServeCluster.deploy_from_bundle(obj, config)
    elif obj_type.is_str:
        return HTTPServeCluster.deploy_from_image(obj, config)
    else:
        return HTTPServeCluster.deploy_from_algorithm(obj, config)
