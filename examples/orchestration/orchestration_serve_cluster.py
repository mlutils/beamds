from beam.orchestration import ServeClusterConfig
from beam import this_dir, logger, deploy_server


def main():

    config = ServeClusterConfig(this_dir().joinpath('orchestration_serve_cluster.yaml').str)

    logger.info('hello world')
    logger.info("API URL:", config.api_url)
    logger.info("API Token:", config.api_token)

    logger.info("deploy manager with config:")
    logger.info(config)

    path_to_bundle = '/tmp/yolo-bundle'
    deploy_server(path_to_bundle, config)


if __name__ == '__main__':
    main()

