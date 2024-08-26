from beam.orchestration import ServeClusterConfig
from beam.orchestration import BeamManager
from beam.orchestration import deploy_server
from beam import logger


def main():
    config = ServeClusterConfig()

    logger.info("deploy manager with config:")
    logger.info(str(config))
    manager = BeamManager(config)

    deploy_server(manager, config)


if __name__ == '__main__':
    main()
