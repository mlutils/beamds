from beam.orchestration import ServeClusterConfig
from beam.orchestration import deploy_server
from runs.manager import BeamManagerWrapper
from beam import logger


def main():
    config = ServeClusterConfig(path_to_state='/tmp/manager')
    # config = ServeClusterConfig()

    logger.info("deploy manager with config :")
    logger.info(str(config))
    manager = BeamManagerWrapper(config)

    deploy_server(manager, config)
    print(manager.info())


if __name__ == '__main__':
    main()
