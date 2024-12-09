from beam.experiment.utils import cleanup
from beam.orchestration import ServeClusterConfig, BeamManager
from beam.orchestration import deploy_server
from runs.manager import BeamManagerWrapper
from beam import logger


def main():
    config = ServeClusterConfig(path_to_state='/tmp/manager')
    # config = ServeClusterConfig()
    # clean_current = BeamManager(config)
    # clean_current.cleanup()
    logger.info("deploy manager with config :")
    logger.info(str(config))
    manager = BeamManagerWrapper(config)

    deploy_server(manager, config)
    # TODO: Deploy server does not cleanup the existing manager when deploying, it just adds a new one because it runs directly from ServerCluster and not BeamManager
    print(manager.info())


if __name__ == '__main__':
    main()
