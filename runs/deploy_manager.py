from beam.orchestration import K8SConfig
from beam.orchestration import BeamManager
from beam.orchestration import deploy_server


def main():
    config = K8SConfig()
    manager = BeamManager(config)
    deploy_server(manager, config)


if __name__ == '__main__':
    main()
