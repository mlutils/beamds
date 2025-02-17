from .dataclasses import *

if len([]):

    from .k8s import BeamK8S
    from .deploy import BeamDeploy
    from .statefulsets import BeamStatefulSet
    from .pod import BeamPod
    from .jobs import BeamJobManager, BeamJob, BeamCronJob
    from .units import K8SUnits
    from .utils import ensure_rfc1123_compliance
    from .config import (K8SConfig, RayClusterConfig, ServeClusterConfig, RnDClusterConfig, BeamManagerConfig,
                         JobConfig, CronJobConfig)
    from .cluster import ServeCluster, RayCluster, RnDCluster
    from .manager import BeamManager
    from .resource import deploy_server, deploy_job


__all__ = ['BeamK8S', 'BeamDeploy', 'BeamPod', 'K8SUnits', 'K8SConfig', 'RayClusterConfig', 'ServeClusterConfig',
           'JobConfig', 'CronJobConfig', 'RnDClusterConfig','BeamManagerConfig', 'ServeCluster', 'RayCluster',
           'BeamJobManager', 'RnDCluster', 'BeamManager', 'deploy_server', 'BeamStatefulSet', 'deploy_job', 'ensure_rfc1123_compliance']


def __getattr__(name):
    if name == 'deploy_server':
        from .resource import deploy_server
        return deploy_server
    elif name == 'deploy_job':
        from .resource import deploy_job
        return deploy_job
    elif name == 'BeamManager':
        from .manager import BeamManager
        return BeamManager
    elif name == 'ServeCluster':
        from .cluster import ServeCluster
        return ServeCluster
    elif name == 'RayCluster':
        from .cluster import RayCluster
        return RayCluster
    elif name == 'RnDCluster':
        from .cluster import RnDCluster
        return RnDCluster
    elif name == 'BeamJobManager':
        from .jobs import BeamJobManager
        return BeamJobManager
    elif name == 'BeamJob':
        from .jobs import BeamJob
        return BeamJob
    elif name == 'BeamCronJob':
        from .jobs import BeamCronJob
        return BeamCronJob
    elif name == 'JobConfig':
        from .config import JobConfig
        return JobConfig
    elif name == 'CronJobConfig':
        from .config import CronJobConfig
        return CronJobConfig
    elif name == 'K8SConfig':
        from .config import K8SConfig
        return K8SConfig
    elif name == 'RayClusterConfig':
        from .config import RayClusterConfig
        return RayClusterConfig
    elif name == 'ServeClusterConfig':
        from .config import ServeClusterConfig
        return ServeClusterConfig
    elif name == 'RnDClusterConfig':
        from .config import RnDClusterConfig
        return RnDClusterConfig
    elif name == 'BeamManagerConfig':
        from .config import BeamManagerConfig
        return BeamManagerConfig
    elif name == 'K8SUnits':
        from .units import K8SUnits
        return K8SUnits
    elif name == 'ensure_rfc1123_compliance':
        from .utils import ensure_rfc1123_compliance
        return ensure_rfc1123_compliance
    elif name == 'BeamPod':
        from .pod import BeamPod
        return BeamPod
    elif name == 'BeamDeploy':
        from .deploy import BeamDeploy
        return BeamDeploy
    elif name == 'BeamStatefulSet':
        from .statefulsets import BeamStatefulSet
        return BeamStatefulSet
    elif name == 'BeamK8S':
        from .k8s import BeamK8S
        return BeamK8S
    elif name == 'ServiceConfig':
        from .dataclasses import ServiceConfig
        return ServiceConfig
    elif name == 'CommandConfig':
        from .dataclasses import CommandConfig
        return CommandConfig
    elif name == 'RayPortsConfig':
        from .dataclasses import RayPortsConfig
        return RayPortsConfig
    elif name == 'StorageConfig':
        from .dataclasses import StorageConfig
        return StorageConfig
    elif name == 'MemoryStorageConfig':
        from .dataclasses import MemoryStorageConfig
        return MemoryStorageConfig
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")


