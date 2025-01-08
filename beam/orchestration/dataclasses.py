from dataclasses import dataclass, field, asdict
from typing import List, Union, Optional
from .units import K8SUnits
from typing import Dict, Any


@dataclass
class ServiceConfig:
    port: int
    service_name: str
    service_type: str  # NodePort, ClusterIP, LoadBalancer
    port_name: str
    create_route: bool = False  # Indicates whether to create a route for this service
    route_protocol: str = 'http'  # Default to 'http', can be overridden to 'https' as needed
    create_ingress: bool = False  # Indicates whether to create an ingress for this service
    ingress_host: str = None  # Optional: specify a host for the ingress
    ingress_path: str = '/'  # Default path for ingress, can be overridden
    ingress_tls_secret: str = None  # Optional: specify a TLS secret for ingress TLS
    # route_timeout: str = '599'
    annotations: Optional[Dict[str, str]] = field(default_factory=dict)



@dataclass
class CommandConfig:
    executable: str
    arguments: List[str] = field(default_factory=list)

    def as_list(self) -> List[str]:
        """Converts the command configuration to a list format suitable for Kubernetes containers."""
        return [self.executable] + self.arguments


@dataclass
class RayPortsConfig:
    ray_ports: List[int] = field(default_factory=list)


@dataclass
class StorageConfig:
    pvc_name: str
    pvc_mount_path: str
    create_pvc: bool = False  # Indicates whether to create a route for this service
    pvc_size: Union[K8SUnits, str, int] = '1Gi'
    pvc_access_mode: str = 'ReadWriteOnce'
    storage_class_name: Optional[str] = None

    def __post_init__(self):
        self.pvc_size = K8SUnits(self.pvc_size)


@dataclass
class MemoryStorageConfig:
    name: str
    mount_path: str
    size_gb: Union[K8SUnits, str, int] = None  # Optional size in GB
    enabled: bool = True  # Indicates whether this memory storage should be applied

    def __post_init__(self):
        self.size_gb = K8SUnits(self.size_gb)


@dataclass
class UserIdmConfig:
    user_name: str
    role_name: str
    role_binding_name: str
    project_name: str
    role_namespace: str = 'default'  # Default to 'default' namespace
    create_role_binding: bool = False  # Indicates whether to create a role_binding this project


@dataclass
class SecurityContextConfig:
    privileged: bool = False
    runAsUser: str = None
    add_capabilities: List[str] = field(default_factory=list)
    enable_security_context: bool = False


@dataclass
class PodMetadata:
    name: str


@dataclass
class PodInfos:
    raw_pod_data: Dict[str, Any]
    metadata: PodMetadata

    @property
    def name(self):
        return self.metadata.name


@dataclass
class RestartPolicyConfig:
    condition: str  # "OnFailure" or "Never"
    max_attempts: int  # Kubernetes backoffLimit
    delay: str  # Delay between retries, e.g., "5s"
    active_deadline_seconds: int  # Maximum time a pod can be active
    window: str  # Optional, custom retry window, might require custom logic in your app

    def to_dict(self):
        return asdict(self)

# Todo: build configuration classes according to the deployment layer structure
# Todo: check online existing packages that have these configurations

@dataclass
class PodConfig:
    pass


@dataclass
class ContainerConfig:
    pass


@dataclass
class DeploymentConfig:
    container: ContainerConfig
    pod: PodConfig

    def get_labels(self) -> Dict[str, str]:
        """Returns the labels as a dictionary."""
        return self.labels.as_dict()
