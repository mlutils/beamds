from ..core import Processor
from .pod import BeamPod
from dataclasses import dataclass, field
from kubernetes import client
from kubernetes.client.rest import ApiException
from ..logger import beam_logger as logger
from .units import K8SUnits
from typing import List, Union


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


@dataclass
class RayPortsConfig:
    ray_ports: List[int] = field(default_factory=list)
    enable_ray_ports: bool = False


@dataclass
class StorageConfig:
    pvc_name: str
    pvc_mount_path: str
    create_pvc: bool = False  # Indicates whether to create a route for this service
    pvc_size: Union[K8SUnits, str, int] = '1Gi'
    pvc_access_mode: str = 'ReadWriteMany'

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
    add_capabilities: List[str] = field(default_factory=list)
    enable_security_context: bool = False


class BeamDeploy(Processor):

    def __init__(self, k8s=None, project_name=None, namespace=None,
                 replicas=None, labels=None, image_name=None,
                 deployment_name=None, use_scc=False, deployment=None,
                 cpu_requests=None, cpu_limits=None, memory_requests=None,
                 gpu_requests=None, gpu_limits=None, memory_limits=None, storage_configs=None,
                 service_configs=None, user_idm_configs=None, ray_ports_configs=None, memory_storage_configs=None,
                 security_context_config=None, scc_name=None,
                 service_type=None, entrypoint_args=None, entrypoint_envs=None):
        super().__init__()
        self.k8s = k8s
        self.deployment = deployment
        self.entrypoint_args = entrypoint_args or []
        self.entrypoint_envs = entrypoint_envs or {}
        self.project_name = project_name
        self.namespace = namespace
        self.replicas = replicas
        self.labels = labels
        self.image_name = image_name
        self.deployment_name = deployment_name
        self.service_type = service_type
        self.service_account_name = f"svc{deployment_name}"
        self.use_scc = use_scc
        self.scc_name = scc_name if use_scc else None
        self.cpu_requests = cpu_requests
        self.cpu_limits = cpu_limits
        self.memory_requests = memory_requests
        self.memory_limits = memory_limits
        self.gpu_requests = gpu_requests
        self.gpu_limits = gpu_limits
        self.service_configs = service_configs or []
        self.ray_ports_configs = ray_ports_configs or RayPortsConfig()
        self.storage_configs = storage_configs or []
        self.memory_storage_configs = memory_storage_configs or []
        self.user_idm_configs = user_idm_configs or []
        self.security_context_config = security_context_config or []

    def launch(self, replicas=None):
        if replicas is None:
            replicas = self.replicas

        self.k8s.create_project(self.namespace)

        self.k8s.create_service_account(self.service_account_name, self.namespace)

        if self.storage_configs:
            for storage_config in self.storage_configs:
                try:
                    self.k8s.core_v1_api.read_namespaced_persistent_volume_claim(name=storage_config.pvc_name,
                                                                                 namespace=self.namespace)
                    logger.info(f"PVC '{storage_config.pvc_name}' already exists in namespace '{self.namespace}'.")
                except ApiException as e:
                    if e.status == 404 and storage_config.create_pvc:
                        logger.info(f"Creating PVC for storage config: {storage_config.pvc_name}")
                        self.k8s.create_pvc(
                            pvc_name=storage_config.pvc_name,
                            pvc_size=storage_config.pvc_size.as_str,
                            pvc_access_mode=storage_config.pvc_access_mode,
                            namespace=self.namespace
                        )
                    else:
                        logger.info(f"Skipping PVC creation for: {storage_config.pvc_name} as create_pvc is False")

        enabled_memory_storages = [config for config in self.memory_storage_configs if config.enabled]

        for svc_config in self.service_configs:
            service_name = f"{self.deployment_name}-{svc_config.service_name}-{svc_config.port}"
            # Unique name based on service name and port
            self.k8s.create_service(
                base_name=f"{self.deployment_name}-{svc_config.service_name}-{svc_config.port}",
                namespace=self.namespace,
                ports=[svc_config.port],
                labels=self.labels,
                service_type=svc_config.service_type
            )

            # Check if a route needs to be created for this service
            if svc_config.create_route:
                self.k8s.create_route(
                    service_name=service_name,
                    namespace=self.namespace,
                    protocol=svc_config.route_protocol,
                    port=svc_config.port  # Corrected from port_name to port, passing the numeric port value
                )

            # Check if an ingress needs to be created for this service
            if svc_config.create_ingress:
                self.k8s.create_ingress(
                    service_configs=[svc_config],  # Pass only the current ServiceConfig
                )
        if self.user_idm_configs:
            self.k8s.create_role_bindings(self.user_idm_configs)

        if self.use_scc is True:
            self.k8s.add_scc_to_service_account(self.service_account_name, self.namespace, self.scc_name)

        extracted_ports = [svc_config.port for svc_config in self.service_configs]

        for ray_ports_config in self.ray_ports_configs:
            extracted_ports += [ray_port for ray_port in ray_ports_config.ray_ports]

        deployment = self.k8s.create_deployment(
            image_name=self.image_name,
            labels=self.labels,
            deployment_name=self.deployment_name,
            namespace=self.namespace,
            project_name=self.project_name,
            replicas=replicas,
            ports=extracted_ports,
            service_account_name=self.service_account_name,  # Pass this
            storage_configs=self.storage_configs,
            memory_storage_configs=enabled_memory_storages,
            cpu_requests=self.cpu_requests,
            cpu_limits=self.cpu_limits,
            memory_requests=self.memory_requests,
            memory_limits=self.memory_limits,
            gpu_requests=self.gpu_requests,
            gpu_limits=self.gpu_limits,
            security_context_config=self.security_context_config,
            entrypoint_args=self.entrypoint_args,
            entrypoint_envs=self.entrypoint_envs,
        )
        # self.k8s.apply_deployment(deployment, namespace=self.namespace)
        pod_info = self.k8s.apply_deployment(deployment, namespace=self.namespace)

        if pod_info is list:
            # If the deployment was successfully applied, create and return a BeamPod instance
            return [self.generate_beam_pod(pod) for pod in pod_info]
        elif pod_info is not None:
            # If the deployment was successfully applied, create and return a BeamPod instance
            return self.generate_beam_pod(pod_info)
        else:
            # Handle the case where the deployment application failed
            logger.error("Failed to apply deployment")
            return None

    def generate_beam_pod(self, pod_info):
        logger.info(f"Generating BeamPod for pod: '{pod_info}'")
        # Assuming pod_info is an object, extract the pod name as a string
        pod_name = pod_info.name  # Adjust this line based on the actual structure of pod_info
        return BeamPod(pod_name, namespace=self.namespace, k8s=self.k8s)

    def delete_deployment(self):
        # Delete deployment
        try:
            self.k8s.apps_v1_api.delete_namespaced_deployment(
                name=self.deployment.metadata.name,
                namespace=self.deployment.metadata.namespace,
                body=client.V1DeleteOptions()
            )
            logger.info(f"Deleted deployment '{self.deployment.metadata.name}' "
                        f"from namespace '{self.deployment.metadata.namespace}'.")
        except ApiException as e:
            logger.error(f"Error deleting deployment '{self.deployment.metadata.name}': {e}")

        # Delete related services
        try:
            self.k8s.delete_service(deployment_name=self.deployment_name)
        except ApiException as e:
            logger.error(f"Error deleting service '{self.deployment_name}: {e}")

        # Delete related routes
        try:
            self.k8s.delete_route(
                route_name=f"{self.deployment.metadata.name}-route",
                namespace=self.deployment.metadata.namespace,
            )
            logger.info(f"Deleted route '{self.deployment.metadata.name}-route' "
                        f"from namespace '{self.deployment.metadata.namespace}'.")
        except ApiException as e:
            logger.error(f"Error deleting route '{self.deployment.metadata.name}-route': {e}")

        # Delete related ingress
        try:
            self.k8s.delete_service(deployment_name=self.deployment_name)
        except ApiException as e:
            logger.error(f"Error deleting service for deployment '{self.deployment_name}': {e}")

    # move to BeamDeploy
    # def list_pods(self):
    #     label_selector = f"app={self.deployment_name}"
    #     pods = self.core_v1_api.list_namespaced_pod(namespace=self.namespace, label_selector=label_selector)
    #     for pod in pods.items:
    #         print(f"Pod Name: {pod.metadata.name}, Pod Status: {pod.status.phase}")
