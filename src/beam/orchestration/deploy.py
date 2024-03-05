from ..core import Processor
from .pod import BeamPod
from dataclasses import dataclass
from kubernetes.client.rest import ApiException
from ..logger import beam_logger as logger


@dataclass
class ServiceConfig:
    port: int
    service_name: str
    service_type: str
    port_name: str
    create_route: bool = False  # Indicates whether to create a route for this service
    route_protocol: str = 'http'  # Default to 'http', can be overridden to 'https' as needed
    create_ingress: bool = False  # Indicates whether to create an ingress for this service
    ingress_host: str = None  # Optional: specify a host for the ingress
    ingress_path: str = '/'  # Default path for ingress, can be overridden
    ingress_tls_secret: str = None  # Optional: specify a TLS secret for ingress TLS


@dataclass
class StorageConfig:
    pvc_name: str
    pvc_mount_path: str
    create_pvc: bool = False  # Indicates whether to create a route for this service
    pvc_size: str = '1Gi'
    pvc_access_mode: str = 'ReadWriteMany'


@dataclass
class UserIdmConfig:
    user_name: str
    role_name: str
    role_binding_name: str
    project_name: str
    role_namespace: str = 'default'  # Default to 'default' namespace
    create_role_binding: bool = False  # Indicates whether to create a role_binding this project


class BeamDeploy(Processor):

    def __init__(self, k8s=None, project_name=None, namespace=None,
                 replicas=None, labels=None, image_name=None,
                 deployment_name=None, use_scc=False, cpu_requests=None, cpu_limits=None, memory_requests=None,
                 gpu_requests=None, gpu_limits=None, memory_limits=None, storage_configs=None,
                 service_configs=None, user_idm_configs=None, scc_name='anyuid',
                 service_type=None, *entrypoint_args, **entrypoint_envs):
        super().__init__()
        self.k8s = k8s
        self.entrypoint_args = entrypoint_args
        self.entrypoint_envs = entrypoint_envs
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
        self.service_configs = service_configs
        self.storage_configs = storage_configs
        self.user_idm_configs = user_idm_configs or []

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
                            pvc_size=storage_config.pvc_size,
                            pvc_access_mode=storage_config.pvc_access_mode,
                            namespace=self.namespace
                        )
                    else:
                        logger.info(f"Skipping PVC creation for: {storage_config.pvc_name} as create_pvc is False")

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

        extracted_ports = [svc_config.port for svc_config in self.service_configs]

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
            cpu_requests=self.cpu_requests,
            cpu_limits=self.cpu_limits,
            memory_requests=self.memory_requests,
            memory_limits=self.memory_limits,
            gpu_requests=self.gpu_requests,
            gpu_limits=self.gpu_limits,
            *self.entrypoint_args, **self.entrypoint_envs
        )

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

        #
        # if pod_info is not None:
        #     # If the deployment was successfully applied, create and return a BeamPod instance
        #     return BeamPod(pod_info, deployment=self, k8s=self.k8s)  # Pass the applied deployment
        #     # and any other required arguments
        # else:
        #     # Handle the case where the deployment application failed
        #     logger.error("Failed to apply deployment")
        #     return None
        #

    def generate_beam_pod(self, pod_info):
        return BeamPod(pod_info, deployment=self, k8s=self.k8s)
