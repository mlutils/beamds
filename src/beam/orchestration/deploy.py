from ..processor import Processor
from .pod import BeamPod
from kubernetes import client
from kubernetes.client.rest import ApiException
from ..logger import beam_logger as logger
from .dataclasses import *


class BeamDeploy(Processor):

    def __init__(self, k8s=None, check_project_exists=False, project_name=None, namespace=None,
                 replicas=None, labels=None, image_name=None,
                 deployment_name=None, use_scc=False, deployment=None, create_service_account=None,
                 cpu_requests=None, cpu_limits=None, memory_requests=None, use_gpu=None,
                 gpu_requests=None, gpu_limits=None, memory_limits=None, storage_configs=None,
                 service_configs=None, user_idm_configs=None, enable_ray_ports=False, ray_ports_configs=None,
                 memory_storage_configs=None, security_context_config=None, use_node_selector=False,
                 scc_name=None, node_selector=None,
                 service_type=None, entrypoint_args=None, entrypoint_envs=None):
        super().__init__()
        self.k8s = k8s
        self.deployment = deployment
        self.entrypoint_args = entrypoint_args or []
        self.entrypoint_envs = entrypoint_envs or {}
        self.check_project_exists = check_project_exists
        self.project_name = project_name
        self.create_service_account = create_service_account
        self.namespace = namespace
        self.replicas = replicas
        self.labels = labels
        self.image_name = image_name
        self.deployment_name = deployment_name
        self.service_type = service_type
        self.service_account_name = f"{deployment_name}svc"
        self.use_scc = use_scc
        self.use_node_selector = use_node_selector
        self.node_selector = node_selector
        self.scc_name = scc_name if use_scc else None
        self.cpu_requests = cpu_requests
        self.cpu_limits = cpu_limits
        self.memory_requests = memory_requests
        self.memory_limits = memory_limits
        self.use_gpu = use_gpu
        self.gpu_requests = gpu_requests
        self.gpu_limits = gpu_limits
        self.service_configs = service_configs or []
        self.enable_ray_ports = enable_ray_ports
        self.ray_ports_configs = ray_ports_configs or RayPortsConfig()
        self.storage_configs = storage_configs or []
        self.memory_storage_configs = memory_storage_configs or []
        self.user_idm_configs = user_idm_configs or []
        self.security_context_config = security_context_config or []

    def launch(self, replicas=None):
        if replicas is None:
            replicas = self.replicas

        if self.check_project_exists is True:
            self.k8s.create_project(self.namespace)

        if self.create_service_account is True:
            self.k8s.create_service_account(self.service_account_name, self.namespace)
        else:
            self.service_account_name = 'default'
            logger.info(f"using default service account '{self.service_account_name}' in namespace '{self.namespace}'.")

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
                    port=svc_config.port
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

        if self.enable_ray_ports is True:
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
            create_service_account=self.create_service_account,
            service_account_name=self.service_account_name,  # Pass this
            storage_configs=self.storage_configs,
            memory_storage_configs=enabled_memory_storages,
            use_node_selector=self.node_selector,
            node_selector=self.node_selector,
            cpu_requests=self.cpu_requests,
            cpu_limits=self.cpu_limits,
            memory_requests=self.memory_requests,
            memory_limits=self.memory_limits,
            use_gpu=self.use_gpu,
            gpu_requests=self.gpu_requests,
            gpu_limits=self.gpu_limits,
            security_context_config=self.security_context_config,
            entrypoint_args=self.entrypoint_args,
            entrypoint_envs=self.entrypoint_envs,
        )

        pod_infos = self.k8s.apply_deployment(deployment, namespace=self.namespace)

        beam_pod_instances = []

        if isinstance(pod_infos, list) and pod_infos:
            for pod_info in pod_infos:
                pod_name = getattr(pod_info, 'name', None)
                # Print each pod_info for debugging
                print(f"Processing pod_info: {pod_info}")
                # Extract the pod name from pod_info
                print(f"Extracted pod_name: {pod_name}")
                if pod_name:
                    actual_pod_info = self.k8s.get_pod_info(pod_name, self.namespace)
                    # print(f"Fetched actual_pod_info for pod_name '{pod_name}': {actual_pod_info}")
                    # Create a BeamPod instance with the detailed Pod info
                    beam_pod_instance = BeamPod(pod_infos=[actual_pod_info], namespace=self.namespace, k8s=self.k8s)
                    beam_pod_instances.append(beam_pod_instance)
                else:
                    logger.warning("PodInfo object does not have a 'name' attribute.")

        # If pod_infos is not a list but a single object with a name attribute
        elif pod_infos and hasattr(pod_infos, 'name'):
            pod_name = pod_infos.name
            print(f"Single pod_info with pod_name: {pod_name}")

            actual_pod_info = self.k8s.get_pod_info(pod_name, self.namespace)
            print(f"Fetched actual_pod_info for pod_name '{pod_name}': {actual_pod_info}")

            # Directly return the single BeamPod instance
            return BeamPod(pod_infos=[actual_pod_info], namespace=self.namespace, k8s=self.k8s)

        # Handle cases where deployment failed or no pods were returned
        if not beam_pod_instances:
            logger.error("Failed to apply deployment or no pods were returned.")
            return None

        # Return a single BeamPod instance or a list of them, based on the number of instances created
        return beam_pod_instances if len(beam_pod_instances) > 1 else beam_pod_instances[0]

    def generate_beam_pod(self, pod_infos):
        # logger.info(f"Generating BeamPod for pods: '{pod_infos}'")
        # Ensure pod_infos is a list of PodInfo objects
        return BeamPod(pod_infos=pod_infos, k8s=self.k8s, namespace=self.namespace)

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

