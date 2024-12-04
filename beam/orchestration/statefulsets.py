from ..base import BeamBase
from .k8s import BeamK8S
from .pod import BeamPod
from ..logging import beam_logger as logger
from .config import StatefulSetConfig
from .utils import ensure_rfc1123_compliance
from kubernetes.client.rest import ApiException
from .dataclasses import *


class BeamStatefulSet(BeamBase):
    """
    Handles StatefulSet deployment, monitoring, logs, and interaction via the k8s API.
    """

    def __init__(self, hparams, k8s, *args, **kwargs):
        super().__init__(hparams, *args, _config_scheme=StatefulSetConfig, **kwargs)

        # Validate and assign the k8s object
        if not isinstance(k8s, BeamK8S):
            raise ValueError("The 'k8s' parameter must be an instance of BeamK8S.")
        self.k8s = k8s

        # Process configurations
        self.security_context_config = SecurityContextConfig(**self.get_hparam('security_context_config', {}))
        self.memory_storage_configs = [MemoryStorageConfig(**v) for v in self.get_hparam('memory_storage_configs', [])]
        self.service_configs = [ServiceConfig(**v) for v in self.get_hparam('service_configs', [])]
        self.storage_configs = [StorageConfig(**v) for v in self.get_hparam('storage_configs', [])]
        self.user_idm_configs = [UserIdmConfig(**v) for v in self.get_hparam('user_idm_configs', [])]
        self.restart_policy_configs = RestartPolicyConfig(**self.get_hparam('restart_policy_configs', []))

        # Command configuration
        command = self.get_hparam('command', None)
        if command:
            command = CommandConfig(**command).as_list()
        else:
            command = None
        debug_sleep = self.get_hparam('debug_sleep')
        if debug_sleep:
            # If debug_sleep is True, set both executable and arguments directly
            command = CommandConfig(executable="/bin/bash", arguments=["-c", "sleep infinity"])
        self.command = command

        # Instance attributes
        self.project_name = self.get_hparam('project_name', 'default-project')
        self.create_service_account = self.get_hparam('create_service_account')
        self.namespace = self.project_name
        self.statefulset_name = ensure_rfc1123_compliance(self.get_hparam('statefulset_name', 'default-statefulset'))
        self.replicas = self.get_hparam('replicas', 1)
        self.labels = self.get_hparam('labels', {})
        self.image_name = self.get_hparam('image_name', 'default-image:latest')
        self.entrypoint_args = self.get_hparam('entrypoint_args', [])
        self.entrypoint_envs = self.get_hparam('entrypoint_envs', {})
        self.cpu_requests = self.get_hparam('cpu_requests', '500m')
        self.cpu_limits = self.get_hparam('cpu_limits', '1000m')
        self.memory_requests = self.get_hparam('memory_requests', '512Mi')
        self.memory_limits = self.get_hparam('memory_limits', '1Gi')
        self.use_gpu = self.get_hparam('use_gpu')
        self.gpu_requests = self.get_hparam('gpu_requests')
        self.gpu_limits = self.get_hparam('gpu_limits')

        self.use_node_selector = self.get_hparam('use_node_selector', False)
        self.use_scc = self.get_hparam('use_scc')
        self.scc_name = self.get_hparam('scc_name') if self.use_scc else None
        self.node_selector = self.get_hparam('node_selector', {})
        self.volume_claims = self.get_hparam('volume_claims', [])
        self.update_strategy = self.get_hparam('update_strategy', 'RollingUpdate')
        self.pod_management_policy = self.get_hparam('pod_management_policy', 'OrderedReady')
        self.pod_info_state = self.get_hparam('pod_info_state') or []
        self.beam_pod_instances = self.get_hparam('beam_pod_instances') or []


        # Additional attributes for statefulset handling
        self.service_name = self.get_hparam('service_name', f"{self.statefulset_name}-service")
        self.service_port = self.get_hparam('service_port', 80)
        self.service_account_name = f"{self.statefulset_name}svc"

    def delete_statefulset(self):
        """
        Delete the StatefulSet.
        """
        try:
            self.k8s.delete_statefulsets_by_name(self.statefulset_name, self.namespace)
            logger.info(f"StatefulSet {self.statefulset_name} deleted successfully.")
        except Exception as e:
            logger.error(f"Error occurred while deleting the StatefulSet: {str(e)}")

    def monitor_statefulset(self):
        """
        Monitor the StatefulSet status and retrieve logs from associated pods.
        """
        try:
            # Monitor the StatefulSet's status
            self.k8s.monitor_statefulset(statefulset_name=self.statefulset_name, namespace=self.namespace)

            # Fetch logs from all pods in the StatefulSet
            pods = self.k8s.get_pods_by_label({'app': self.statefulset_name}, self.namespace)
            if pods:
                for pod in pods:
                    logs = self.k8s.get_pod_logs(pod.metadata.name, namespace=self.namespace)
                    if logs:
                        logger.info(f"Logs for Pod '{pod.metadata.name}' in StatefulSet "
                                    f"'{self.statefulset_name}':\n{logs}")
        except Exception as e:
            logger.error(f"Failed to monitor StatefulSet '{self.statefulset_name}': {str(e)}")

    def get_statefulset_logs(self):
        """
        Retrieve logs from all pods associated with the StatefulSet.
        """
        try:
            return self.k8s.get_logs_for_statefulset(
                statefulset_name=self.statefulset_name,
                namespace=self.namespace
            )
        except Exception as e:
            logger.error(f"Error retrieving logs for StatefulSet '{self.statefulset_name}': {str(e)}")
            return None

    def launch(self, replicas=None):
        """
        Launch a StatefulSet using the k8s class.
        """
        # Use the replicas passed as an argument, or fall back to the instance's default
        if replicas is None:
            replicas = self.replicas

        # if self.check_project_exists is True:
        self.k8s.create_project(self.namespace)

        if self.create_service_account:
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

        if self.user_idm_configs:
            self.k8s.create_role_bindings(self.user_idm_configs)

        if self.use_scc is True:
            self.k8s.add_scc_to_service_account(self.service_account_name, self.namespace, self.scc_name)

        extracted_ports = [svc_config.port for svc_config in self.service_configs]

        statefulset = self.k8s.create_statefulset(
            image_name=self.image_name,
            command=self.command,
            labels=self.labels,
            namespace=self.namespace,
            statefulset_name=self.statefulset_name,
            project_name=self.project_name,
            replicas=replicas,
            ports=extracted_ports,
            create_service_account=self.create_service_account,
            service_account_name=self.service_account_name,
            storage_configs=self.storage_configs,
            memory_storage_configs=self.memory_storage_configs,
            use_node_selector=self.use_node_selector,
            node_selector=self.node_selector,
            cpu_requests=self.cpu_requests,
            cpu_limits=self.cpu_limits,
            memory_requests=self.memory_requests,
            memory_limits=self.memory_limits,
            use_gpu=self.use_gpu,
            gpu_requests=self.gpu_requests,
            gpu_limits=self.gpu_limits,
            security_context_config=self.security_context_config,
            restart_policy_configs=self.restart_policy_configs,
            entrypoint_args=self.entrypoint_args,
            entrypoint_envs=self.entrypoint_envs,
        )

        pod_infos = self.k8s.apply_statefulset(statefulset, namespace=self.namespace)
        logger.debug(f"pod_infos type: {type(pod_infos)}")

        logger.debug(f"pod_infos content: {pod_infos}")
        self.pod_info_state = [BeamPod.extract_pod_info(self.k8s.get_pod_info(pod.name, self.namespace))
                               for pod in pod_infos]

        self.beam_pod_instances = [] if isinstance(pod_infos, list) else [pod_infos]

        if isinstance(pod_infos, list) and pod_infos:
            for pod_info in pod_infos:
                pod_name = pod_info.name
                if pod_name:
                    actual_pod_info = self.k8s.get_pod_info(pod_name, self.namespace)
                    beam_pod_instance = BeamPod(pod_infos=[BeamPod.extract_pod_info(actual_pod_info)],
                                                namespace=self.namespace, k8s=self.k8s, replicas=self.replicas)
                    self.beam_pod_instances.append(beam_pod_instance)
                else:
                    logger.warning("PodInfo object does not have a 'name' attribute.")
        # If pod_infos is not a list but a single object with a name attribute
        elif pod_infos and hasattr(pod_infos, 'name'):
            pod_name = pod_infos.name
            print(f"Single pod_info with pod_name: {pod_name}")

            actual_pod_info = self.k8s.get_pod_info(pod_name, self.namespace)
            print(f"Fetched actual_pod_info for pod_name '{pod_name}': {actual_pod_info}")

            # Directly return the single BeamPod instance
            return BeamPod(pod_infos=[BeamPod.extract_pod_info(actual_pod_info)], namespace=self.namespace,
                           k8s=self.k8s)

        # Handle cases where deployment failed or no pods were returned
        if not self.beam_pod_instances:
            logger.error("Failed to apply statefulset or no pods were returned.")
            return None

        for pod_instance in self.beam_pod_instances:
            pod_suffix = (f"{self.statefulset_name}-"
                          f"{pod_instance.pod_infos[0].raw_pod_data['metadata']['name'].split('-')[-1]}")
            rs_env_vars = []

            for svc_config in self.service_configs:
                service_name = f"{svc_config.service_name}-{svc_config.port}-{pod_suffix}"

                service_details = self.k8s.create_service(
                    base_name=service_name,
                    namespace=self.namespace,
                    ports=[svc_config.port],
                    labels=self.labels,
                    service_type=svc_config.service_type
                )
            #    rs_env_vars.append({'name': f"KUBERNETES_{svc_config.service_name.upper()}_URL", 'value': service_details['url']})
            #    rs_env_vars.append({'name': f"KUBERNETES_{svc_config.service_name.upper()}_PORT", 'value': str(service_details['ports'][0])})
            #    rs_env_vars.append({'name': f"KUBERNETES_{svc_config.service_name.upper()}_NAME", 'value': service_details['name']})

                # Create routes and ingress if configured
                if svc_config.create_route:
                    route_details = self.k8s.create_route(
                        service_name=service_name,
                        namespace=self.namespace,
                        protocol=svc_config.route_protocol,
                        port=svc_config.port,
                        route_timeout=svc_config.route_timeout,
                    )
                   # rs_env_vars.append({'name': f"KUBERNETES_{svc_config.service_name.upper()}_ROUTE_NAME", 'value': route_details['name']})
                   # rs_env_vars.append({'name': f"KUBERNETES_{svc_config.service_name.upper()}_ROUTE_HOST", 'value': route_details['host']})
                if svc_config.create_ingress:
                    ingress_details = self.k8s.create_ingress(
                        service_configs=[svc_config],
                    )
            #rs_env_vars.append({'name': f"PLATFORM_ENGINE", 'value': 'Kuberenetes'})
            #self.update_config_maps_rs_env_vars(self.deployment_name, self.namespace, rs_env_vars)

        return self.beam_pod_instances if len(self.beam_pod_instances) > 1 else self.beam_pod_instances[0]
