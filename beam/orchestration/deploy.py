from logging import debug
from multiprocessing.managers import Namespace

from numpy.f2py.crackfortran import true_intent_list

from .utils import convert_datetimes
from ..base import BeamBase
from .pod import BeamPod, PodInfos
from kubernetes import client
from kubernetes.client.rest import ApiException
from ..logging import beam_logger as logger
from .dataclasses import *
from ..resources import resource
from .config import K8SConfig
from .utils import ensure_rfc1123_compliance
from datetime import datetime
import json


class BeamDeploy(BeamBase):

    def __init__(self, hparams, k8s, *args, **kwargs):

        super().__init__(hparams, *args, _config_scheme=K8SConfig, **kwargs)

        security_context_config = SecurityContextConfig(**self.get_hparam('security_context_config', {}))
        memory_storage_configs = [MemoryStorageConfig(**v) for v in self.get_hparam('memory_storage_configs', [])]
        service_configs = [ServiceConfig(**v) for v in self.get_hparam('service_configs', [])]
        storage_configs = [StorageConfig(**v) for v in self.get_hparam('storage_configs', [])]
        # ray_ports_configs = [RayPortsConfig(**v) for v in self.get_hparam('ray_ports_configs', [])]
        user_idm_configs = [UserIdmConfig(**v) for v in self.get_hparam('user_idm_configs', [])]
        restart_policy_configs = RestartPolicyConfig(**self.get_hparam('restart_policy_configs', []))

        command = self.get_hparam('command', None)
        if command:
            command = CommandConfig(**command)
        else:
            command = None

        debug_sleep = self.get_hparam('debug_sleep')
        if debug_sleep:
            # If debug_sleep is True, set both executable and arguments directly
            command = CommandConfig(executable="/bin/bash", arguments=["-c", "sleep infinity"])
        # else:
        #     # Retrieve from config if not in debug mode
        #     # command = CommandConfig(**self.get_hparam('command'))
        #     command = None

        self.entrypoint_args = self.get_hparam('entrypoint_args') or []
        self.entrypoint_envs = self.get_hparam('entrypoint_envs') or {}
        # self.check_project_exists = self.get_hparam('check_project_exists')
        self.project_name = self.get_hparam('project_name')
        self.create_service_account = self.get_hparam('create_service_account')
        self.namespace = self.project_name
        self.replicas = self.get_hparam('replicas')
        self.labels = self.get_hparam('labels')
        self.image_name = self.get_hparam('image_name')
        self.cron_job_name = self.get_hparam('cron_job_name')
        self.container_name = self.get_hparam('container_name')
        self.job_name = self.get_hparam('job_name')

        # Retrieve and ensure RFC 1123 compliance for deployment name
        deployment_name = self.get_hparam('deployment_name')
        self.deployment_name = ensure_rfc1123_compliance(deployment_name)

        self.job_schedule = self.get_hparam('job_schedule')
        self.service_type = self.get_hparam('service_type')
        self.service_account_name = f"{self.deployment_name}svc"
        self.use_scc = self.get_hparam('use_scc')
        self.use_node_selector = self.get_hparam('use_node_selector')
        self.node_selector = self.get_hparam('node_selector')
        self.scc_name = self.get_hparam('scc_name') if self.use_scc else None
        self.cpu_requests = self.get_hparam('cpu_requests')
        self.cpu_limits = self.get_hparam('cpu_limits')
        self.memory_requests = self.get_hparam('memory_requests')
        self.memory_limits = self.get_hparam('memory_limits')
        self.use_gpu = self.get_hparam('use_gpu')
        self.gpu_requests = self.get_hparam('gpu_requests')
        self.gpu_limits = self.get_hparam('gpu_limits')
        self.pod_info_state = self.get_hparam('pod_info_state') or []
        self.beam_pod_instances = self.get_hparam('beam_pod_instances') or []
        # self.deployment_state = self.get_hparam('deployment_state') or {}
        # self.cluster_info = self.get_hparam('cluster_info') or []

        self.k8s = k8s
        self.command = command
        self.service_configs = service_configs or []
        # self.ray_ports_configs = ray_ports_configs or RayPortsConfig()
        self.memory_storage_configs = memory_storage_configs or []
        self.storage_configs = storage_configs or []
        self.user_idm_configs = user_idm_configs or []
        self.security_context_config = security_context_config or []
        self.restart_policy_configs = restart_policy_configs or {}

    def launch(self, replicas=None):


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
                            namespace=self.namespace,
                            storage_class_name = storage_config.storage_class_name
                        )
                    else:
                        logger.info(f"Skipping PVC creation for: {storage_config.pvc_name} as create_pvc is False")

        if self.user_idm_configs:
            self.k8s.create_role_bindings(self.user_idm_configs)

        if self.use_scc is True:
            self.k8s.add_scc_to_service_account(self.service_account_name, self.namespace, self.scc_name)

        extracted_ports = [svc_config.port for svc_config in self.service_configs]

        deployment = self.k8s.create_deployment(
            image_name=self.image_name,
            command=self.command,
            labels=self.labels,
            deployment_name=self.deployment_name,
            namespace=self.namespace,
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

        pod_infos = self.k8s.apply_deployment(deployment, namespace=self.namespace)

        logger.debug(f"pod_infos type: {type(pod_infos)}")

        logger.debug(f"pod_infos content: {pod_infos}")

        self.pod_info_state = [BeamPod.extract_pod_info(self.k8s.get_pod_info(pod.name, self.namespace))
                               for pod in pod_infos]

        # self.beam_pod_instances = []
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
            logger.error("Failed to apply deployment or no pods were returned.")
            return None

        for pod_instance in self.beam_pod_instances:
            pod_suffix = (f"{self.deployment_name}-"
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
                rs_env_vars.append({'name': f"KUBERNETES_{svc_config.service_name.upper()}_URL", 'value': service_details['url']})
                rs_env_vars.append({'name': f"KUBERNETES_{svc_config.service_name.upper()}_PORT", 'value': str(service_details['ports'][0])})
                rs_env_vars.append({'name': f"KUBERNETES_{svc_config.service_name.upper()}_NAME", 'value': service_details['name']})

                # Create routes and ingress if configured
                if svc_config.create_route:
                    route_details = self.k8s.create_route(
                        service_name=service_name,
                        namespace=self.namespace,
                        protocol=svc_config.route_protocol,
                        port=svc_config.port,
                        annotations=svc_config.annotations,
                    )
                    rs_env_vars.append({'name': f"KUBERNETES_{svc_config.service_name.upper()}_ROUTE_NAME", 'value': route_details['name']})
                    rs_env_vars.append({'name': f"KUBERNETES_{svc_config.service_name.upper()}_ROUTE_HOST", 'value': route_details['host']})
                if svc_config.create_ingress:
                    ingress_details = self.k8s.create_ingress(
                        service_configs=[svc_config],
                    )
            rs_env_vars.append({'name': f"PLATFORM_ENGINE", 'value': 'Kuberenetes'})
            # self.update_config_maps_rs_env_vars(self.deployment_name, self.namespace, rs_env_vars) #todo: uncomment this line to inject vars into pods

        return self.beam_pod_instances if len(self.beam_pod_instances) > 1 else self.beam_pod_instances[0]

    def launch_job(self):

        # Delegate Job creation to the k8s class
        job = self.k8s.create_job(
            namespace=self.namespace,
            job_name=self.job_name,
            image_name=self.image_name,
            container_name=self.container_name,
            command=self.command,
            entrypoint_args=self.entrypoint_args,
            entrypoint_envs=self.entrypoint_envs,
            cpu_requests=self.cpu_requests,
            cpu_limits=self.cpu_limits,
            memory_requests=self.memory_requests,
            memory_limits=self.memory_limits,
            use_node_selector=self.use_node_selector,
            node_selector=self.node_selector,
            use_gpu=self.use_gpu,
            gpu_requests=self.gpu_requests,
            gpu_limits=self.gpu_limits,
            labels=self.labels,
            storage_configs=self.storage_configs,
            security_context_config=self.security_context_config,
            restart_policy_configs=self.restart_policy_configs
        )

        logger.info(f"Job '{self.job_name}' created successfully.")

        return job

    def launch_cron_job(self):
        if not self.job_schedule:
            logger.error("CronJob schedule not provided.")
            return


        # Delegate CronJob creation to k8s class
        cronjob = self.k8s.create_cron_job(config=self.hparams)

        logger.info(
            f"CronJob '{self.cron_job_name}' created. Pods will be scheduled according to '{self.job_schedule}'.")

        return cronjob

    # TODO: use process_pod_infos in regular deployment as well
    def process_pod_infos(self, pod_infos):
        # Handle cases where no pods might be immediately available
        if not pod_infos:
            logger.warning("No pods found for the given CronJob. They may not have been scheduled yet.")
            return None

        self.pod_info_state = [BeamPod.extract_pod_info(self.k8s.get_pod_info(pod.name, self.namespace))
                               for pod in pod_infos]
        self.beam_pod_instances = [
            BeamPod(pod_infos=[BeamPod.extract_pod_info(self.k8s.get_pod_info(pod.name, self.namespace))],
                    namespace=self.namespace, k8s=self.k8s)
            for pod in pod_infos]
        return self.beam_pod_instances if len(self.beam_pod_instances) > 1 else self.beam_pod_instances[0]

    def update_config_maps_rs_env_vars(self, deployment_name, namespace, rs_env_vars):
        # Prepare ConfigMap data
        config_map_name = f"{deployment_name}-var-config"
        config_data = {var['name']: var['value'] for var in rs_env_vars}

        # Create or update ConfigMap
        try:
            existing_cm = self.k8s.core_v1_api.read_namespaced_config_map(config_map_name, namespace)
            existing_cm.data.update(config_data)
            # TODO: updating ConfigMap causing the pod to restart -
            #  new a way to assosiate new pod names to existing routes/services
            self.k8s.core_v1_api.replace_namespaced_config_map(config_map_name, namespace, existing_cm)
            logger.info(f"Updated ConfigMap {config_map_name} in namespace {namespace}.")
        except ApiException as e:
            if e.status == 404:
                # Create new ConfigMap if not exists
                new_cm = client.V1ConfigMap(metadata=client.V1ObjectMeta(name=config_map_name), data=config_data)
                self.k8s.core_v1_api.create_namespaced_config_map(namespace, new_cm)
                logger.info(f"Created new ConfigMap {config_map_name} in namespace {namespace}.")
            else:
                raise

        # Update deployment to use this ConfigMap
        deployment = self.k8s.apps_v1_api.read_namespaced_deployment(deployment_name, namespace)
        for container in deployment.spec.template.spec.containers:
            container.env_from = [
                client.V1EnvFromSource(config_map_ref=client.V1ConfigMapEnvSource(name=config_map_name))]

        self.k8s.apps_v1_api.replace_namespaced_deployment(deployment_name, namespace, deployment)
        logger.info(f"Deployment {deployment_name} updated to use ConfigMap in namespace {namespace}.")

    def extract_ports(self):
        extracted_ports = [svc_config.port for svc_config in self.service_configs]
        return extracted_ports

    def generate_beam_pod(self, pod_infos):
        logger.info(f"Generating BeamPod for pods: '{pod_infos}'")
        # Ensure pod_infos is a list of PodInfo objects
        return BeamPod(pod_infos=pod_infos, k8s=self.k8s, namespace=self.namespace)

    def delete_deployment(self):
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

    @property
    def cluster_info(self):
        services_info = self.k8s.get_services_info(self.namespace)
        routes_info = self.k8s.get_routes_info(self.namespace)
        if self.beam_pod_instances and self.beam_pod_instances[0].pod_infos:
            host_ip = self.beam_pod_instances[0].pod_infos[0].raw_pod_data['status'].get('host_ip', 'Host IP NONE')
        else:
            host_ip = 'Host IP NONE'  # Default if no pods are available
        # host_ip = self.beam_pod_instances[0].pod_infos[0].raw_pod_data['status'].get('host_ip') or 'Host IP NONE'
        service_info_lines = []
        route_info_lines = []

        if services_info:
            for service_info in services_info:
                if 'node_port' in service_info:
                    service_line = f"Service: {service_info['service_name']}   | Host IP: {host_ip} | NodePort: {service_info['node_port']} "
                    # service_line = f"Service: {service_info['service_name']} | Cluster IP: {service_info['cluster_ip']} | Port: {service_info['port']} | Host IP: {host_ip} | NodePort: {service_info['node_port']} | Ingress Access"
                else:
                    service_line = f"Service: {service_info['service_name']}  | Port: {service_info['port']}"
                    # service_line = f"Service: {service_info['service_name']} | Cluster IP: {service_info['cluster_ip']} | Port: {service_info['port']}"
                service_info_lines.append(service_line)

        if routes_info:
            # route_info_lines = [f"Route link: <a href='http://{route_info['host']}</a>" for
             route_info_lines = [f"Route link: <a href='http://{route_info['host']}'>{route_info['host']}</a>" for
                                route_info in routes_info]

        # cluster_info = "<br>".join(service_info_lines + route_info_lines)
        cluster_info = "\n\n".join(service_info_lines) + "\n\n" + "\n\n".join(route_info_lines)
        resource("deployment_state.yaml").write(cluster_info)

        # Write the formatted lines to a file
        with open("cluster_info.txt", "w") as file:
            file.write(cluster_info.replace("<br>", "\n"))

        cluster_info = cluster_info.replace("\n", "<br>")
        return cluster_info

    @property
    def pods_state(self):
        state = {
            "pod_info_state": [pod_info.raw_pod_data for pod_info in self.pod_info_state],
            "beam_pod_instances": [pod_info.raw_pod_data for beam_pod_instance in
                                   self.beam_pod_instances for pod_info in beam_pod_instance.pod_infos],
        }
        state = convert_datetimes(state)
        return state

    @property
    def deployment_state(self) -> dict:
        state = {**self.pods_state, "cluster_info": self.cluster_info}
        return state
