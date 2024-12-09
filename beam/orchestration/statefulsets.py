from ..base import BeamBase
from .k8s import BeamK8S
from ..logging import beam_logger as logger
from .config import StatefulSetConfig
from .utils import ensure_rfc1123_compliance
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
        self.restart_policy_configs = RestartPolicyConfig(**self.get_hparam('restart_policy_configs', []))

        # Command configuration
        command = self.get_hparam('command', None)
        if command:
            self.command = CommandConfig(**command)
        else:
            self.command = None

        # Instance attributes
        self.project_name = self.get_hparam('project_name', 'default-project')
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
        self.use_node_selector = self.get_hparam('use_node_selector', False)
        self.node_selector = self.get_hparam('node_selector', {})
        self.volume_claims = self.get_hparam('volume_claims', [])
        self.update_strategy = self.get_hparam('update_strategy', 'RollingUpdate')
        self.pod_management_policy = self.get_hparam('pod_management_policy', 'OrderedReady')

        # Additional attributes for deployment handling
        self.service_name = self.get_hparam('service_name', f"{self.statefulset_name}-service")
        self.service_port = self.get_hparam('service_port', 80)

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
            pods = self.k8s.get_pods_by_label({'app': self.statefulset_name}, self.namespace)
            all_logs = {}
            for pod in pods:
                logs = self.k8s.get_pod_logs(pod.metadata.name, namespace=self.namespace)
                all_logs[pod.metadata.name] = logs
            return all_logs
        except Exception as e:
            logger.error(f"Error retrieving logs for StatefulSet '{self.statefulset_name}': {str(e)}")
            return None

    def launch(self, replicas=None):
        """
        Launch a StatefulSet using the k8s class.
        """
        # Use the replicas passed as an argument, or fall back to the instance's default
        replicas = replicas if replicas is not None else self.replicas

        statefulset = self.k8s.create_statefulset(
            namespace=self.namespace,
            statefulset_name=self.statefulset_name,
            image_name=self.image_name,
            command=self.command,
            entrypoint_args=self.entrypoint_args,
            entrypoint_envs=self.entrypoint_envs,
            cpu_requests=self.cpu_requests,
            cpu_limits=self.cpu_limits,
            memory_requests=self.memory_requests,
            memory_limits=self.memory_limits,
            use_node_selector=self.use_node_selector,
            node_selector=self.node_selector,
            labels=self.labels,
            volume_claims=self.volume_claims,
            replicas=replicas,  # Pass the replicas value here
            project_name=self.project_name
        )

        logger.info(f"StatefulSet '{self.statefulset_name}' created successfully in namespace '{self.namespace}'.")
        return statefulset
