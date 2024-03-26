from .k8s import BeamK8S
from ..core import Processor
from ..logger import beam_logger as logger
from kubernetes import client
from kubernetes.client.rest import ApiException

from ..utils import lazy_property


class BeamPod(Processor):
    def __init__(self, pod_infos=None, namespace=None, k8s=None,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pod_infos = pod_infos
        self.k8s = k8s
        self.namespace = namespace
        # self.refresh_pod_info()

    # @classmethod
    # def from_deployment(cls, deployment, k8s, *args, **kwargs):
    #     return cls(deployment.metadata.name, k8s, *args, **kwargs)

    @classmethod
    def from_existing_pod(cls, pod_name, api_url=None, api_token=None, namespace=None,
                          project_name=None, use_scc=None, scc_name=None, *args, **kwargs):
        k8s = BeamK8S(
            api_url=api_url,
            api_token=api_token,
            project_name=project_name,
            namespace=namespace,
        )
        return cls(pod_name, k8s, *args, **kwargs)

    @lazy_property
    def pod_info(self):
        """Get current pod information."""
        if not self.k8s.get_pod_info(self.pod_name, self.namespace):
            self.refresh_pod_info()
        return self.k8s.get_pod_info(self.pod_name, self.namespace)

    def execute(self, command, **kwargs):
        # Fetch pod information using the get_pod_info method
        pod_info = self.k8s.get_pod_info(self.pod_name, self.namespace)
        if self.pod_info is None:
            logger.error("Failed to fetch pod information")
            return None

        # Assuming pod_info is an object with attributes 'namespace' and 'name'
        namespace = self.pod_info.metadata.namespace
        pod_name = self.pod_info.metadata.name

        # Execute command
        output = self.k8s.execute_command_in_pod(namespace, pod_name, command)  # No split() needed
        logger.info(f"Command output: {output}")

        return output

    # def refresh_pod_info(self):
    #     """Refresh the stored pod information."""
    #     _pod_info = self.k8s.get_pod_info(self.pod_infos, self.namespace)

    @property
    def pod_status(self):
        """Get the current status of the pod."""
        return self.pod_info.status.phase if self.pod_info else "Unknown"

    def get_pod_status(self):
        # Example method to get the status of all pods
        return [pod_info.status for pod_info in self.pod_infos]

    def get_logs(self, **kwargs):
        """Get logs from the pod."""
        return self.k8s.get_pod_logs(self.pod_name, self.namespace, **kwargs)

    def get_pod_resources(self):
        """Get resource usage of the pod."""
        # This might involve metrics API or similar, depending on how you implement it in BeamK8S
        return self.k8s.get_pod_resources(self.pod_name, self.namespace)

    def stop(self):
        """Stop the pod."""
        # Implement stopping the pod, possibly by scaling down the deployment or similar
        self.k8s.stop_pod(self.pod_name, self.namespace)

    def start(self):
        """Start the pod."""
        # Implement starting the pod, possibly by scaling up the deployment or similar
        self.k8s.start_pod(self.pod_name, self.namespace)
