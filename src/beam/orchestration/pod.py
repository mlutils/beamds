from .k8s import BeamK8S
from ..core import Processor
from ..logger import beam_logger as logger
from kubernetes import client
from kubernetes.client.rest import ApiException


class BeamPod(Processor):
    def __init__(self, pod_info, deployment, k8s, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k8s = k8s
        self._pod_info = pod_info

    def execute(self, command, **kwargs):
        namespace = self._pod_info['namespace']
        pod_name = self._pod_info['name']
        # You might need to adjust how you get namespace and pod_name based on your pod_info structure

        # Execute command
        output = self.k8s.execute_command_in_pod(namespace, pod_name, command.split())
        logger.info(f"Command output: {output}")

        return output

    def get_logs(self, **kwargs):
        # Get logs from the pod
        pass

    @property
    def pod_status(self):
        # Get pod status
        return self.k8s.deployment.status

    @property
    def pod_info(self):
        # Get pod info
        return self._pod_info

    def get_pod_resources(self, **kwargs):
        # Get pod resources
        pass

    def stop(self, **kwargs):
        # Stop the pod
        pass

    def start(self, **kwargs):
        # Start the pod
        pass

    def restart(self, **kwargs):
        # Restart the pod
        pass

    @pod_info.setter
    def pod_info(self, value):
        self._pod_info = value
