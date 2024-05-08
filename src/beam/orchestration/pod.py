from .k8s import BeamK8S
from ..processor import Processor
from ..logger import beam_logger as logger
from kubernetes import client
from kubernetes.client.rest import ApiException

from ..utils import lazy_property


class BeamPod(Processor):
    def __init__(self, pod_infos=None, namespace=None, k8s=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pod_infos = pod_infos  # A list of pod_info objects
        self.namespace = namespace
        self.k8s = k8s

    # def execute(self, command, **kwargs):
    #     """Execute a command in each pod."""
    #     outputs = []
    #     for pod_info in self.pod_infos:
    #         pod_name = pod_info.metadata.name
    #         output = self.k8s.execute_command_in_pod(self.namespace, pod_name, command)
    #         outputs.append((pod_name, output))
    #         logger.info(f"Command output for {pod_name}: {output}")
    #     return outputs

    @lazy_property
    def port_mapping(self):
        #TODO: return a dictionary of port forwards
        raise NotImplementedError

    def execute(self, command, pod_name=None, **kwargs):
        """Execute a command on a specific pod or on each pod if no pod name is provided."""
        outputs = []

        if pod_name:
            # Execute the command only on the specified pod
            output = self.k8s.execute_command_in_pod(self.namespace, pod_name, command)
            outputs.append((pod_name, output))
            logger.info(f"Command output for {pod_name}: {output}")
        else:
            # Execute the command on each pod
            for pod_info in self.pod_infos:
                current_pod_name = pod_info.metadata.name
                output = self.k8s.execute_command_in_pod(self.namespace, current_pod_name, command)
                outputs.append((current_pod_name, output))
                logger.info(f"Command output for {current_pod_name}: {output}")

        return outputs

    def get_logs(self, **kwargs):
        """Get logs from each pod."""
        logs = []
        for pod_info in self.pod_infos:
            pod_name = pod_info.metadata.name
            log = self.k8s.get_pod_logs(pod_name, self.namespace, **kwargs)
            logs.append((pod_name, log))
        return logs

    def get_pod_resources(self):
        """Get resource usage for each pod."""
        resources = []
        for pod_info in self.pod_infos:
            pod_name = pod_info.metadata.name
            resource_usage = self.k8s.get_pod_resources(pod_name, self.namespace)
            resources.append((pod_name, resource_usage))
        return resources

    def stop(self):
        """Stop each pod."""
        for pod_info in self.pod_infos:
            pod_name = pod_info.metadata.name
            self.k8s.stop_pod(pod_name, self.namespace)

    def start(self):
        """Start each pod."""
        for pod_info in self.pod_infos:
            pod_name = pod_info.metadata.name
            self.k8s.start_pod(pod_name, self.namespace)

    def get_pod_status(self):
        """Get the status of each pod."""
        statuses = []
        for pod_info in self.pod_infos:
            pod_name = pod_info.metadata.name
            status = self.k8s.get_pod_info(pod_name, self.namespace).status.phase
            statuses.append((pod_name, status))
        return statuses

# before change 27.03.24
# class BeamPod(Processor):
#     def __init__(self, pod_info=None, namespace=None, k8s=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.pod_info = pod_info
#         self.namespace = namespace
#         self.k8s = k8s
#         self.pod_name = pod_info.name if pod_info else None
#
#     def set_pod_name_from_infos(self):
#         if self.pod_infos and isinstance(self.pod_infos, list) and self.pod_infos[0]:
#             # Check if the first item has a 'name' attribute
#             if hasattr(self.pod_infos[0], 'name'):
#                 self.pod_name = self.pod_infos[0].name
#             else:
#                 logger.error("The first pod_info does not have a 'name' attribute.")
#         else:
#             logger.error("pod_infos is empty or not a list.")
#     # @classmethod
#     # def from_deployment(cls, deployment, k8s, *args, **kwargs):
#     #     return cls(deployment.metadata.name, k8s, *args, **kwargs)
#
#     @classmethod
#     def from_existing_pod(cls, pod_name, api_url=None, api_token=None, namespace=None,
#                           project_name=None, use_scc=None, scc_name=None, *args, **kwargs):
#         k8s = BeamK8S(
#             api_url=api_url,
#             api_token=api_token,
#             project_name=project_name,
#             namespace=namespace,
#         )
#         return cls(pod_name, k8s, *args, **kwargs)
#
#     @lazy_property
#     def pod_info(self):
#         """Get current pod information."""
#         if not self.k8s.get_pod_info(self.pod_name, self.namespace):
#             self.refresh_pod_info()
#         return self.k8s.get_pod_info(self.pod_name, self.namespace)
#
#     def execute(self, command, **kwargs):
#         if not self.pod_name:
#             logger.error("Pod name is not set. Cannot execute command.")
#             return None
#
#         # Fetch pod information using the get_pod_info method
#         pod_info = self.k8s.get_pod_info(self.pod_name, self.namespace)
#         if not pod_info:
#             logger.error("Failed to fetch pod information")
#             return None
#
#         # Execute command in the pod
#         output = self.k8s.execute_command_in_pod(self.namespace, self.pod_name, command)
#         logger.info(f"Command output: {output}")
#
#         return output
#
#     @property
#     def pod_status(self):
#         """Get the current status of the pod."""
#         return self.pod_info.status.phase if self.pod_info else "Unknown"
#
#     def get_pod_status(self):
#         # Example method to get the status of all pods
#         return [pod_info.status for pod_info in self.pod_infos]
#
#     def get_logs(self, **kwargs):
#         """Get logs from the pod."""
#         return self.k8s.get_pod_logs(self.pod_name, self.namespace, **kwargs)
#
#     def get_pod_resources(self):
#         """Get resource usage of the pod."""
#         # This might involve metrics API or similar, depending on how you implement it in BeamK8S
#         return self.k8s.get_pod_resources(self.pod_name, self.namespace)
#
#     def stop(self):
#         """Stop the pod."""
#         # Implement stopping the pod, possibly by scaling down the deployment or similar
#         self.k8s.stop_pod(self.pod_name, self.namespace)
#
#     def start(self):
#         """Start the pod."""
#         # Implement starting the pod, possibly by scaling up the deployment or similar
#         self.k8s.start_pod(self.pod_name, self.namespace)
