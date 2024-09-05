from ..processor import Processor
from ..logging import beam_logger as logger
from .dataclasses import PodInfos, PodMetadata
from ..utils import lazy_property


class BeamPod(Processor):
    def __init__(self, pod_infos=None, namespace=None, k8s=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pod_infos = pod_infos  # A list of PodInfos objects
        self.namespace = namespace
        self.k8s = k8s

    @staticmethod
    def extract_pod_info(pod):
        # Ensure pod includes metadata with a name
        metadata = PodMetadata(name=pod.metadata.name)
        return PodInfos(raw_pod_data=pod.to_dict(), metadata=metadata)

    @lazy_property
    def port_mapping(self):
        # TODO: return a dictionary of port forwards
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
                current_pod_name = pod_info.raw_pod_data['metadata']['name']
                output = self.k8s.execute_command_in_pod(self.namespace, current_pod_name, command)
                outputs.append((current_pod_name, output))
                logger.info(f"Command output for {current_pod_name}: {output}")

        return outputs

    def get_logs(self, **kwargs):
        """Get logs from each pod."""
        logs = []
        for pod_info in self.pod_infos:
            pod_name = pod_info.raw_pod_data['metadata']['name']
            log = self.k8s.get_pod_logs(pod_name, self.namespace, **kwargs)
            logs.append((pod_name, log))
        return logs

    def get_pod_resources(self):
        """Get resource usage for each pod."""
        resources = []
        for pod_info in self.pod_infos:
            pod_name = pod_info.raw_pod_data['metadata']['name']
            resource_usage = self.k8s.get_pod_resources(pod_name, self.namespace)
            resources.append((pod_name, resource_usage))
        return resources

    def stop(self):
        """Stop each pod."""
        for pod_info in self.pod_infos:
            pod_name = pod_info.raw_pod_data['metadata']['name']
            self.k8s.stop_pod(pod_name, self.namespace)

    def start(self):
        """Start each pod."""
        for pod_info in self.pod_infos:
            pod_name = pod_info.raw_pod_data['metadata']['name']
            self.k8s.start_pod(pod_name, self.namespace)

    def get_pod_status(self):
        """Get the status of each pod."""
        statuses = []
        for pod_info in self.pod_infos:
            pod_name = pod_info.raw_pod_data['metadata']['name']
            try:
                pod_data = self.k8s.get_pod_info(pod_name, self.namespace)
                logger.debug(f"Pod data for {pod_name}: {pod_data}")
                status = pod_data.status.phase
                statuses.append((pod_name, status))
            except Exception as e:
                logger.error(f"Error retrieving status for pod {pod_name}: {str(e)}")
        return statuses

    def to_dict(self):
        return {
            "pod_name": self.pod_infos[0].raw_pod_data['metadata']['name'],
            "pod_ip": self.pod_infos[0].raw_pod_data['status']['pod_ip'],
            "namespace": self.namespace,
            "services": [svc.to_dict() for svc in self.k8s.list_services(self.namespace)],
            "routes": [route.to_dict() for route in self.k8s.list_routes(self.namespace)],
        }
