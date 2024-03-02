from .beamk8s import BeamK8S
from ..logger import beam_logger as logger
from kubernetes import client
from kubernetes.client.rest import ApiException


class BeamPod(BeamK8S):
    def __init__(self, deployment, api_url, api_token, *args, **kwargs):
        super().__init__(api_url=api_url, api_token=api_token, *args, **kwargs)  # Initialize BeamK8S part of BeamPod
        self.deployment = deployment  # Store deployment info
        self.deployment_name = deployment.metadata.name

    def delete_deployment(self):
        # Use the inherited apps_v1_api to delete the deployment
        try:
            self.apps_v1_api.delete_namespaced_deployment(
                name=self.deployment['metadata']['name'],
                namespace=self.deployment['metadata']['namespace'],
                body=client.V1DeleteOptions()
            )
            logger.info(f"Deleted deployment '{self.deployment['metadata']['name']}' "
                        f"from namespace '{self.deployment['metadata']['namespace']}'.")
        except ApiException as e:
            logger.error(f"Error deleting deployment '{self.deployment['metadata']['name']}': {e}")

    def list_pods(self):
        label_selector = f"app={self.deployment_name}"
        pods = self.core_v1_api.list_namespaced_pod(namespace=self.namespace, label_selector=label_selector)
        for pod in pods.items:
            print(f"Pod Name: {pod.metadata.name}, Pod Status: {pod.status.phase}")

    def query_available_resources(self):
        total_resources = {'cpu': '0', 'memory': '0', 'nvidia.com/gpu': '0', 'amd.com/gpu': '0', 'storage': '0Gi'}
        node_list = self.core_v1_api.list_node()

        # Summing up the allocatable CPU, memory, and GPU resources from each node
        for node in node_list.items:
            for key, quantity in node.status.allocatable.items():
                if key in ['cpu', 'memory', 'nvidia.com/gpu', 'amd.com/gpu']:
                    if quantity.endswith('m'):  # Handle milliCPU
                        total_resources[key] = str(
                            int(total_resources.get(key, '0')) + int(float(quantity.rstrip('m')) / 1000))
                    else:
                        total_resources[key] = str(
                            int(total_resources.get(key, '0')) + int(quantity.strip('Ki')))

        # Summing up the storage requests for all PVCs in the namespace
        pvc_list = self.core_v1_api.list_namespaced_persistent_volume_claim(namespace=self.namespace)
        for pvc in pvc_list.items:
            for key, quantity in pvc.spec.resources.requests.items():
                if key == 'storage':
                    total_resources['storage'] = str(
                        int(total_resources['storage'].strip('Gi')) + int(quantity.strip('Gi'))) + 'Gi'

        # Remove resources with a count of '0'
        total_resources = {key: value for key, value in total_resources.items() if value != '0'}

        logger.info(f"Total Available Resources in the Namespace '{self.namespace}': {total_resources}")
        return total_resources

