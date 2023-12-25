
from ..core import Processor
from ..utils import lazy_property

class BeamK8S:

    def __init__(self, api_url=None, api_token=None):
        self.api_token = api_token
        self.api_url = api_url

    def find_available_nodeports(self, count=20, port_range=(30000, 32767)):

        # Get all the services in all namespaces
        services = self.client.list_service_for_all_namespaces()

        # Gather all used nodePorts
        used_ports = set()
        for service in services.items:
            if not len(service.spec.ports):
                continue
            for port in service.spec.ports:
                if port.node_port is not None:
                    used_ports.add(port.node_port)

        # Find available nodePorts
        available_ports = []
        for port in range(port_range[0], port_range[1] + 1):
            if port not in used_ports:
                available_ports.append(port)
                if len(available_ports) >= count:
                    break

        return available_ports


    @lazy_property
    def client(self):

        configuration = Configuration()
        configuration.host = self.api_url
        configuration.verify_ssl = False  # Depends on your SSL setup
        configuration.debug = False
        configuration.api_key = {
            'authorization': f"Bearer {self.api_token}"
        }

        # Create the API client
        return client.CoreV1Api(client.ApiClient(configuration))

    @lazy_property
    def total_gpus(self):
        total_gpus = 0

        # Get all the nodes
        nodes = self.client.list_node().items
        for node in nodes:
            # Check if the node has GPU resources
            if 'nvidia.com/gpu' in node.status.capacity:
                num_gpus = int(node.status.capacity['nvidia.com/gpu'])
                total_gpus += num_gpus
        return total_gpus



