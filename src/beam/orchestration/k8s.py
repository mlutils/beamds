from ..path import beam_path
from ..core import Processor
from ..utils import lazy_property
from kubernetes import client, config
from kubernetes.client import Configuration
from kubernetes.client.rest import ApiException
from ..logger import beam_logger as logger
import json


class BeamK8S(Processor):  # processor is a another class and the BeamK8S inherits the method of processor
    """BeamK8S is a class that provides a simple interface to the Kubernetes API."""

    def __init__(self, api_url=None, api_token=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_token = api_token
        self.api_url = api_url
        # Initialize the configuration and API clients
        # self.configuration = self.create_configuration()
        # self.api_client = client.ApiClient(self.configuration)
        # self.core_v1_api = client.CoreV1Api(self.api_client)

    @lazy_property
    def core_v1_api(self):
        return client.CoreV1Api(self.api_client)

    @lazy_property
    def api_client(self):
        return client.ApiClient(self.configuration)

    @lazy_property
    def configuration(self):
        configuration = Configuration()
        configuration.host = self.api_url
        configuration.verify_ssl = False  # Depends on your SSL setup
        configuration.debug = False
        configuration.api_key = {
            'authorization': f"Bearer {self.api_token}"
        }
        return configuration

    @lazy_property
    def client(self):

        # Returns CoreV1Api client
        return client.CoreV1Api(client.ApiClient(self.configuration))
        # configuration = Configuration()
        # configuration.host = self.api_url
        # configuration.verify_ssl = False  # Depends on your SSL setup
        # configuration.debug = False
        # configuration.api_key = {
        #  'authorization': f"Bearer {self.api_token}"
        # }
        # # Create the API client
        # # return configuration
        # return client.CoreV1Api(client.ApiClient(configuration))

    @property
    def namespaces(self):
        n_spaces = None
        try:
            n_spaces = self.core_v1_api.list_namespace()
        except ApiException as e:
            logger.exception(f"Exception when calling CoreV1Api->list_namespace: {e}")

        return n_spaces

    @staticmethod
    def generate_config(namespace, deployment_name, image, replicas=1, port=80, cpu_request=0.1, cpu_limit=1,):
        """
        Generate a YAML configuration for a Kubernetes Deployment.
        """
        # Create a dictionary representation of the Deployment
        config = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': deployment_name,
                'labels': {
                    'app': deployment_name
                }
            },
            'spec': {
                'replicas': replicas,
                'selector': {
                    'matchLabels': {
                        'app': deployment_name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': deployment_name
                        }
                    },
                    'spec': {
                        'containers': [
                            {
                                'name': deployment_name,
                                'image': image,
                                'ports': [
                                    {
                                        'containerPort': port
                                    }
                                ],
                                'resources': {
                                    'requests': {
                                        'cpu': f'{cpu_request}m'
                                    },
                                    'limits': {
                                        'cpu': f'{cpu_limit}m'
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }

        return config

    def generate_yaml_config(self, path, *args, **kwargs):
        path = beam_path(path)
        conf = self.generate_config(*args, **kwargs)
        path.write(conf, ext='.yaml')

    def generate_file_config(self, path, *args, **kwargs):
        path = beam_path(path)
        conf = self.generate_config(*args, **kwargs)
        path.write(conf)

    def get_deployments(self, namespace='default'):
        try:
            # Use AppsV1Api to access deployment-related methods
            apps_v1_api = client.AppsV1Api(self.api_client)
            deployments = apps_v1_api.list_namespaced_deployment(namespace)

            for deployment in deployments.items:
                logger.info(f"Deployment Name: {deployment.metadata.name}")
                # You can also print additional details about each deployment here

        except ApiException as e:
            logger.exception(f"Exception when calling AppsV1Api->list_namespaced_deployment: {e}")

    def get_pods(self, namespace='default'):
        try:
            pods = self.client.list_namespaced_pod(namespace)
            for pod in pods.items:
                print(f"Pod Name: {pod.metadata.name}")
        except ApiException as e:
            print(f"Exception when calling CoreV1Api->list_namespaced_pod: {e}")

    def get_cpu_resources(self, namespace, deployment_name):
        try:
            apps_v1_api = client.AppsV1Api(self.api_client)
            core_v1_api = client.CoreV1Api(self.api_client)

            # Get deployment
            deployment = apps_v1_api.read_namespaced_deployment(deployment_name, namespace)
            match_labels = deployment.spec.selector.match_labels

            # List all pods matching deployment labels
            label_selector = ",".join([f"{k}={v}" for k, v in match_labels.items()])
            pods = core_v1_api.list_namespaced_pod(namespace, label_selector=label_selector)

            total_cpu_requests = 0
            total_cpu_limits = 0

            for pod in pods.items:
                for container in pod.spec.containers:
                    if container.resources.requests and 'cpu' in container.resources.requests:
                        total_cpu_requests += self.parse_cpu_value(container.resources.requests['cpu'])
                    if container.resources.limits and 'cpu' in container.resources.limits:
                        total_cpu_limits += self.parse_cpu_value(container.resources.limits['cpu'])

            print(f"Total CPU Requests in Deployment '{deployment_name}': {total_cpu_requests}")
            print(f"Total CPU Limits in Deployment '{deployment_name}': {total_cpu_limits}")

        except ApiException as e:
            print(f"Exception when accessing Kubernetes API: {e}")

    def parse_cpu_value(self, cpu_value):
        """
        Parse Kubernetes CPU value to raw.
        Example: '100m' -> 0.1, '1' -> 1
        """
        if cpu_value.endswith('m'):
            return int(cpu_value[:-1]) / 1000
        return int(cpu_value)

# @lazy_property
# def configure_kubernetes_client(self):
#     configuration = Configuration()
#     configuration.host = self.api_url
#     configuration.verify_ssl = False  # Adjust as needed
#     configuration.api_key = {
#         'authorization': f"Bearer {self.api_token}"
#     }
#     return configuration

# config.load_kube_config()


#    def find_available_nodeports(self, count=20, port_range=(30000, 32767)):

# Get all the services in all namespaces
#        services = self.client.list_service_for_all_namespaces()

# # Gather all used nodePort
# used_ports = set()
# for service in services.items:
#     if not len(service.spec.ports):
#         continue
#     for port in service.spec.ports:
#         if port.node_port is not None:
#             used_ports.add(port.node_port)
#
# # Find available nodePorts
# available_ports = []
# for port in range(port_range[0], port_range[1] + 1):
#     if port not in used_ports:
#         available_ports.append(port)
#         if len(available_ports) >= count:
#             break
#
# return available_ports


# @lazy_property
# def total_gpus(self):
#     total_gpus = 0
#
#     # Get all the nodes
#     nodes = self.client.list_node().items
#     for node in nodes:
#         # Check if the node has GPU resources
#         if 'nvidia.com/gpu' in node.status.capacity:
#             num_gpus = int(node.status.capacity['nvidia.com/gpu'])
#             total_gpus += num_gpus
#    return total_gpus
