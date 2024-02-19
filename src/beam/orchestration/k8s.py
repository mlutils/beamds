from ..path import beam_path
from ..core import Processor
from ..utils import lazy_property
from kubernetes import client, config
from kubernetes.client import Configuration
from kubernetes.client import V1Ingress, V1IngressSpec, V1IngressRule, V1HTTPIngressRuleValue, V1HTTPIngressPath, V1IngressBackend, V1ServiceBackendPort, V1IngressTLS, V1ObjectMeta
from kubernetes.client.rest import ApiException
from ..logger import beam_logger as logger
from dataclasses import dataclass
import json


@dataclass
class ServiceConfig:
    port: int
    service_name: str
    service_type: str
    port_name: str
    create_route: bool = False  # Indicates whether to create a route for this service
    route_protocol: str = 'http'  # Default to 'http', can be overridden to 'https' as needed
    create_ingress: bool = False  # Indicates whether to create an ingress for this service
    ingress_host: str = None  # Optional: specify a host for the ingress
    ingress_path: str = '/'  # Default path for ingress, can be overridden
    ingress_tls_secret: str = None  # Optional: specify a TLS secret for ingress TLS


class BeamPod(Processor):
    pass


class BeamDeploy(Processor):

    def __init__(self, k8s=None, project_name=None, namespace=None, replicas=None, labels=None, image_name=None,
                 deployment_name=None, use_scc=False, service_configs=None, scc_name='anyuid',
                 service_type=None, *entrypoint_args, **entrypoint_envs):
        super().__init__()
        self.k8s = k8s
        self.entrypoint_args = entrypoint_args
        self.entrypoint_envs = entrypoint_envs
        self.project_name = project_name
        self.namespace = namespace
        self.replicas = replicas
        self.labels = labels
        self.image_name = image_name
        self.deployment_name = deployment_name
        self.service_type = service_type
        self.service_account_name = f"svc{deployment_name}"
        self.use_scc = use_scc
        self.scc_name = scc_name if use_scc else None
        self.service_configs = service_configs

    def launch(self, replicas=None):
        if replicas is None:
            replicas = self.replicas

        self.k8s.create_service_account(self.service_account_name, self.namespace)

        if self.use_scc:
            # Ensure the Service Account exists or create it
            self.k8s.create_service_account(self.service_account_name, self.namespace)
            # Add the Service Account to the specified SCC
            self.k8s.add_scc_to_service_account(self.service_account_name, self.namespace, self.scc_name)

        for svc_config in self.service_configs:
            service_name = f"{svc_config.service_name}-{svc_config.port}"  # Unique name based on service name and port
            self.k8s.create_service(
                base_name=service_name,
                namespace=self.namespace,
                ports=[svc_config.port],  # Wrap the port in a list
                labels=self.labels,
                service_type=svc_config.service_type
            )

            # Check if a route needs to be created for this service
            if svc_config.create_route:
                self.k8s.create_route(
                    service_name=service_name,
                    namespace=self.namespace,
                    protocol=svc_config.route_protocol,
                    port_name=svc_config.port_name
                )

            # Check if an ingress needs to be created for this service
            if svc_config.create_ingress:
                self.k8s.create_ingress(
                    service_configs=[svc_config],  # Pass only the current ServiceConfig
                )

        extracted_ports = [svc_config.port for svc_config in self.service_configs]

        deployment = self.k8s.create_deployment(image_name=self.image_name, labels=self.labels,
                                                deployment_name=self.deployment_name, namespace=self.namespace,
                                                project_name=self.project_name, replicas=replicas,
                                                ports=extracted_ports,
                                                service_account_name=self.service_account_name, *self.entrypoint_args,
                                                **self.entrypoint_envs)

        pod = self.k8s.apply_deployment(deployment, namespace=self.namespace)
        return BeamPod(pod)


class BeamK8S(Processor):  # processor is a another class and the BeamK8S inherits the method of processor
    """BeamK8S is a class  that  provides a simple interface to the Kubernetes API."""

    def __init__(self, api_url=None, api_token=None, namespace=None,
                 project_name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_token = api_token
        self.api_url = api_url
        self.project_name = project_name
        self.namespace = namespace

    @lazy_property
    def core_v1_api(self):
        return client.CoreV1Api(self.api_client)

    @lazy_property
    def api_client(self):
        return client.ApiClient(self.configuration)

    @lazy_property
    def apps_v1_api(self):
        return client.AppsV1Api(self.api_client)

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
    def dyn_client(self):
        from openshift.dynamic import DynamicClient
        # Ensure the api_client is initialized before creating the DynamicClient
        return DynamicClient(self.api_client)

    def add_scc_to_service_account(self, service_account_name, namespace, scc_name='anyuid'):
        scc = self.dyn_client.resources.get(api_version='security.openshift.io/v1', kind='SecurityContextConstraints')
        scc_obj = scc.get(name=scc_name)
        user_name = f"system:serviceaccount:{namespace}:{service_account_name}"
        if user_name not in scc_obj.users:
            scc_obj.users.append(user_name)
            scc.patch(body=scc_obj, name=scc_name, content_type='application/merge-patch+json')

    @staticmethod
    def create_container(image_name, deployment_name=None, project_name=None, ports=None, *entrypoint_args, **envs):
        container_name = f"{project_name}-{deployment_name}-container" \
            if project_name and deployment_name else "container"
        if ports is None:
            ports = []
        return client.V1Container(
            name=container_name,
            image=image_name,
            ports=BeamK8S.create_container_ports(ports),
            args=list(entrypoint_args),
            env=BeamK8S.create_environment_variables(**envs)
        )

    @staticmethod
    def create_container_ports(ports):
        # Check if self.ports is a single integer and convert it to a list if so
        ports = [ports] if isinstance(ports, int) else ports
        return [client.V1ContainerPort(container_port=port) for port in ports]

    @staticmethod
    def create_environment_variables(**envs):
        env_vars = []
        if envs:
            for env_var in envs:
                if isinstance(env_var, dict) and 'name' in env_var and 'value' in env_var:
                    # Ensure value is a string, convert if necessary
                    value = str(env_var['value']) if not isinstance(env_var['value'], str) else env_var['value']
                    env_vars.append(client.V1EnvVar(name=env_var['name'], value=value))
                elif isinstance(env_var, str):
                    # If env_var is a string, assume it's in "name=value" format
                    parts = env_var.split('=', 1)
                    if len(parts) == 2:
                        env_vars.append(client.V1EnvVar(name=parts[0], value=parts[1]))
                    else:
                        # For a plain string without '=', assign a generic name
                        env_vars.append(client.V1EnvVar(name=f"ENV_{env_var}", value=env_var))
                elif isinstance(env_var, (int, float)):
                    # For numeric types, convert to string and assign a generic name
                    env_vars.append(client.V1EnvVar(name=f"NUM_ENV_{env_var}", value=str(env_var)))
                else:
                    raise TypeError(f"Unsupported environment variable type: {type(env_var)}")
        return env_vars

    @staticmethod
    def create_pod_template(image_name, labels=None, deployment_name=None, project_name=None,
                            ports=None, service_account_name=None, *entrypoint_args, **envs):

        if labels is None:
            labels = {}
        if project_name is not None:
            labels['project'] = project_name

        container = BeamK8S.create_container(image_name, deployment_name=deployment_name,
                                             project_name=project_name, ports=ports, *entrypoint_args, **envs)

        pod_spec = client.V1PodSpec(containers=[container], service_account_name=service_account_name)

        return client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels=labels),
            spec=pod_spec
        )

    def create_deployment_spec(self, image_name, labels=None, deployment_name=None, project_name=None, replicas=None,
                               ports=None, *entrypoint_args, **envs):

        if replicas is None:
            replicas = 1

        pod_template = self.create_pod_template(image_name, labels=labels, deployment_name=deployment_name,
                                                project_name=project_name, ports=ports,
                                                *entrypoint_args, **envs)
        return client.V1DeploymentSpec(
            logger.info(f"Replicas before conversion: {replicas}"),
            replicas=int(replicas),  # Cast replicas to int
            template=pod_template,
            selector={'matchLabels': pod_template.metadata.labels}
        )

    def create_deployment(self, image_name, labels=None, deployment_name=None, namespace=None, project_name=None,
                          replicas=None, ports=None, *entrypoint_args, **envs):
        if namespace is None:
            namespace = self.namespace

        if project_name is None:
            project_name = self.project_name

        # Generate a unique name for the deployment if it's not provided
        if deployment_name is None:
            deployment_name = self.generate_unique_deployment_name(base_name=image_name.split(':')[0],
                                                                   namespace=namespace)
            # Include the 'app' label set to the unique deployment name
            if labels is None:
                labels = {}
            labels['app'] = deployment_name  # Set the 'app' label to the unique deployment name

        deployment_name = self.generate_unique_deployment_name(deployment_name, namespace)

        deployment_spec = self.create_deployment_spec(image_name, labels=labels, deployment_name=deployment_name,
                                                      project_name=project_name, replicas=replicas, ports=ports,
                                                      *entrypoint_args, **envs)

        # Optionally add the project name to the deployment's metadata
        deployment_metadata = client.V1ObjectMeta(name=deployment_name, namespace=namespace,
                                                  labels={"project": project_name})

        logger.info(f"Deployment {deployment_name} created in namespace {namespace}.")
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=deployment_metadata,
            spec=deployment_spec
        )
        return deployment

    def generate_unique_deployment_name(self, base_name, namespace):
        unique_name = base_name
        suffix = 1
        while True:
            try:
                self.apps_v1_api.read_namespaced_deployment(name=unique_name, namespace=namespace)
                # If the deployment exists, append/increment the suffix and try again
                unique_name = f"{base_name}-{suffix}"
                suffix += 1
            except ApiException as e:
                if e.status == 404:  # Not Found, the name is unique
                    return unique_name
                raise  # Reraise exceptions that are not related to the deployment not existing

    def apply_deployment(self, deployment, namespace=None):
        logger.info(f"Deployment object to be created: {deployment}")  # Adjust logging level/method as needed

        if namespace is None:
            namespace = self.namespace
        if namespace is None:
            namespace = self.project_name

        try:
            self.apps_v1_api.create_namespaced_deployment(body=deployment, namespace=namespace)
        except ApiException as e:
            logger.exception(f"Exception when applying the deployment: {e}")

    def create_service(self, base_name, namespace, ports, labels, service_type):
        # Initialize the service name with the base name
        service_name = base_name

        # Ensure ports is a list, even if it's None or empty
        if ports is None:
            ports = []

        # Check if the service already exists
        try:
            existing_service = self.core_v1_api.read_namespaced_service(name=base_name, namespace=namespace)
            if existing_service:
                print(f"Service '{base_name}' already exists in namespace '{namespace}'. Generating a unique name.")
                # Generate a unique name for the service
                service_name = self.generate_unique_service_name(base_name, namespace)
        except client.exceptions.ApiException as e:
            if e.status != 404:  # If the error is not 'Not Found', raise it
                raise

        # Do not override 'app' label if it's already set in the labels dictionary
        if 'app' not in labels:
            labels['app'] = service_name

        # Define service metadata with the unique name
        metadata = client.V1ObjectMeta(name=service_name, labels=labels)

        # Dynamically create service ports from the ports list, including unique names for each
        service_ports = []
        for idx, port in enumerate(ports):
            port_name = f"{service_name}-port-{idx}-{port}"
            service_ports.append(client.V1ServicePort(name=port_name, port=port, target_port=port))

        # Define service spec with dynamically set ports
        service_spec = client.V1ServiceSpec(
            ports=service_ports,
            selector=labels,
            type=service_type
        )

        # Create the Service object with the unique name
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=metadata,
            spec=service_spec
        )

        # Create the service in the specified namespace
        try:
            self.core_v1_api.create_namespaced_service(namespace=namespace, body=service)
            print(f"Service '{service_name}' created successfully in namespace '{namespace}'.")
        except client.exceptions.ApiException as e:
            print(f"Failed to create service '{service_name}' in namespace '{namespace}': {e}")

    def generate_unique_service_name(self, base_name, namespace):
        unique_name = base_name
        suffix = 1
        while True:
            try:
                self.core_v1_api.read_namespaced_service(name=unique_name, namespace=namespace)
                # If the service exists, append/increment the suffix and try again
                unique_name = f"{base_name}-{suffix}"
                suffix += 1
            except client.exceptions.ApiException as e:
                if e.status == 404:  # Not Found, the name is unique
                    return unique_name
                raise  # Reraise exceptions that are not related to the service not existing

    def create_service_account(self, name, namespace):
        try:
            self.core_v1_api.read_namespaced_service_account(name, namespace)
            logger.info(f"Service Account {name} already exists in namespace {namespace}.")
        except ApiException as e:
            if e.status == 404:  # Not Found
                metadata = client.V1ObjectMeta(name=name)
                service_account = client.V1ServiceAccount(api_version="v1", kind="ServiceAccount", metadata=metadata)
                self.core_v1_api.create_namespaced_service_account(namespace=namespace, body=service_account)
                logger.info(f"Service Account {name} created in namespace {namespace}.")
            else:
                logger.error(f"Failed to check or create Service Account {name} in namespace {namespace}: {e}")
                raise

    def create_route(self, service_name, namespace, protocol, port_name):
        from openshift.dynamic import DynamicClient
        dyn_client = DynamicClient(self.api_client)

        # Get the Route resource from the OpenShift API
        route_resource = dyn_client.resources.get(api_version='route.openshift.io/v1', kind='Route')

        # Check if the route already exists
        try:
            existing_route = route_resource.get(name=service_name, namespace=namespace)
            # If the route exists, log the information and skip creation
            logger.info(f"Route {service_name} already exists in namespace {namespace}. Skipping creation.")
            return
        except Exception as e:
            if e.status != 404:  # If the error is not 'Not Found', raise it
                logger.error(f"Failed to check existence of route {service_name} in namespace {namespace}: {e}")
                raise

        # Define the route manifest for creation if it does not exist
        route_manifest = {
            "apiVersion": "route.openshift.io/v1",
            "kind": "Route",
            "metadata": {
                "name": service_name,
                "namespace": namespace
            },
            "spec": {
                "to": {
                    "kind": "Service",
                    "name": service_name
                },
                "port": {
                    "targetPort": port_name
                }
            }
        }

        # Add TLS termination if protocol is 'https'
        if protocol.lower() == 'https':
            route_manifest["spec"]["tls"] = {
                "termination": "edge"
            }

        # Attempt to create the route
        try:
            route_resource.create(body=route_manifest, namespace=namespace)
            logger.info(f"Route for service {service_name} created successfully in namespace {namespace}.")
        except Exception as e:
            logger.error(f"Failed to create route for service {service_name} in namespace {namespace}: {e}")

    def create_ingress(self, service_configs, default_host=None, default_path="/", default_tls_secret=None):
        # Initialize the NetworkingV1Api
        networking_v1_api = client.NetworkingV1Api()

        for svc_config in service_configs:
            if not svc_config.create_ingress:
                continue  # Skip if create_ingress is False for this service config

            # Use specific values from svc_config or fall back to default parameters
            host = svc_config.ingress_host if svc_config.ingress_host else f"{svc_config.service_name}.example.com"
            path = svc_config.ingress_path if svc_config.ingress_path else default_path
            tls_secret = svc_config.ingress_tls_secret if svc_config.ingress_tls_secret else default_tls_secret

            # Define Ingress metadata
            metadata = V1ObjectMeta(
                name=f"{svc_config.service_name}-ingress",
                namespace=self.namespace,
                annotations={
                    "kubernetes.io/ingress.class": "nginx",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true",
                    "nginx.ingress.kubernetes.io/force-ssl-redirect": "true",
                }
            )

            # Define the backend service
            backend = V1IngressBackend(
                service=V1ServiceBackendPort(
                    name=svc_config.port_name,
                    number=svc_config.port
                )
            )

            # Define the Ingress rule
            rule = V1IngressRule(
                host=host,
                http=V1HTTPIngressRuleValue(
                    paths=[
                        V1HTTPIngressPath(
                            path=path,
                            path_type="Prefix",
                            backend=backend
                        )
                    ]
                )
            )

            # Define Ingress Spec with optional TLS configuration
            spec = V1IngressSpec(rules=[rule])
            if tls_secret:
                spec.tls = [
                    V1IngressTLS(
                        hosts=[host],
                        secret_name=tls_secret
                    )
                ]

            # Create the Ingress object
            ingress = V1Ingress(
                api_version="networking.k8s.io/v1",
                kind="Ingress",
                metadata=metadata,
                spec=spec
            )

            # Use the NetworkingV1Api to create the Ingress
            try:
                networking_v1_api.create_namespaced_ingress(namespace=self.namespace, body=ingress)
                logger.info(
                    f"Ingress for service {svc_config.service_name} created successfully in namespace {self.namespace}.")
            except Exception as e:
                logger.error(
                    f"Failed to create Ingress for service {svc_config.service_name} in namespace {self.namespace}: {e}")

    def get_internal_endpoints_with_nodeport(self, namespace):
        endpoints = []
        try:
            services = self.core_v1_api.list_namespaced_service(namespace=namespace)
            nodes = self.core_v1_api.list_node()
            node_ips = {node.metadata.name:
                            [address.address for address in node.status.addresses if address.type == "InternalIP"][0]
                        for node in nodes.items}

            for service in services.items:
                if service.spec.type == "NodePort":
                    for port in service.spec.ports:
                        for node_name, node_ip in node_ips.items():
                            endpoint = {'node_ip': node_ip, 'node_port': port.node_port,
                                        'service_name': service.metadata.name}
                            if endpoint not in endpoints:  # Check for uniqueness
                                endpoints.append(endpoint)
                                print(
                                    f"Debug: Adding endpoint for service {service.metadata.name} "
                                    f"on node {node_name} - {endpoint}")

        except client.exceptions.ApiException as e:
            print(f"Failed to retrieve services or nodes in namespace '{namespace}': {e}")

        return endpoints

    @property
    def namespaces(self):
        n_spaces = None
        try:
            n_spaces = self.core_v1_api.list_namespace()
        except ApiException as e:
            logger.exception(f"Exception when calling CoreV1Api->list_namespace: {e}")

        return n_spaces

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
            pods = self.core_v1_api.list_namespaced_pod(namespace)
            for pod in pods.items:
                print(f"Pod Name: {pod.metadata.name}")
        except ApiException as e:
            print(f"Exception when calling CoreV1Api->list_namespaced_pod: {e}")

    # def generate_yaml_config(self, path, *args, **kwargs):
    #     path = beam_path(path)
    #     conf = self.generate_config(*args, **kwargs)
    #     path.write(conf, ext='.yaml')
    #
    # def generate_file_config(self, path, *args, **kwargs):
    #     path = beam_path(path)
    #     conf = self.generate_config(*args, **kwargs)
    #     path.write(conf)
    # @staticmethod
    # def generate_config(namespace, deployment_name, image, replicas=1, port=80, cpu_request=0.1, cpu_limit=1, ):
    #     """
    #     Generate a YAML configuration for a Kubernetes Deployment.
    #     """
    #     # Create a dictionary representation of the Deployment
    #     config = {
    #         'apiVersion': 'apps/v1',
    #         'kind': 'Deployment',
    #         'metadata': {
    #             'name': deployment_name,
    #             'labels': {
    #                 'app': deployment_name
    #             }
    #         },
    #         'spec': {
    #             'replicas': replicas,
    #             'selector': {
    #                 'matchLabels': {
    #                     'app': deployment_name
    #                 }
    #             },
    #             'template': {
    #                 'metadata': {
    #                     'labels': {
    #                         'app': deployment_name
    #                     }
    #                 },
    #                 'spec': {
    #                     'containers': [
    #                         {
    #                             'name': container.image,
    #                             'image': image,
    #                             'ports': [
    #                                 {
    #                                     'containerPort': port
    #                                 }
    #                             ],
    #                             'resources': {
    #                                 'requests': {
    #                                     'cpu': f'{cpu_request}m'
    #                                 },
    #                                 'limits': {
    #                                     'cpu': f'{cpu_limit}m'
    #                                 }
    #                             }
    #                         }
    #                     ]
    #                 }
    #             }
    #         }
    #     }
    #
    #     return config
    #
    # deploy = client.V1Container

    # def get_cpu_resources(self, namespace, deployment_name):
    #     try:
    #         apps_v1_api = client.AppsV1Api(self.api_client)
    #         core_v1_api = client.CoreV1Api(self.api_client)
    #
    #         # Get deployment
    #         deployment = apps_v1_api.read_namespaced_deployment(deployment_name, namespace)
    #         match_labels = deployment.spec.selector.match_labels
    #
    #         # List all pods matching deployment labels
    #         label_selector = ",".join([f"{k}={v}" for k, v in match_labels.items()])
    #         pods = core_v1_api.list_namespaced_pod(namespace, label_selector=label_selector)
    #
    #         total_cpu_requests = 0
    #         total_cpu_limits = 0
    #
    #         for pod in pods.items:
    #             for container in pod.spec.containers:
    #                 if container.resources.requests and 'cpu' in container.resources.requests:
    #                     total_cpu_requests += self.parse_cpu_value(container.resources.requests['cpu'])
    #                 if container.resources.limits and 'cpu' in container.resources.limits:
    #                     total_cpu_limits += self.parse_cpu_value(container.resources.limits['cpu'])
    #
    #         print(f"Total CPU Requests in Deployment '{deployment_name}': {total_cpu_requests}")
    #         print(f"Total CPU Limits in Deployment '{deployment_name}': {total_cpu_limits}")
    #
    #     except ApiException as e:
    #         print(f"Exception when accessing Kubernetes API: {e}")

    # def parse_cpu_value(self, cpu_value):
    #     """
    #     Parse Kubernetes CPU value to raw.
    #     Example: '100m' -> 0.1, '1' -> 1
    #     """
    #     if cpu_value.endswith('m'):
    #         return int(cpu_value[:-1]) / 1000
    #     return int(cpu_value)

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
