from .beamk8s import *
from kubernetes import client
from kubernetes.client import Configuration



class BeamNetwork(Processor):

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

    def create_route(self, service_name, namespace, protocol, port):
        from openshift.dynamic import DynamicClient
        from openshift.dynamic.exceptions import NotFoundError

        dyn_client = DynamicClient(self.api_client)

        # Get the Route resource from the OpenShift API
        route_resource = dyn_client.resources.get(api_version='route.openshift.io/v1', kind='Route')

        try:
            # Try to get the existing route
            existing_route = route_resource.get(name=service_name, namespace=namespace)
            # If the route exists, log a message and return
            logger.info(f"Route {service_name} already exists in namespace {namespace}, skipping creation.")
            return
        except NotFoundError:
            # The route does not exist, proceed with creation
            logger.info(f"Route {service_name} does not exist in namespace {namespace}, proceeding with creation.")
        except Exception as e:
            # Handle other exceptions that are not related to route not found
            logger.error(f"Error checking route {service_name} in namespace {namespace}: {e}")
            return

        # Define the route manifest for creation
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
                    "targetPort": port  # Use numeric port
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
        from kubernetes.client import (V1Ingress, V1IngressSpec, V1IngressRule, V1HTTPIngressRuleValue,
                                       V1HTTPIngressPath,
                                       V1IngressBackend, V1ServiceBackendPort, V1IngressTLS, V1ObjectMeta)
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
                    f"Ingress for service {svc_config.service_name} "
                    f"created successfully in namespace {self.namespace}.")
            except Exception as e:
                logger.error(
                    f"Failed to create Ingress for service {svc_config.service_name} "
                    f"in namespace {self.namespace}: {e}")