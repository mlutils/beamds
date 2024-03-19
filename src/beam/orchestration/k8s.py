from typing import Union

from dataclasses import make_dataclass
from ..core import Processor
from ..utils import lazy_property
from kubernetes import client, watch
from kubernetes.client import Configuration, RbacAuthorizationV1Api, V1DeleteOptions
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream
from ..logger import beam_logger as logger
from .units import K8SUnits
import time
import json


class BeamK8S(Processor):  # processor is another class and the BeamK8S inherits the method of processor
    """BeamK8S is a class  that  provides a simple interface to the Kubernetes API."""

    def __init__(self, api_url=None, api_token=None, namespace=None,
                 project_name=None, use_scc=None, scc_name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_token = api_token
        self.api_url = api_url
        self.project_name = project_name
        self.namespace = namespace
        self.use_scc = use_scc
        self.scc_name = scc_name

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

    @lazy_property
    def rbac_api(self):
        return RbacAuthorizationV1Api(self.api_client)

    def create_project(self, project_name):
        try:
            # Attempt to get the project to see if it already exists
            self.dyn_client.resources.get(api_version='project.openshift.io/v1', kind='Project').get(name=project_name)
            logger.info(f"Project '{project_name}' already exists.")
        except ApiException as e:
            if e.status == 404:  # Project does not exist, create it
                project_request = {
                    "kind": "ProjectRequest",
                    "apiVersion": "project.openshift.io/v1",
                    "metadata": {"name": project_name}
                }
                self.dyn_client.resources.get(api_version='project.openshift.io/v1',
                                              kind='ProjectRequest').create(body=project_request)
                logger.info(f"Project '{project_name}' created successfully.")
            else:
                logger.error(f"Failed to check or create project '{project_name}': {e}")

    def create_service_account(self, name, namespace=None):
        namespace = namespace or self.namespace

        # Attempt to read the service account, create it if it does not exist
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

    def add_scc_to_service_account(self, service_account_name, namespace, scc_name):
        scc = self.dyn_client.resources.get(api_version='security.openshift.io/v1', kind='SecurityContextConstraints')
        scc_obj = scc.get(name=scc_name)
        user_name = f"system:serviceaccount:{namespace}:{service_account_name}"
        if user_name not in scc_obj.users:
            scc_obj.users.append(user_name)
            scc.patch(body=scc_obj, name=scc_name, content_type='application/merge-patch+json')

    @staticmethod
    def create_container(image_name, deployment_name=None, project_name=None, ports=None, pvc_mounts=None,
                         cpu_requests=None, cpu_limits=None, memory_requests=None,
                         memory_limits=None, gpu_requests=None, memory_storage_configs=None,
                         gpu_limits=None, security_context_config=None, security_context=None,
                         entrypoint_args=None, entrypoint_envs=None):
        container_name = f"{project_name}-{deployment_name}-container" \
            if project_name and deployment_name else "default-container"

        # Preparing environment variables from entrypoint_args and envs
        env_vars = []
        entrypoint_args = entrypoint_args or []
        entrypoint_envs = entrypoint_envs or {}

        for arg in entrypoint_args:
            env_vars.append(client.V1EnvVar(name=f"ARG_{arg}", value=str(arg)))
        for key, value in entrypoint_envs.items():
            env_vars.append(client.V1EnvVar(name=key, value=str(value)))

        # Preparing volume mounts
        volume_mounts = []
        if pvc_mounts:
            for mount in pvc_mounts:
                volume_mounts.append(client.V1VolumeMount(
                    name=mount['pvc_name'],  # Using PVC name as the volume name
                    mount_path=mount['mount_path']  # Mount path specified in pvc_mounts
                ))

            # Handling memory_storage_configs - this is the only addition
        if memory_storage_configs:
            for mem_storage in memory_storage_configs:
                volume_mounts.append(client.V1VolumeMount(
                    name=mem_storage.name,
                    mount_path=mem_storage.mount_path
                    # Ensure this matches the attribute name in your memory_storage_configs items
                ))

        resources = {
            'requests': {},
            'limits': {}
        }

        if cpu_requests and cpu_limits:
            resources['requests']['cpu'] = K8SUnits(cpu_requests).as_str
            resources['limits']['cpu'] = K8SUnits(cpu_limits).as_str
        if memory_requests and memory_limits:
            resources['requests']['memory'] = K8SUnits(memory_requests).as_str
            resources['limits']['memory'] = K8SUnits(memory_limits).as_str
        if gpu_requests and gpu_limits:
            resources['requests']['nvidia.com/gpu'] = K8SUnits(gpu_limits).as_str
            resources['limits']['nvidia.com/gpu'] = K8SUnits(gpu_requests).as_str

        if security_context_config and security_context_config.enable_security_context:
            security_context = {
                "capabilities": {
                    "add": security_context_config.add_capabilities
                }
            }

        return client.V1Container(
            name=container_name,
            image=image_name,
            ports=[client.V1ContainerPort(container_port=port) for port in ports] if ports else [],
            env=env_vars,
            volume_mounts=volume_mounts,
            resources=client.V1ResourceRequirements(requests=resources['requests'], limits=resources['limits']),
            security_context=security_context
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

    def create_pod_template(self, image_name, labels=None, deployment_name=None, project_name=None,
                            ports=None, service_account_name=None, pvc_mounts=None,
                            cpu_requests=None, cpu_limits=None, memory_requests=None,
                            memory_limits=None, gpu_requests=None, gpu_limits=None, memory_storage_configs=None,
                            security_context_config=None, entrypoint_args=None, entrypoint_envs=None, ):
        if labels is None:
            labels = {}
        if project_name:
            labels['project'] = project_name

        volumes = []

        if memory_storage_configs:
            for mem_storage in memory_storage_configs:
                # Initialize the memory_volume_spec without size_limit
                memory_volume_spec = client.V1EmptyDirVolumeSource(medium="Memory")

                # Conditionally set size_limit if specified
                if mem_storage.size_gb is not None:
                    # Attempt to set size_limit directly, converting explicitly to a string
                    memory_volume_spec.size_limit = mem_storage.size_gb.as_str  # Explicit string conversion

                volumes.append(client.V1Volume(
                    name=mem_storage.name,
                    empty_dir=memory_volume_spec  # The correct parameter name is empty_dir, not emptyDir
                ))

        if pvc_mounts:
            for mount in pvc_mounts:
                volumes.append(client.V1Volume(
                    name=mount['pvc_name'],  # Volume name matches the PVC name
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name=mount['pvc_name'])
                ))

        # Ensure image_name is not included in entrypoint_args or envs
        container = self.create_container(
            image_name=image_name,  # Ensure this is the only place where image_name is provided
            deployment_name=deployment_name,
            project_name=project_name,
            ports=ports,
            pvc_mounts=pvc_mounts,
            cpu_requests=cpu_requests,
            cpu_limits=cpu_limits,
            memory_requests=memory_requests,
            memory_limits=memory_limits,
            memory_storage_configs=memory_storage_configs,
            gpu_requests=gpu_requests,
            gpu_limits=gpu_limits,
            security_context_config=security_context_config,
            entrypoint_args=entrypoint_args,
            entrypoint_envs=entrypoint_envs
        )
        # Defining volumes for the pod spec based on PVC mounts

        pod_spec = client.V1PodSpec(
            containers=[container],
            service_account_name=service_account_name,
            volumes=volumes  # Including PVC volumes here
        )

        return client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels=labels),
            spec=pod_spec
        )

    def create_pvc(self, pvc_name, pvc_size, pvc_access_mode, namespace):
        logger.info(f"Attempting to create PVC: {pvc_name} in namespace: {namespace}")
        pvc_manifest = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": pvc_name},
            "spec": {
                "accessModes": [pvc_access_mode],
                "resources": {"requests": {"storage": pvc_size}}
            }
        }
        self.core_v1_api.create_namespaced_persistent_volume_claim(namespace=namespace, body=pvc_manifest)
        logger.info(f"Created PVC '{pvc_name}' in namespace '{namespace}'.")

    def create_deployment_spec(self, image_name, labels=None, deployment_name=None, project_name=None, replicas=None,
                               ports=None, service_account_name=None, storage_configs=None,
                               cpu_requests=None, cpu_limits=None, memory_requests=None,
                               memory_limits=None, gpu_requests=None, gpu_limits=None, memory_storage_configs=None,
                               security_context_config=None, entrypoint_args=None,
                               entrypoint_envs=None):
        # Ensure pvc_mounts are prepared correctly from storage_configs if needed
        pvc_mounts = [{
            'pvc_name': sc.pvc_name,
            'mount_path': sc.pvc_mount_path
        } for sc in storage_configs if sc.create_pvc] if storage_configs else []

        # Create the pod template with correct arguments
        pod_template = self.create_pod_template(
            image_name=image_name,
            labels=labels,
            deployment_name=deployment_name,
            project_name=project_name,
            ports=ports,
            service_account_name=service_account_name,  # Use it here
            pvc_mounts=pvc_mounts,  # Assuming pvc_mounts is prepared earlier in the method
            cpu_requests=cpu_requests,
            cpu_limits=cpu_limits,
            memory_storage_configs=memory_storage_configs,
            memory_requests=memory_requests,
            memory_limits=memory_limits,
            gpu_requests=gpu_requests,
            gpu_limits=gpu_limits,
            security_context_config=security_context_config,
            entrypoint_args=entrypoint_args,
            entrypoint_envs=entrypoint_envs
        )

        # Create and return the deployment spec
        return client.V1DeploymentSpec(
            replicas=int(replicas),  # Ensure replicas is an int
            template=pod_template,
            selector={'matchLabels': pod_template.metadata.labels}
        )

    def create_deployment(self, image_name, labels=None, deployment_name=None, namespace=None, project_name=None,
                          replicas=None, ports=None, service_account_name=None, storage_configs=None,
                          cpu_requests=None, cpu_limits=None, memory_requests=None, memory_storage_configs=None,
                          memory_limits=None, gpu_requests=None, gpu_limits=None,
                          security_context_config=None, entrypoint_args=None, entrypoint_envs=None):
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

        deployment_spec = self.create_deployment_spec(
            image_name, labels=labels, deployment_name=deployment_name,
            project_name=project_name, replicas=replicas, ports=ports,
            service_account_name=service_account_name,  # Pass this
            storage_configs=storage_configs, cpu_requests=cpu_requests, cpu_limits=cpu_limits,
            memory_requests=memory_requests, memory_limits=memory_limits,
            memory_storage_configs=memory_storage_configs,
            gpu_requests=gpu_requests, gpu_limits=gpu_limits, security_context_config=security_context_config,
            entrypoint_args=entrypoint_args, entrypoint_envs=entrypoint_envs,
        )

        # Optionally add the project name to the deployment's metadata
        deployment_metadata = client.V1ObjectMeta(name=deployment_name, namespace=namespace,
                                                  labels={"project": project_name})

        # logger.info(f"Deployment {deployment_name} created in namespace {namespace}.")
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=deployment_metadata,
            spec=deployment_spec
        )
        logger.info(f"Created deployment object type: {type(deployment)}")
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
        # logger.info(f"Applying deployment object type: {type(deployment)}")
        # Determine the namespace
        if namespace is None:
            namespace = self.namespace

        try:
            # Apply the deployment
            self.apps_v1_api.create_namespaced_deployment(body=deployment, namespace=namespace)
            logger.info(f"Successfully applied deployment in namespace '{namespace}'")

            # Construct the selector based on the deployment object type
            if isinstance(deployment, dict):
                # If deployment is a dictionary, use dictionary access
                selector = ','.join([f'{k}={v}' for k, v in deployment['spec']['selector']['matchLabels'].items()])
            else:
                # If deployment is an object, use attribute access
                selector = ','.join([f'{k}={v}' for k, v in deployment.spec.selector['matchLabels'].items()])

            w = watch.Watch()
            timeout = 300  # Define a timeout for the watch (e.g., 300 seconds)
            start = time.time()

            for event in w.stream(self.core_v1_api.list_namespaced_pod, namespace=namespace, label_selector=selector,
                                  timeout_seconds=60):
                pod = event['object']
                if pod.status.phase == 'Running':
                    # Extract and return pod info
                    pod_info = self.extract_pod_info(pod)
                    logger.info(f"Pod info: '{pod_info}'")
                    w.stop()
                    return pod_info

                if time.time() - start > timeout:
                    logger.error("Timeout waiting for pods to become ready.")
                    w.stop()
                    return None

        except ApiException as e:
            logger.exception(f"Exception when applying the deployment: {e}")
            return None

    @staticmethod
    def extract_pod_info(pod):
        # Extract pod name, namespace, ports, and PVCs
        pod_name = pod.metadata.name
        namespace = pod.metadata.namespace
        ports = [container.ports for container in pod.spec.containers]
        pvcs = [volume.persistent_volume_claim.claim_name for volume
                in pod.spec.volumes if volume.persistent_volume_claim]

        # Create a dynamic data class for PodInfo
        podinfo = make_dataclass('PodInfo', [('name', str), ('namespace', str), ('ports', list), ('pvcs', list)])
        return podinfo(name=pod_name, namespace=namespace, ports=ports, pvcs=pvcs)

        # TODO: return pod info - extract pod name and namespace from the applied_deployment object,
        # if needed, beam_k8s will query the server to retrieve all the necessary information

    def create_role_bindings(self, user_idm_configs):
        for config in user_idm_configs:
            if config.create_role_binding:
                role_binding = {
                    "apiVersion": "rbac.authorization.k8s.io/v1",
                    "kind": "RoleBinding",
                    "metadata": {
                        "name": config.role_binding_name,
                        "namespace": config.project_name  # Namespace is derived from project_name
                    },
                    "subjects": [{
                        "kind": "User",
                        "name": config.user_name,
                        "apiGroup": "rbac.authorization.k8s.io"
                    }],
                    "roleRef": {
                        "kind": "ClusterRole",
                        "name": config.role_name,
                        # Assuming 'admin' or equivalent ClusterRole that provides namespace-level admin access
                        "apiGroup": "rbac.authorization.k8s.io"
                    }
                }

                try:
                    self.dyn_client.resources.get(api_version='rbac.authorization.k8s.io/v1',
                                                  kind='RoleBinding').create(body=role_binding,
                                                                             namespace=config.project_name)
                    logger.info(
                        f"Admin role binding '{config.role_binding_name}' for user "
                        f"'{config.user_name}' created in namespace '{config.project_name}'.")
                except ApiException as e:
                    if e.status == 409:  # Conflict error - RoleBinding already exists
                        logger.info(
                            f"Role binding '{config.role_binding_name}' "
                            f"already exists in namespace '{config.project_name}', skipping.")
                    else:
                        logger.error(
                            f"Failed to create admin role binding for '{config.user_name}' "
                            f"in namespace '{config.project_name}': {e}")

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
            logger.info(
                f"Service '{service_name}' of type '{service_type}' created with ports: "
                f"{', '.join([f'Port: {port.port}, TargetPort: {port.target_port}' for port in service_ports])}")
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

    def delete_service(self, deployment_name, namespace=None):
        from kubernetes.client import V1DeleteOptions

        if namespace is None:
            namespace = self.namespace

        try:
            # Get the service associated with the deployment
            service_list = self.core_v1_api.list_namespaced_service(namespace=namespace)
            for service in service_list.items:
                if service.metadata.labels.get("app") == deployment_name:
                    service_name = service.metadata.name
                    # Use the core_v1_api to delete the Service
                    self.core_v1_api.delete_namespaced_service(
                        name=service_name,
                        namespace=namespace,
                        body=V1DeleteOptions()
                    )
                    logger.info(f"Deleted service '{service_name}' from namespace '{namespace}'.")
                    return  # Exit the loop once the service is deleted
        except ApiException as e:
            logger.error(f"Error deleting service for deployment '{deployment_name}': {e}")

    def delete_services_by_label_selector(self, label_selector, namespace=None):
        if namespace is None:
            namespace = self.namespace

        try:
            services = self.core_v1_api.list_namespaced_service(namespace=namespace, label_selector=label_selector)
            for service in services.items:
                service_name = service.metadata.name
                self.core_v1_api.delete_namespaced_service(
                    name=service_name,
                    namespace=namespace,
                    body=V1DeleteOptions()
                )
                logger.info(f"Deleted service '{service_name}' in namespace '{namespace}'")
        except Exception as e:
            logger.error(
                f"Error deleting services with label selector '{label_selector}' in namespace '{namespace}': {e}")

    def create_route(self, service_name, namespace, protocol, port):
        from openshift.dynamic.exceptions import NotFoundError
        from openshift.dynamic import DynamicClient

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
            created_route = route_resource.create(body=route_manifest, namespace=namespace)
            logger.info(f"Route for service {service_name} created successfully in namespace {namespace}.")
            # Print the DNS name of the route
            dns_name = created_route.spec.host  # Accessing the DNS name from the route response
            logger.info(f"The DNS name of the created route is: {dns_name}")  # Log the DNS name
        except Exception as e:
            logger.error(f"Failed to create route for service {service_name} in namespace {namespace}: {e}")

    def delete_route(self, route_name, namespace):
        from openshift.dynamic.exceptions import NotFoundError
        from openshift.dynamic import DynamicClient

        dyn_client = DynamicClient(self.api_client)

        # Get the Route resource from the OpenShift API
        route_resource = dyn_client.resources.get(api_version='route.openshift.io/v1', kind='Route')

        try:
            # Try to get the existing route
            existing_route = route_resource.get(name=route_name, namespace=namespace)
            # If the route exists, delete it
            existing_route.delete()
            logger.info(f"Deleted route '{route_name}' from namespace '{namespace}'.")
        except NotFoundError:
            # If the route doesn't exist, log a message
            logger.info(f"Route '{route_name}' does not exist in namespace '{namespace}'.")
        except Exception as e:
            # Handle other exceptions
            logger.error(f"Error deleting route '{route_name}' in namespace '{namespace}': {e}")

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

    def delete_ingress(self, service_name):
        from kubernetes.client import NetworkingV1Api, V1DeleteOptions

        try:
            # Initialize the NetworkingV1Api
            networking_v1_api = NetworkingV1Api(self.api_client)

            # Use the NetworkingV1Api to delete the Ingress
            networking_v1_api.delete_namespaced_ingress(
                name=f"{service_name}-ingress",
                namespace=self.namespace,
                body=V1DeleteOptions()
            )
            logger.info(f"Ingress for service {service_name} deleted successfully.")
        except Exception as e:
            logger.error(f"Failed to delete Ingress for service {service_name}: {e}")

    def print_pod_node_info(self, deployment_name):
        # Step 1: Find the deployment to get its selector
        deployment = self.apps_v1_api.read_namespaced_deployment(deployment_name, self.namespace)
        selector = ','.join([f'{k}={v}' for k, v in deployment.spec.selector.match_labels.items()])

        # Step 2: List all pods in the deployment using the selector
        pods = self.core_v1_api.list_namespaced_pod(namespace=self.namespace, label_selector=selector)

        # Collect unique node names where the pods are scheduled
        node_names = set()
        for pod in pods.items:
            node_name = pod.spec.node_name
            node_names.add(node_name)
            logger.info(f"Pod: {pod.metadata.name} is running on Node: {node_name}")

        # Step 3: List services and check if they target the deployment
        services = self.core_v1_api.list_namespaced_service(namespace=self.namespace)
        for service in services.items:
            if service.spec.type == "NodePort":
                for port in service.spec.ports:
                    if port.node_port:
                        logger.info(
                            f"NodePort Service: {service.metadata.name}, Port: {port.port}, NodePort: {port.node_port}")
                        # Since we can't get the external IP, we'll just inform about using the cluster's node IPs
                        logger.info(
                            "To access the NodePort service, use the IP "
                            "address of any cluster node with the listed NodePort.")

        # Inform about the limitation regarding node IPs
        logger.info(
            "Node external IPs cannot be retrieved with namespace-scoped "
            "permissions. Use known node IPs to access NodePort services.")

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

    def execute_command_in_pod(self, namespace, pod_name, command):
        """
        Execute a command in a pod.

        :param namespace: The namespace of the pod.
        :param pod_name: The name of the pod where the command will be executed.
        :param command: The command to execute inside the pod, as a list of strings.
        :return: The output from the command execution.
        """
        # Ensure that a list is provided for the command
        if not isinstance(command, list):
            command = [command]

        # Executing the command
        response = stream(self.core_v1_api.connect_get_namespaced_pod_exec,
                          pod_name,
                          namespace,
                          command=command,
                          stderr=True,
                          stdin=False,
                          stdout=True,
                          tty=False)
        return response

    def get_pod_info(self, pod_name, namespace):
        """Retrieve information about a specific pod."""
        try:
            return self.core_v1_api.read_namespaced_pod(name=pod_name, namespace=namespace or self.namespace)
        except ApiException as e:
            logger.error(f"Failed to get pod info for {pod_name} in {namespace}: {e}")
            return None

    def get_pod_logs(self, pod_name, namespace=None, **kwargs):
        """Retrieve logs for a specific pod."""
        try:
            return self.core_v1_api.read_namespaced_pod_log(name=pod_name, namespace=namespace or self.namespace,
                                                            **kwargs)
        except ApiException as e:
            logger.error(f"Failed to get logs for {pod_name} in {namespace}: {e}")
            return None

    def get_pod_resources(self, pod_name, namespace=None):
        """Get resource usage for a specific pod using the dynamic client."""
        try:
            resource = self.dyn_client.resources.get(api_version='metrics.k8s.io/v1beta1', kind='PodMetrics')
            return resource.get(name=pod_name, namespace=namespace or self.namespace)
        except ApiException as e:
            logger.error(f"Failed to get resources for {pod_name} in {namespace}: {e}")
            return None

    def stop_pod(self, pod_name, namespace=None):
        """Stop a specific pod. This is usually done by deleting the pod."""
        try:
            self.core_v1_api.delete_namespaced_pod(name=pod_name, namespace=namespace or self.namespace,
                                                   body=V1DeleteOptions())
            logger.info(f"Pod {pod_name} in {namespace} deleted successfully.")
        except ApiException as e:
            logger.error(f"Failed to delete pod {pod_name} in {namespace}: {e}")

    def start_pod(self, pod_name, namespace=None, pod_body=None):
        """Start a specific pod by creating it. `pod_body` should be a V1Pod manifest."""
        try:
            self.core_v1_api.create_namespaced_pod(namespace=namespace or self.namespace, body=pod_body)
            logger.info(f"Pod {pod_name} in {namespace} started successfully.")
        except ApiException as e:
            logger.error(f"Failed to create pod {pod_name} in {namespace}: {e}")

    # def list_pods(self):
    #     label_selector = f"app={self.deployment_name}"
    #     pods = self.core_v1_api.list_namespaced_pod(namespace=self.namespace, label_selector=label_selector)
    #     for pod in pods.items:
    #         print(f"Pod Name: {pod.metadata.name}, Pod Status: {pod.status.phase}")
