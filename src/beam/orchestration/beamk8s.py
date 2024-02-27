from ..core import Processor
from ..utils import lazy_property
from kubernetes import client
from kubernetes.client import Configuration
from kubernetes.client.rest import ApiException
from ..logger import beam_logger as logger

class BeamK8S(Processor):  # processor is another class and the BeamK8S inherits the method of processor
    """BeamK8S is a class  that  provides a simple interface to the Kubernetes API."""

    def __init__(self, api_url=None, api_token=None, namespace=None,
                 project_name=None, use_scc=True, scc_name="anyuid", *args, **kwargs):
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

    def add_scc_to_service_account(self, service_account_name, namespace, scc_name='anyuid'):
        scc = self.dyn_client.resources.get(api_version='security.openshift.io/v1', kind='SecurityContextConstraints')
        scc_obj = scc.get(name=scc_name)
        user_name = f"system:serviceaccount:{namespace}:{service_account_name}"
        if user_name not in scc_obj.users:
            scc_obj.users.append(user_name)
            scc.patch(body=scc_obj, name=scc_name, content_type='application/merge-patch+json')

    def create_service_account(self, name, service_account_name, namespace=None):
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

    def create_container(self, image_name, deployment_name=None, project_name=None, ports=None, pvc_mounts=None,
                         cpu_requests=None, cpu_limits=None, memory_requests=None, memory_limits=None, gpu_requests=None,
                         gpu_limits=None, *entrypoint_args, **envs):
        container_name = f"{project_name}-{deployment_name}-container" \
            if project_name and deployment_name else "default-container"

        # Preparing environment variables from entrypoint_args and envs
        env_vars = []
        for arg in entrypoint_args:
            env_vars.append(client.V1EnvVar(name=f"ARG_{arg}", value=str(arg)))
        for key, value in envs.items():
            env_vars.append(client.V1EnvVar(name=key, value=str(value)))

        # Preparing volume mounts
        volume_mounts = []
        if pvc_mounts:
            for mount in pvc_mounts:
                volume_mounts.append(client.V1VolumeMount(
                    name=mount['pvc_name'],  # Using PVC name as the volume name
                    mount_path=mount['mount_path']  # Mount path specified in pvc_mounts
                ))

        resources = {
            'requests': {},
            'limits': {}
        }

        if cpu_requests and cpu_limits:
            resources['requests']['cpu'] = cpu_requests
            resources['limits']['cpu'] = cpu_limits
        if memory_requests and memory_limits:
            resources['requests']['memory'] = memory_requests
            resources['limits']['memory'] = memory_limits
        if gpu_requests and gpu_limits:
            resources['limits']['nvidia.com/gpu'] = gpu_requests
            resources['requests']['nvidia.com/gpu'] = gpu_limits

        return client.V1Container(
            name=container_name,
            image=image_name,
            ports=[client.V1ContainerPort(container_port=port) for port in ports] if ports else [],
            env=env_vars,
            volume_mounts=volume_mounts,
            resources=client.V1ResourceRequirements(requests=resources['requests'], limits=resources['limits'])
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
                            ports=None, service_account_name=None, pvc_mounts=None, entrypoint_args=None, envs=None):
        if labels is None:
            labels = {}
        if project_name:
            labels['project'] = project_name

        # Ensure image_name is not included in entrypoint_args or envs
        container = self.create_container(
            image_name=image_name,  # Ensure this is the only place where image_name is provided
            deployment_name=deployment_name,
            project_name=project_name,
            ports=ports,
            pvc_mounts=pvc_mounts,
            entrypoint_args=entrypoint_args,
            envs=envs
        )

        # Defining volumes for the pod spec based on PVC mounts
        volumes = []
        if pvc_mounts:
            for mount in pvc_mounts:
                volumes.append(client.V1Volume(
                    name=mount['pvc_name'],  # Volume name matches the PVC name
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name=mount['pvc_name'])
                ))

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
                               ports=None, service_account_name=None, storage_configs=None, *entrypoint_args, **envs):
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
            entrypoint_args=entrypoint_args,
            envs=envs
        )

        # Create and return the deployment spec
        return client.V1DeploymentSpec(
            replicas=int(replicas),  # Ensure replicas is an int
            template=pod_template,
            selector={'matchLabels': pod_template.metadata.labels}
        )

    def create_deployment(self, image_name, labels=None, deployment_name=None, namespace=None, project_name=None,
                          replicas=None, ports=None, service_account_name=None, storage_configs=None, *entrypoint_args,
                          **envs):
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
            storage_configs=storage_configs,
            *entrypoint_args, **envs
        )

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
        import json
        logger.debug(json.dumps(client.ApiClient().sanitize_for_serialization(deployment), indent=2))
        logger.info(f"Deployment object to be created: {deployment}")  # Adjust logging level/method as needed

        if namespace is None:
            namespace = self.namespace
        if namespace is None:
            namespace = self.project_name

        try:
            self.apps_v1_api.create_namespaced_deployment(body=deployment, namespace=namespace)
        except ApiException as e:
            logger.exception(f"Exception when applying the deployment: {e}")

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



