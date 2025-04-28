import base64
from dataclasses import make_dataclass # todo: why is this needed?
import kubernetes
from kubernetes import client, watch
from kubernetes.client import (Configuration, RbacAuthorizationV1Api, V1DeleteOptions, BatchV1Api,
                               V1ObjectMeta, V1RoleBinding, V1RoleRef, V1ClusterRoleBinding, V1Pod, V1PodSpec, )
from kubernetes.client.rest import ApiException
from ..logging import beam_logger as logger
from ..utils import cached_property
from .utils import ensure_rfc1123_compliance
from ..processor import Processor
from .units import K8SUnits
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from .dataclasses import *
import time
import json
import subprocess
import re

class BeamK8S(Processor):  # processor is another class and the BeamK8S inherits the method of processor
    """BeamK8S is a class  that  provides a simple interface to the Kubernetes API."""

    def __init__(self, *args, api_url=None, api_token=None, namespace=None,
                 project_name=None, use_scc=None, scc_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_token = api_token
        self.api_url = api_url
        self.project_name = project_name
        self.namespace = namespace
        self.use_scc = use_scc
        self.scc_name = scc_name

    @cached_property
    def core_v1_api(self):
        return client.CoreV1Api(self.api_client)

    @cached_property
    def batch_v1_api(self):
        return BatchV1Api(self.api_client)

    @cached_property
    def api_client(self):
        return client.ApiClient(self.configuration)

    @cached_property
    def apps_v1_api(self):
        return client.AppsV1Api(self.api_client)

    @cached_property
    def configuration(self):
        configuration = Configuration()
        configuration.host = self.api_url
        configuration.verify_ssl = False  # Depends on your SSL setup
        configuration.debug = False
        configuration.api_key = {
            'authorization': f"Bearer {self.api_token}"
        }
        return configuration

    @cached_property
    def dyn_client(self):
        from openshift.dynamic import DynamicClient
        # Ensure the api_client is initialized before creating the DynamicClient
        return DynamicClient(self.api_client)

    @cached_property
    def rbac_api(self):
        return RbacAuthorizationV1Api(self.api_client)

    @cached_property
    def custom_objects_api(self):
        return client.CustomObjectsApi(self.api_client)

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

    def get_service_accounts(self, namespace):
        service_accounts = self.core_v1_api.list_namespaced_service_account(namespace)
        return [sa.metadata.name for sa in service_accounts.items]

    def create_service_account(self, name, namespace):
        from kubernetes.client import V1ServiceAccount, V1ObjectMeta
        try:
            # Attempt to read the service account
            self.core_v1_api.read_namespaced_service_account(name, namespace)
            logger.info(f"Service Account {name} already exists in namespace {namespace}.")
            return False  # Service account exists, no need to create a new one
        except ApiException as e:
            if e.status == 404:
                # Create the service account if it does not exist
                service_account = V1ServiceAccount(metadata=V1ObjectMeta(name=name))
                self.core_v1_api.create_namespaced_service_account(namespace=namespace, body=service_account)
                logger.info(f"Service Account {name} created in namespace {namespace}.")
                # Wait for Kubernetes to potentially auto-create the secret
                time.sleep(10)
                return True
            else:
                logger.error(f"Failed to check or create Service Account {name} in namespace {namespace}: {e}")
                raise

    def create_image_stream(self, namespace, container_image):
        """
        Create or retrieve an OpenShift ImageStream for the specified container image.

        Args:
            namespace (str): The OpenShift namespace to create the ImageStream in.
            container_image (str): The full container image reference (e.g., quay.io/my-app:v1).

        Returns:
            str: The name of the ImageStream created or retrieved.
        """
        # Extract the container name and tag from the image
        if ":" in container_image:
            container_name, _ = container_image.rsplit(":", 1)
        else:
            container_name = container_image

        # Auto-generate ImageStream name from the container name
        image_stream_name = re.sub(r"[^a-z0-9-]", "-", container_name.split("/")[-1].lower())

        # Initialize the ImageStream API client
        api = self.dyn_client.resources.get(api_version="image.openshift.io/v1", kind="ImageStream")

        try:
            # Check if the ImageStream already exists
            image_stream = api.get(name=image_stream_name, namespace=namespace)
            logger.info(f"ImageStream '{image_stream_name}' already exists in namespace '{namespace}'.")
        except ApiException as e:
            if e.status == 404:
                # ImageStream does not exist, create it
                imagestream = {
                    "apiVersion": "image.openshift.io/v1",
                    "kind": "ImageStream",
                    "metadata": {
                        "name": image_stream_name,
                        "namespace": namespace,
                    },
                    "spec": {
                        "tags": [
                            {
                                "name": "latest",  # Default tag
                                "from": {
                                    "kind": "DockerImage",
                                    "name": container_image,  # Reference the container image
                                },
                                "importPolicy": {"insecure": False},
                            }
                        ]
                    },
                }
                try:
                    api.create(body=imagestream, namespace=namespace)
                    logger.info(f"ImageStream '{image_stream_name}' created successfully in namespace '{namespace}'.")
                except Exception as creation_error:
                    logger.error(
                        f"Failed to create ImageStream '{image_stream_name}' in namespace '{namespace}': {creation_error}"
                    )
                    raise
            else:
                logger.error(f"Failed to check ImageStream '{image_stream_name}': {e}")
                raise

        # Return the ImageStream name
        return image_stream_name

    def bind_service_account_to_role(self, service_account_name, namespace, role='admin'):
        from kubernetes.client import V1RoleBinding, V1RoleRef, V1ObjectMeta
        try:
            # Attempt to read the role binding
            self.rbac_api.read_namespaced_role_binding(f"{service_account_name}-admin-binding", namespace)
            logger.info(f"Role binding {service_account_name}-admin-binding already exists in namespace {namespace}.")
            return False  # Role binding exists, no need to create a new one
        except ApiException as e:
            if e.status == 404:
                # Create the role binding if it does not exist
                role_binding = V1RoleBinding(
                    metadata=V1ObjectMeta(name=f"{service_account_name}-admin-binding"),
                    subjects=[{"kind": "ServiceAccount", "name": service_account_name, "namespace": namespace}],
                    role_ref={"api_group": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": role}
                )
                self.rbac_api.create_namespaced_role_binding(namespace, role_binding)
                logger.info(f"Role binding {service_account_name}-admin-binding created in namespace {namespace}.")
                return True
            else:
                logger.error(
                    f"Failed to check or create role binding {service_account_name}-admin-binding in namespace {namespace}: {e}")
                raise

    def create_service_account_secret(self, service_account_name, namespace):
        from kubernetes.client import V1Secret, V1ObjectMeta
        # Proper annotations are needed to link the secret with the service account
        annotations = {
            'kubernetes.io/service-account.name': service_account_name
        }
        secret = V1Secret(
            metadata=V1ObjectMeta(
                name=f"{service_account_name}-token",
                annotations=annotations
            ),
            type="kubernetes.io/service-account-token"
        )
        try:
            self.core_v1_api.create_namespaced_secret(namespace, secret)
            logger.info(f"Secret {secret.metadata.name} created for Service Account {service_account_name}.")
        except ApiException as e:
            logger.error(f"Failed to create secret for Service Account {service_account_name}: {e}")
            raise

    def retrieve_service_account_token(self, service_account_name, namespace):
        """Retrieve the token for a specified service account."""
        from kubernetes.client.exceptions import ApiException
        import base64

        try:
            # List all secrets in the given namespace
            secrets = self.core_v1_api.list_namespaced_secret(namespace)
            for secret in secrets.items:
                # Check if the secret is a service account token and matches the specified service account
                if (secret.type == "kubernetes.io/service-account-token" and
                        secret.metadata.annotations.get('kubernetes.io/service-account.name') == service_account_name):
                    # Decode the token from base64
                    token = base64.b64decode(secret.data['token']).decode('utf-8')
                    return token
            # If no valid token was found
            raise Exception(
                f"No valid token found for Service Account {service_account_name} in namespace {namespace}.")
        except ApiException as e:
            raise Exception(f"Failed to retrieve token for Service Account {service_account_name}: {e}")

    def create_or_retrieve_service_account_token(self, service_account_name, namespace):
        """Create a new secret for a service account or retrieve an existing one."""
        import base64
        from kubernetes.client.exceptions import ApiException

        # Check if there is already a secret for this service account
        secrets = self.core_v1_api.list_namespaced_secret(namespace)
        for secret in secrets.items:
            if (secret.type == "kubernetes.io/service-account-token" and
                    secret.metadata.annotations.get('kubernetes.io/service-account.name') == service_account_name):
                # Check if the token data exists
                if 'token' in secret.data and secret.data['token']:
                    # Get the token from the secret
                    token = base64.b64decode(secret.data['token']).decode('utf-8')
                    logger.info(
                        f"Retrieved existing token for service account {service_account_name} in namespace {namespace}.")
                    return token
                else:
                    logger.error(f"Secret {secret.metadata.name} exists but does not contain a valid token.")

        # If no valid secret found, create a new one
        logger.info(f"No valid token found for Service Account {service_account_name}. Creating a new token.")
        try:
            self.create_service_account_secret(service_account_name, namespace)
            # Re-check the secrets after creating one
            secrets = self.core_v1_api.list_namespaced_secret(namespace)
            for secret in secrets.items:
                if (secret.type == "kubernetes.io/service-account-token" and
                        secret.metadata.annotations.get('kubernetes.io/service-account.name') == service_account_name):
                    if 'token' in secret.data and secret.data['token']:
                        token = base64.b64decode(secret.data['token']).decode('utf-8')
                        logger.info(
                            f"New token generated for service account {service_account_name} in namespace {namespace}.")
                        return token
        except ApiException as e:
            logger.error(f"Failed to create a new token for Service Account {service_account_name}: {e}")
            raise

        raise Exception(f"Failed to generate or retrieve a valid token for Service Account {service_account_name}.")

    def add_scc_to_service_account(self, service_account_name, namespace, scc_name):
        scc = self.dyn_client.resources.get(api_version='security.openshift.io/v1', kind='SecurityContextConstraints')
        scc_obj = scc.get(name=scc_name)
        user_name = f"system:serviceaccount:{namespace}:{service_account_name}"

        if user_name not in scc_obj.users:
            scc_obj.users.append(user_name)
            scc.patch(body=scc_obj, name=scc_name, content_type='application/merge-patch+json')
            logger.info(f"Added {user_name} to {scc_name} SCC.")
        else:
            logger.info(f"{user_name} is already in the {scc_name} SCC.")

        # Verification Step
        scc_obj = scc.get(name=scc_name)  # Re-fetch the SCC to ensure it was updated
        if user_name in scc_obj.users:
            logger.info(f"Verification successful: {user_name} is now in the {scc_name} SCC.")
        else:
            logger.warning(f"Verification failed: {user_name} is NOT in the {scc_name} SCC.")

    def create_cluster_role_binding_for_scc(self, service_account_name, namespace):
        cluster_role_binding = client.V1ClusterRoleBinding(
            metadata=client.V1ObjectMeta(name=f"{service_account_name}-scc-binding"),
            role_ref=client.V1RoleRef(
                api_group="rbac.authorization.k8s.io",
                kind="ClusterRole",
                name="cluster-admin"  # This can be 'cluster-admin' or a custom role with SCC permissions
            ),
            subjects=[{
                "kind": "ServiceAccount",
                "name": service_account_name,
                "namespace": namespace
            }]
        )

        try:
            self.rbac_api.create_cluster_role_binding(cluster_role_binding)
            logger.info(f"ClusterRoleBinding created for service account {service_account_name}.")
        except client.exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                logger.info(f"ClusterRoleBinding already exists for service account {service_account_name}.")
            else:
                logger.error(f"Failed to create ClusterRoleBinding: {e}")

    def delete_service_account_secret(self, service_account_name, namespace):
        # List all secrets in the namespace
        secrets = self.core_v1_api.list_namespaced_secret(namespace)

        # Iterate through the secrets and find the one associated with the service account
        for secret in secrets.items:
            if (secret.type == "kubernetes.io/service-account-token" and
                    secret.metadata.annotations['kubernetes.io/service-account.name'] == service_account_name):
                # Delete the secret
                secret_name = secret.metadata.name
                try:
                    self.core_v1_api.delete_namespaced_secret(secret_name, namespace)
                    logger.info(f"Secret {secret_name} associated with Service Account {service_account_name} deleted.")
                except ApiException as e:
                    if e.status == 404:
                        logger.info(f"Secret {secret_name} not found. It might have been deleted already.")
                    else:
                        logger.error(
                            f"Failed to delete secret {secret_name} for Service Account {service_account_name}: {e}")
                    raise
                return  # Once the secret is found and deleted, exit the method

        logger.info(f"No secret found for Service Account {service_account_name} in namespace {namespace}.")

    def delete_service_account(self, service_account_name, namespace):
        """
        Delete a service account in the given namespace. If the service account doesn't exist, log a message and continue.
        """
        # Append 'svc' to the service account name
        service_account_name = f"{service_account_name}svc"

        try:
            self.core_v1_api.delete_namespaced_service_account(service_account_name, namespace)
            logger.info(f"Service Account '{service_account_name}' deleted successfully in namespace '{namespace}'.")
        except ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"Service Account '{service_account_name}' not found in namespace '{namespace}'. Continuing...")
            else:
                logger.error(
                    f"Failed to delete Service Account '{service_account_name}' in namespace '{namespace}': {e}")
                raise

    @staticmethod
    def create_container(image_name, deployment_name=None, project_name=None, ports=None, pvc_mounts=None,
                         cpu_requests=None, cpu_limits=None, memory_requests=None, command=None,
                         memory_limits=None, gpu_requests=None, memory_storage_configs=None,
                         use_gpu=None, gpu_limits=None, security_context_config=None,
                         security_context=None, entrypoint_args=None, rs_env_vars=None, entrypoint_envs=None):

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

        # todo - fix this with the existing CommandConfig dataclass
        cmd = []
        args_ = []

        if command:
            # 1) If it's a dictionary
            if isinstance(command, dict):

                logger.error("Wrong type of command provided. It should be a string or CommandConfig dataclass.")
                raise TypeError("Command should be a string or CommandConfig dataclass.")

                # executable = command.get("executable")
                # if executable:
                #     cmd = [executable]
                # # arguments could be "arguments" or "args" in a dict. Decide which one you have
                # if "arguments" in command:
                #     args_ = command["arguments"]
                # elif "args" in command:
                #     args_ = command["args"]

            # 2) If it's a CommandConfig dataclass
            elif isinstance(command, CommandConfig):

                if command.executable:
                    cmd = [command.executable]
                args_ = command.arguments or []

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
                ))

        resources = {
            'requests': {},
            'limits': {}
        }

        if cpu_requests and cpu_limits:
            resources['requests']['cpu'] = K8SUnits(cpu_requests, resource_type="cpu").as_str
            resources['limits']['cpu'] = K8SUnits(cpu_limits, resource_type="cpu").as_str
        if memory_requests and memory_limits:
            resources['requests']['memory'] = K8SUnits(memory_requests, resource_type="memory").as_str
            resources['limits']['memory'] = K8SUnits(memory_limits, resource_type="memory").as_str
        # if gpu_requests and gpu_limits:
        if use_gpu is True:
            resources['requests']['nvidia.com/gpu'] = gpu_requests
            resources['limits']['nvidia.com/gpu'] = gpu_limits

        if security_context_config and isinstance(security_context_config, SecurityContextConfig):
            # convert to a dict so the rest of the code can do
            # security_context_config['enable_security_context'] etc.
            security_context_config = {
                'enable_security_context': security_context_config.enable_security_context,
                'privileged': security_context_config.privileged,
                'add_capabilities': security_context_config.add_capabilities,
                'runAsUser': security_context_config.runAsUser,
            }

        if security_context_config and security_context_config['enable_security_context']:
            security_context = {
                "capabilities": {
                    "add": security_context_config['add_capabilities']
                },
                "privileged": security_context_config['privileged']  # Setting the privileged status
            }

        return client.V1Container(
            name=container_name,
            image=image_name,
            command=cmd,
            args=args_,
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

    @staticmethod
    def create_pod_template(image_name, command=None, labels=None, deployment_name=None,
                            project_name=None, cron_job_name=None, job_schedule=None,
                            ports=None, create_service_account=None, service_account_name=None, pvc_mounts=None,
                            cpu_requests=None, cpu_limits=None, memory_requests=None, memory_storage_configs=None,
                            memory_limits=None, use_gpu=False, gpu_requests=None, restart_policy_configs=None,
                            gpu_limits=None, use_node_selector=None, node_selector=None,
                            security_context_config=None, entrypoint_args=None, entrypoint_envs=None):

        if labels is None:
            labels = {}
        if project_name:
            labels['project'] = project_name

        volumes = []

        # Handle memory_storage_configs
        if memory_storage_configs:
            for mem_storage in memory_storage_configs:
                if mem_storage.enabled is True:
                    memory_volume_spec = client.V1EmptyDirVolumeSource(medium="Memory")
                    memory_volume_spec.size_limit = mem_storage.size_gb.as_str
                    volumes.append(client.V1Volume(
                        name=mem_storage.name,
                        empty_dir=memory_volume_spec
                    ))
                else:
                    client.V1EmptyDirVolumeSource()

        # Handle PVC mounts
        if pvc_mounts:
            for mount in pvc_mounts:

                volumes.append(client.V1Volume(
                    name=mount['pvc_name'],
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name=mount['pvc_name'])
                ))

        # Call the create_container static method
        container = BeamK8S.create_container(
            image_name=image_name,
            command=command,
            deployment_name=deployment_name,
            project_name=project_name,
            ports=ports,
            pvc_mounts=pvc_mounts,
            cpu_requests=cpu_requests,
            cpu_limits=cpu_limits,
            memory_requests=memory_requests,
            memory_limits=memory_limits,
            memory_storage_configs=memory_storage_configs,
            use_gpu=use_gpu,
            gpu_requests=gpu_requests,
            gpu_limits=gpu_limits,
            security_context_config=security_context_config,
            entrypoint_args=entrypoint_args,
            entrypoint_envs=entrypoint_envs
        )

        # Initialize pod_spec without node_selector
        pod_spec = client.V1PodSpec(
            containers=[container],
            service_account_name=service_account_name,
            volumes=volumes
        )

        # Apply the restart policy if provided
        if restart_policy_configs:
            pod_spec.restart_policy = restart_policy_configs.condition
        else:
            raise ValueError(f"Unsupported restart condition: {restart_policy_configs.condition}")

        # Conditionally add node_selector if it's not None
        if use_node_selector is True:
            pod_spec.node_selector = node_selector

        return client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels=labels),
            spec=pod_spec
        )

    def create_pvc(self, pvc_name, pvc_size, pvc_access_mode, namespace, storage_class_name=None):
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
        if storage_class_name:
            pvc_manifest["spec"]["storageClassName"] = storage_class_name

        self.core_v1_api.create_namespaced_persistent_volume_claim(namespace=namespace, body=pvc_manifest)
        logger.info(f"Created PVC '{pvc_name}' in namespace '{namespace}'.")

    def create_deployment_spec(self, image_name, command=None, labels=None, deployment_name=None,
                               project_name=None, replicas=None, restart_policy_configs=None,
                               ports=None, create_service_account=None, service_account_name=None, storage_configs=None,
                               cpu_requests=None, cpu_limits=None, memory_requests=None, use_node_selector=None,
                               node_selector=None, memory_limits=None, use_gpu=None,
                               gpu_requests=None, gpu_limits=None, memory_storage_configs=None,
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
            command=command,
            labels=labels,
            deployment_name=deployment_name,
            project_name=project_name,
            ports=ports,
            use_node_selector=use_node_selector,
            node_selector=node_selector,
            create_service_account=create_service_account,
            service_account_name=service_account_name,  # Use it here
            pvc_mounts=pvc_mounts,  # Assuming pvc_mounts is prepared earlier in the method
            cpu_requests=cpu_requests,
            cpu_limits=cpu_limits,
            memory_storage_configs=memory_storage_configs,
            memory_requests=memory_requests,
            memory_limits=memory_limits,
            use_gpu=use_gpu,
            gpu_requests=gpu_requests,
            gpu_limits=gpu_limits,
            security_context_config=security_context_config,
            restart_policy_configs=restart_policy_configs,
            entrypoint_args=entrypoint_args,
            entrypoint_envs=entrypoint_envs
        )

        # Create and return the deployment spec
        return client.V1DeploymentSpec(
            replicas=int(replicas),  # Ensure replicas is an int
            template=pod_template,
            selector={'matchLabels': pod_template.metadata.labels}
        )

    def deployment_exists(self, deployment_name, namespace):
        """
        Check if a deployment exists in the specified namespace.
        """
        try:
            self.apps_v1_api.read_namespaced_deployment(name=deployment_name, namespace=namespace)
            logger.info(f"Deployment '{deployment_name}' exists in namespace '{namespace}'.")
            return True
        except ApiException as e:
            if e.status == 404:
                logger.info(f"Deployment '{deployment_name}' not found in namespace '{namespace}', skipping.")
                return False
            else:
                logger.error(f"Unexpected error while checking deployment '{deployment_name}': {e}")
                raise

    def get_relevant_configuration(self, key, value, config=None):
        """
        Get the relevant configuration value for the given key from the config object.
        """
        if value is None and config is not None:
            default = getattr(config, key, None)
            value = config.get(key, default=default)

        return  value


    def create_deployment(self, image_name, config=None, command=None, labels=None, deployment_name=None,
                          namespace=None, project_name=None,
                          replicas=None, ports=None, create_service_account=None, service_account_name=None,
                          storage_configs=None, cpu_requests=None, cpu_limits=None, memory_requests=None,
                          use_node_selector=None, node_selector=None, memory_storage_configs=None,
                          memory_limits=None, use_gpu=False, gpu_requests=None, gpu_limits=None,
                          security_context_config=None, restart_policy_configs=None,
                          entrypoint_args=None, entrypoint_envs=None):

        if namespace is None:
            namespace = self.namespace

        # project_name = self.get_relevant_configuration('project_name', project_name, config)
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
        else:
            deployment_name = self.generate_unique_deployment_name(deployment_name, namespace)

        # Ensure deployment name complies with RFC 1123
        deployment_name = ensure_rfc1123_compliance(deployment_name)

        deployment_spec = self.create_deployment_spec(
            image_name, command=command, labels=labels, deployment_name=deployment_name,
            project_name=project_name, replicas=replicas, ports=ports, create_service_account=create_service_account,
            service_account_name=service_account_name, use_node_selector=use_node_selector, node_selector=node_selector,
            storage_configs=storage_configs, cpu_requests=cpu_requests, cpu_limits=cpu_limits,
            memory_requests=memory_requests, memory_limits=memory_limits,
            memory_storage_configs=memory_storage_configs, use_gpu=use_gpu,
            gpu_requests=gpu_requests, gpu_limits=gpu_limits, restart_policy_configs=restart_policy_configs,
            security_context_config=security_context_config,
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

    #
    def apply_deployment(self, deployment, namespace=None):
        if namespace is None:
            namespace = self.namespace

        try:
            self.apps_v1_api.create_namespaced_deployment(body=deployment, namespace=namespace)
            logger.info(f"Successfully applied deployment in namespace '{namespace}'")

            selector = self.get_selector_from_deployment(deployment)

            # Wait briefly for pods to be created
            time.sleep(5)  # Adjust this value as needed

            pod_list = self.core_v1_api.list_namespaced_pod(namespace=namespace, label_selector=selector)
            pod_infos = [self.extract_pod_info(pod) for pod in pod_list.items if pod.metadata.labels is not None]

            # logger.info(f"Pod infos: '{pod_infos}'")
            return pod_infos

        except ApiException as e:
            logger.exception(f"Exception when applying the deployment: {e}")
            return None

    @staticmethod
    def get_selector_from_deployment(deployment):
        print("Debugging get_selector_from_deployment method")
        print(f"Received deployment object type: {type(deployment)}")

        try:
            # Convert the Kubernetes client object to a dictionary if it's not already a dict
            if not isinstance(deployment, dict):
                print("Converting Kubernetes client object to dictionary")
                deployment_dict = deployment.to_dict()
            else:
                deployment_dict = deployment

            # Now that we have a dictionary, access the matchLabels
            match_labels = deployment_dict.get('spec', {}).get('selector', {}).get('matchLabels', {})
            print(f"Extracted matchLabels: {match_labels}")

            # Construct the selector string from matchLabels
            selector_str = ','.join([f'{k}={v}' for k, v in match_labels.items()])
            print(f"Selector string: {selector_str}")
        except Exception as e:
            print(f"Error extracting selector from deployment: {e}")
            selector_str = ""

        return selector_str

    def statefulset_exists(self, statefulset_name, namespace):
        """
        Check if a StatefulSet exists in the specified namespace.
        """
        try:
            self.apps_v1_api.read_namespaced_stateful_set(name=statefulset_name, namespace=namespace)
            logger.info(f"StatefulSet '{statefulset_name}' exists in namespace '{namespace}'.")
            return True
        except ApiException as e:
            if e.status == 404:
                logger.info(f"StatefulSet '{statefulset_name}' not found in namespace '{namespace}', skipping.")
                return False
            else:
                logger.error(f"Unexpected error while checking StatefulSet '{statefulset_name}': {e}")
                raise

    def create_statefulset(self, image_name, command=None, labels=None, statefulset_name=None,
                           namespace=None, project_name=None, replicas=None, ports=None,
                           volume_claims=None, entrypoint_args=None, entrypoint_envs=None):
        if namespace is None:
            namespace = self.namespace

        if project_name is None:
            project_name = self.project_name

        # Generate a unique name for the StatefulSet if it's not provided
        if statefulset_name is None:
            statefulset_name = self.generate_unique_deployment_name(base_name=image_name.split(':')[0], namespace=namespace)
            if labels is None:
                labels = {}
            labels['app'] = statefulset_name  # Set the 'app' label
        else:
            statefulset_name = self.generate_unique_deployment_name(statefulset_name, namespace)

        # Ensure StatefulSet name complies with RFC 1123
        statefulset_name = ensure_rfc1123_compliance(statefulset_name)

        # Create the StatefulSet spec using create_statefulset_spec
        statefulset_spec = self.create_statefulset_spec(
            image_name=image_name,
            command=command,
            labels=labels,
            statefulset_name=statefulset_name,
            project_name=project_name,
            replicas=replicas,
            ports=ports,
            storage_configs=volume_claims,  # Assuming `volume_claims` matches `storage_configs` structure
            entrypoint_args=entrypoint_args,
            entrypoint_envs=entrypoint_envs
        )

        # Build StatefulSet metadata
        statefulset_metadata = client.V1ObjectMeta(
            name=statefulset_name,
            namespace=namespace,
            labels={"project": project_name}
        )

        # Create StatefulSet object
        statefulset = client.V1StatefulSet(
            api_version="apps/v1",
            kind="StatefulSet",
            metadata=statefulset_metadata,
            spec=statefulset_spec
        )

        logger.info(f"Created StatefulSet '{statefulset_name}' in namespace '{namespace}'.")
        return statefulset

    def create_statefulset_spec(self, image_name, command=None, labels=None, statefulset_name=None,
                                project_name=None, replicas=None, restart_policy_configs=None,
                                ports=None, create_service_account=None, service_account_name=None,
                                storage_configs=None,
                                cpu_requests=None, cpu_limits=None, memory_requests=None, use_node_selector=None,
                                node_selector=None, memory_limits=None, use_gpu=None,
                                gpu_requests=None, gpu_limits=None, memory_storage_configs=None,
                                security_context_config=None, entrypoint_args=None,
                                entrypoint_envs=None):
        # Ensure pvc_mounts are prepared correctly from storage_configs if needed
        pvc_mounts = [
            {
                'pvc_name': sc.pvc_name,
                'mount_path': sc.pvc_mount_path
            } for sc in storage_configs if sc.create_pvc
        ] if storage_configs else []

        # Create the pod template with correct arguments
        pod_template = self.create_pod_template(
            image_name=image_name,
            command=command,
            labels=labels,
            deployment_name=statefulset_name,  # Reuse the StatefulSet name
            project_name=project_name,
            ports=ports,
            use_node_selector=use_node_selector,
            node_selector=node_selector,
            create_service_account=create_service_account,
            service_account_name=service_account_name,
            pvc_mounts=pvc_mounts,
            cpu_requests=cpu_requests,
            cpu_limits=cpu_limits,
            memory_storage_configs=memory_storage_configs,
            memory_requests=memory_requests,
            memory_limits=memory_limits,
            use_gpu=use_gpu,
            gpu_requests=gpu_requests,
            gpu_limits=gpu_limits,
            security_context_config=security_context_config,
            restart_policy_configs=restart_policy_configs,
            entrypoint_args=entrypoint_args,
            entrypoint_envs=entrypoint_envs
        )

        # Create volume claim templates for StatefulSet
        volume_claim_templates = [
            client.V1PersistentVolumeClaim(
                metadata=client.V1ObjectMeta(name=vc['pvc_name']),
                spec=vc['spec']
            ) for vc in storage_configs if vc.create_pvc
        ] if storage_configs else []

        # Create and return the StatefulSet spec
        return client.V1StatefulSetSpec(
            replicas=int(replicas) if replicas else 1,  # Ensure replicas is an int
            template=pod_template,
            selector=client.V1LabelSelector(match_labels=pod_template.metadata.labels),
            service_name=f"{statefulset_name}-service",  # Service name is mandatory for StatefulSets
            volume_claim_templates=volume_claim_templates
        )

    def scale_statefulset_to_zero(self, statefulset_name, namespace):
        """
        Scale down the StatefulSet to zero replicas.
        """
        try:
            self.apps_v1_api.patch_namespaced_stateful_set_scale(
                name=statefulset_name,
                namespace=namespace,
                body={'spec': {'replicas': 0}}
            )
            logger.info(f"Scaled down StatefulSet '{statefulset_name}' to zero replicas.")
        except ApiException as e:
            logger.error(f"Failed to scale StatefulSet '{statefulset_name}' to zero replicas: {e}")

    def cleanup_statefulsets(self, namespace, app_name):
        """
        Cleanup StatefulSets associated with the application.
        """
        if self.statefulset_exists(app_name, namespace):
            self.scale_statefulset_to_zero(app_name, namespace)
            self.delete_statefulsets_by_name(app_name, namespace)
            logger.info(f"StatefulSet '{app_name}' cleaned up successfully in namespace '{namespace}'.")
        else:
            logger.info(f"No StatefulSets found for '{app_name}' in namespace '{namespace}' to clean up.")

    def delete_statefulsets_by_name(self, app_name, namespace):
        """
        Delete a StatefulSet by name or associated StatefulSets by app label in the specified namespace.
        """
        try:
            # Check if a specific StatefulSet by name exists
            statefulset = None
            try:
                statefulset = self.apps_v1_api.read_namespaced_stateful_set(app_name, namespace)
            except ApiException as e:
                if e.status != 404:
                    logger.error(f"Failed to retrieve StatefulSet '{app_name}' in namespace '{namespace}': {e}")
                    return

            if statefulset:
                # If a StatefulSet with the given name exists, delete it
                self.apps_v1_api.delete_namespaced_stateful_set(
                    name=app_name,
                    namespace=namespace,
                    body=client.V1DeleteOptions()
                )
                logger.info(f"Deleted StatefulSet '{app_name}' in namespace '{namespace}'.")
                return

            # If no specific StatefulSet name, use label selector to delete associated StatefulSets
            statefulsets = self.apps_v1_api.list_namespaced_stateful_set(namespace, label_selector=f"app={app_name}")
            for sts in statefulsets.items:
                self.apps_v1_api.delete_namespaced_stateful_set(
                    name=sts.metadata.name,
                    namespace=namespace,
                    body=client.V1DeleteOptions()
                )
                logger.debug(f"Deleted StatefulSet '{sts.metadata.name}' in namespace '{namespace}'.")
            logger.info(f"Deleted all StatefulSets associated with app '{app_name}' in namespace '{namespace}'.")

        except ApiException as e:
            logger.error(f"Failed to delete StatefulSets for '{app_name}' in namespace '{namespace}': {e}")

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

    def get_pods_by_label(self, labels, namespace: str):
        """
        Retrieve pods matching the given labels within the specified namespace.

        :param labels: Labels to match (usually a dict from the config).
        :param namespace: Namespace to search within.
        :return: List of V1Pod objects matching the labels.
        """
        # Ensure that the namespace is a string
        if not isinstance(namespace, str):
            logger.error(f"Expected namespace to be a string, but got {type(namespace).__name__}.")
            return []

        try:
            if isinstance(labels, dict):
                label_selector = ",".join([f"{key}={value}" for key, value in labels.items()])
            elif isinstance(labels, str):
                label_selector = labels
            else:
                logger.error(f"Unsupported labels format: {type(labels)}. Expected dict or str.")
                return []

            logger.debug(f"Using label selector: {label_selector} in namespace: {namespace}")

            # Retrieve the pods
            pods = self.core_v1_api.list_namespaced_pod(namespace=namespace, label_selector=label_selector)

            if not pods.items:
                logger.warning("No pods found with the given label selector.")
                return []

            logger.debug(f"Retrieved {len(pods.items)} pods.")
            return pods.items  # Returns a list of V1Pod objects

        except kubernetes.client.exceptions.ApiException as e:
            if e.status == 401:
                logger.error(f"Unauthorized access while retrieving pods: {str(e)}")
            else:
                logger.exception(f"Error retrieving pods by label: {str(e)}")
            return []
        except Exception as e:
            logger.exception(f"Unexpected error retrieving pods by label: {str(e)}")
            return []

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
            logger.info(f"Service '{service_name}' created successfully in namespace '{namespace}'.")
            logger.info(
                f"Service '{service_name}' of type '{service_type}' created with ports: "
                f"{', '.join([f'Port: {port.port}, TargetPort: {port.target_port}' for port in service_ports])}")
        except client.exceptions.ApiException as e:
            logger.info(f"Failed to create service '{service_name}' in namespace '{namespace}': {e}")

        service_url = f"http://{base_name}.{namespace}.svc.cluster.local:{ports[0]}"
        service_details = {
            'url': service_url,
            'name': base_name,
            'ports': ports  # List of ports
        }
        return service_details

    def create_route(self, service_name, namespace, protocol, port, annotations, route_timeout=None):
        from openshift.dynamic.exceptions import NotFoundError
        from openshift.dynamic import DynamicClient

        dyn_client = DynamicClient(self.api_client)

        # Get the Route resource from the OpenShift API
        route_resource = dyn_client.resources.get(api_version='route.openshift.io/v1', kind='Route')

        try:
            # Try to get the existing route
            existing_route = route_resource.get(name=service_name, namespace=namespace)
            if existing_route:  # If the route exists, log a message and return
                logger.warning(f"Route {service_name} already exists in namespace {namespace}, skipping creation.")
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
                "namespace": namespace,
                "annotations": annotations or {}
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

        # if route_timeout:
        #     route_manifest["metadata"]["annotations"] = {"haproxy.router.openshift.io/timeout": route_timeout}

        # Add TLS termination if protocol is 'https'
        #  if protocol.lower() == 'https':
        if annotations and annotations.get("route.openshift.io/termination") == "passthrough":
            route_manifest["spec"]["tls"] = {
                "termination": "passthrough"
            }

        # Attempt to create the route
        try:
            created_route = route_resource.create(body=route_manifest, namespace=namespace)
            logger.info(f"Route for service {service_name} created successfully in namespace {namespace}.")
            # Print the DNS name of the route
            dns_name = created_route.spec.host  # Accessing the DNS name from the route response
            logger.info(f"The DNS name of the created route is: {dns_name}")
            route_details = {
                'host': created_route.spec.host,
                'name': service_name,
                "annotations": annotations or {},
            }
            return route_details
        except Exception as e:
            logger.error(f"Failed to create route for service {service_name} in namespace {namespace}: {e}")

    def generate_route_service_variables(self):
        rs_env_vars = []
        services_info = self.get_services_info(self.namespace)
        routes_info = self.get_routes_info(self.namespace)

        for service in services_info:
            rs_env_vars.append(client.V1EnvVar(
                name=f"SERVICE_{service['service_name'].upper()}_URL",
                value=f"http://{service['service_name']}.{self.namespace}:{service['port']}"
            ))

        for route in routes_info:
            rs_env_vars.append(client.V1EnvVar(
                name=f"ROUTE_{route['host'].replace('.', '_').upper()}_URL",
                value=f"http://{route['host']}"
            ))

        return rs_env_vars

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

    def create_job(self, namespace, job_name, image_name, container_name, command,
                   entrypoint_args, entrypoint_envs, cpu_requests, cpu_limits, memory_requests, memory_limits,
                   use_gpu, gpu_requests, gpu_limits, labels, node_selector,
                   use_node_selector, storage_configs, security_context_config, restart_policy_configs):

        pvc_mounts = [{
            'pvc_name': sc.pvc_name,
            'mount_path': sc.pvc_mount_path
        } for sc in storage_configs if sc.create_pvc] if storage_configs else []

        # Create the container definition
        container = self.create_container(
            image_name=image_name,
            command=command,
            pvc_mounts=pvc_mounts,
            cpu_requests=cpu_requests,
            cpu_limits=cpu_limits,
            memory_requests=memory_requests,
            memory_limits=memory_limits,
            use_gpu=use_gpu,
            gpu_requests=gpu_requests,
            gpu_limits=gpu_limits,
            security_context_config=security_context_config,
            entrypoint_args=entrypoint_args,
            entrypoint_envs=entrypoint_envs
        )

        if restart_policy_configs.condition == "Always":
            restart_policy_configs.condition = "OnFailure"



        # Create the pod spec
        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy=restart_policy_configs.condition,
        )

        if use_node_selector is True:
            pod_spec.node_selector = node_selector

        # Create the pod template
        pod_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels=labels),
            spec=pod_spec
        )

        # Create the job spec
        job_spec = client.V1JobSpec(
            template=pod_template,
            backoff_limit=restart_policy_configs.max_attempts
        )

        # Create the job metadata
        job_metadata = client.V1ObjectMeta(name=job_name, namespace=namespace, labels={"project": namespace})

        # Create the job definition
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=job_metadata,
            spec=job_spec
        )

        try:
            self.batch_v1_api.create_namespaced_job(body=job, namespace=namespace)
            logger.info(f"Job '{job_name}' created successfully in namespace '{namespace}'.")

            return self.get_pods_by_label(labels, namespace)
        except ApiException as e:
            logger.error(f"Failed to create Job '{job_name}' in namespace '{namespace}': {e}")
            return None

    # def create_cron_job(self, namespace, cron_job_name, image_name, container_name, job_schedule, command,
    #                     entrypoint_args, entrypoint_envs, cpu_requests, cpu_limits, memory_requests, memory_limits,
    #                     use_gpu, gpu_requests, gpu_limits, labels, node_selector,
    #                     use_node_selector, storage_configs, security_context_config, restart_policy_configs):

    def create_cron_job(self, config):

        pvc_mounts = [{'pvc_name': sc['pvc_name'], 'mount_path': sc['pvc_mount_path']}
                      for sc in config.storage_configs if sc['create_pvc']] if config.storage_configs else []

        # Create the container definition
        container = self.create_container(
            image_name=config.image_name,
            command=config.command,
            pvc_mounts=pvc_mounts,
            cpu_requests=config.cpu_requests,
            cpu_limits=config.cpu_limits,
            memory_requests=config.memory_requests,
            memory_limits=config.memory_limits,
            use_gpu=config.use_gpu,
            gpu_requests=config.gpu_requests,
            gpu_limits=config.gpu_limits,
            security_context_config=config.security_context_config,
            entrypoint_args=config.entrypoint_args,
            entrypoint_envs=config.entrypoint_envs
        )

        if config.restart_policy_configs['condition'] == "Always":
            config.restart_policy_configs['condition'] = "OnFailure"

        # Create the pod template spec
        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy=config.restart_policy_configs['condition'],
        )

        if config.use_node_selector is True:
            pod_spec.node_selector = config.node_selector

        # Create the pod template
        pod_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels=config.labels),
            spec=pod_spec
        )

        # Create the job spec
        job_spec = client.V1JobSpec(
            template=pod_template,
            backoff_limit=config.restart_policy_configs['max_attempts'],
            active_deadline_seconds=config.restart_policy_configs['active_deadline_seconds']
        )

        # Create the cron job spec
        cron_job_spec = client.V1CronJobSpec(
            schedule=config.job_schedule,
            job_template=client.V1JobTemplateSpec(
                metadata=client.V1ObjectMeta(labels=config.labels),
                spec=job_spec
            )
        )

        cron_job_metadata = client.V1ObjectMeta(name=config.cron_job_name, namespace=config.project_name, labels={"project": config.project_name})

        # Create the cron job definition
        cron_job = client.V1CronJob(
            api_version="batch/v1",
            kind="CronJob",
            metadata=cron_job_metadata,
            spec=cron_job_spec
        )

        try:
            self.batch_v1_api.create_namespaced_cron_job(body=cron_job, namespace=config.project_name)
            logger.info(f"CronJob '{config.cron_job_name}' created successfully in namespace '{config.project_name}'.")

            return self.get_pods_by_label(config.labels, config.project_name)
        except ApiException as e:
            logger.error(f"Failed to create CronJob '{config.cron_job_name}' in namespace '{config.project_name}': {e}")
            return None

    def monitor_job(self, job_name, namespace):
        """
        Monitor the status of a Job until it completes, fails, or hits the backoff limit.
        Fetch and display logs after job completion.
        """
        try:
            job = self.batch_v1_api.read_namespaced_job_status(job_name, namespace)
            while job.status.active or not (job.status.succeeded or job.status.failed):
                logger.info(f"Job '{job_name}' is still active. Waiting for completion...")
                time.sleep(10)
                job = self.batch_v1_api.read_namespaced_job_status(job_name, namespace)

            if job.status.succeeded:
                logger.info(f"Job '{job_name}' completed successfully.")
                # Fetch and display the logs for the job's pods
                logs = self.get_job_logs(job_name=job_name, namespace=namespace)
                if logs:
                    logger.info(f"Logs for Job '{job_name}':\n{logs}")
            elif job.status.failed:
                logger.error(f"Job '{job_name}' failed.")
        except ApiException as e:
            logger.error(f"Failed to monitor Job '{job_name}' in namespace '{namespace}': {e}")
            return None

    def monitor_cron_job(self, cron_job_name, namespace):
        """
        Monitor the status of a CronJob. Checks the Job spawned by the CronJob to ensure its completion.
        Fetch logs of each job spawned by the CronJob.
        """
        try:
            # Get the most recent Job created by the CronJob
            cron_job = self.batch_v1_api.read_namespaced_cron_job(cron_job_name, namespace)
            job_selector = cron_job.spec.job_template.spec.template.metadata.labels

            # Poll for Job status associated with the CronJob
            while True:
                jobs = self.batch_v1_api.list_namespaced_job(namespace, label_selector=','.join(
                    [f"{k}={v}" for k, v in job_selector.items()]))
                if jobs.items:
                    for job in jobs.items:
                        self.monitor_job(job.metadata.name, namespace)
                        logger.info(f"Monitored Job '{job.metadata.name}' triggered by CronJob '{cron_job_name}'")

                        # Fetch and display logs for each job
                        logs = self.get_job_logs(job_name=job.metadata.name, namespace=namespace)
                        if logs:
                            logger.info(
                                f"Logs for Job '{job.metadata.name}' triggered by CronJob '{cron_job_name}':\n{logs}")
                else:
                    logger.info(f"No active Jobs found for CronJob '{cron_job_name}'.")
                time.sleep(30)  # Poll every 30 seconds
        except ApiException as e:
            logger.error(f"Failed to monitor CronJob '{cron_job_name}' in namespace '{namespace}': {e}")
            return None

    def get_job_logs(self, job_name, namespace):
        """
        Get logs from the Pods associated with a Job.
        """
        try:
            # Pods are labeled with `job-name` to associate them with the job
            pods = self.get_pods_by_label({'job-name': job_name}, namespace)
            if not pods:
                logger.error(f"No pods found for Job '{job_name}' in namespace '{namespace}'.")
                return None

            logs = ""
            for pod in pods:
                pod_logs = self.core_v1_api.read_namespaced_pod_log(name=pod.metadata.name, namespace=namespace)
                logger.info(f"Logs from pod '{pod.metadata.name}':\n{pod_logs}")
                logs += pod_logs + "\n"

            return logs
        except ApiException as e:
            logger.error(f"Failed to retrieve logs for Job '{job_name}': {e}")
            return None

    def get_services_info(self, namespace):
        services_info = []
        try:
            services = self.core_v1_api.list_namespaced_service(namespace=namespace)
            for service in services.items:
                if service.spec.type == "NodePort":
                    for port in service.spec.ports:
                        service_info = {
                            "service_name": service.metadata.name,
                            "cluster_ip": service.spec.cluster_ip,
                            "port": port.port,
                            "node_port": port.node_port,
                        }
                        services_info.append(service_info)
                else:
                    for port in service.spec.ports:
                        service_info = {
                            "service_name": service.metadata.name,
                            "cluster_ip": service.spec.cluster_ip,
                            "port": port.port,
                        }
                        services_info.append(service_info)
        except ApiException as e:
            logger.error(f"Failed to get services info for namespace '{namespace}': {e}")
            services_info = None
        return services_info

    def get_routes_info(self, namespace):
        routes_info = []
        try:
            route_resource = self.dyn_client.resources.get(api_version='route.openshift.io/v1', kind='Route')
            routes = route_resource.get(namespace=namespace)
            for route in routes.items:
                route_info = {
                    "route_name": route.metadata.name,
                    "host": route.spec.host,
                }
                routes_info.append(route_info)
        except ApiException as e:
            logger.error(f"Failed to get routes info for namespace '{namespace}': {e}")
            routes_info = None
        return routes_info

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

    def scale_deployment_to_zero(self, deployment_name, namespace):
        """
        Scale down the deployment to zero replicas.
        """
        try:
            self.apps_v1_api.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace=namespace,
                body={'spec': {'replicas': 0}}
            )
            logger.info(f"Scaled down deployment '{deployment_name}' to zero.")
        except ApiException as e:
            logger.error(f"Failed to scale deployment '{deployment_name}' to zero: {e}")

    def delete_all_resources_by_app_label(self, app_name, deployment_name, namespace):
        """
        Delete all Kubernetes resources labeled with the app name.
        """
        try:
            # Delete all objects labeled with app={app_name}
            pods = self.core_v1_api.delete_collection_namespaced_pod(
                namespace=namespace,
                label_selector=f"app={app_name}"
            )
            logger.debug(f"Deleting pods with app label '{app_name}' in namespace '{namespace}'.")
            logger.info(f"Deleted all pods labeled with app '{app_name}' in namespace '{namespace}'.")
        except ApiException as e:
            logger.error(f"Failed to delete resources labeled with app '{app_name}' in namespace '{namespace}': {e}")

    def job_exists(self, job_name, namespace):
        """
        Check if a Job exists in the specified namespace.
        """
        try:
            self.batch_v1_api.read_namespaced_job(name=job_name, namespace=namespace)
            logger.info(f"Job '{job_name}' exists in namespace '{namespace}'.")
            return True
        except ApiException as e:
            if e.status == 404:
                logger.info(f"Job '{job_name}' not found in namespace '{namespace}', skipping.")
                return False
            else:
                logger.error(f"Unexpected error while checking job '{job_name}': {e}")
                raise

    def cronjob_exists(self, cronjob_name, namespace):
        """
        Check if a CronJob exists in the specified namespace.
        """
        try:
            self.batch_v1_api.read_namespaced_cron_job(name=cronjob_name, namespace=namespace)
            logger.info(f"CronJob '{cronjob_name}' exists in namespace '{namespace}'.")
            return True
        except ApiException as e:
            if e.status == 404:
                logger.info(f"CronJob '{cronjob_name}' not found in namespace '{namespace}', skipping.")
                return False
            else:
                logger.error(f"Unexpected error while checking cronjob '{cronjob_name}': {e}")
                raise

    def delete_cronjobs_by_name(self, app_name, namespace):
        try:
            # Check if a specific CronJob by name exists
            cronjob = None
            try:
                cronjob = self.batch_v1_api.read_namespaced_cron_job(app_name, namespace)
            except ApiException as e:
                if e.status != 404:
                    logger.error(f"Failed to retrieve CronJob '{app_name}' in namespace '{namespace}': {e}")
                    return

            if cronjob:
                # If a CronJob with the given name exists, delete it
                self.batch_v1_api.delete_namespaced_cron_job(app_name, namespace)
                logger.info(f"Deleted CronJob '{app_name}' in namespace '{namespace}'.")
                return

            # If no specific CronJob name, use label selector to delete associated CronJobs
            cronjobs = self.batch_v1_api.list_namespaced_cron_job(namespace, label_selector=f"app={app_name}")
            for cronjob in cronjobs.items:
                self.batch_v1_api.delete_namespaced_cron_job(cronjob.metadata.name, namespace)
                logger.debug(f"Deleted CronJob '{cronjob.metadata.name}' in namespace '{namespace}'.")
            logger.info(f"Deleted all CronJobs associated with app '{app_name}' in namespace '{namespace}'.")

        except ApiException as e:
            logger.error(f"Failed to delete CronJobs for '{app_name}' in namespace '{namespace}': {e}")


    def delete_jobs_by_name(self, app_name, namespace, wait_for_deletion=True, timeout=60):
        try:
            job = None
            try:
                job = self.batch_v1_api.read_namespaced_job(app_name, namespace)
            except ApiException as e:
                if e.status != 404:
                    logger.error(f"Failed to retrieve Job '{app_name}' in namespace '{namespace}': {e}")
                    return

            if job:
                self.batch_v1_api.delete_namespaced_job(
                    app_name,
                    namespace,
                    body=kubernetes.client.V1DeleteOptions(propagation_policy='Foreground')
                )
                logger.info(f"Deleted Job '{app_name}' in namespace '{namespace}'.")

                if wait_for_deletion:
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        try:
                            self.batch_v1_api.read_namespaced_job(app_name, namespace)
                            time.sleep(1)  # Wait for 1 second before re-checking
                        except ApiException as e:
                            if e.status == 404:
                                logger.info(f"Job '{app_name}' successfully deleted.")
                                break
                            else:
                                logger.error(f"Error checking deletion status for Job '{app_name}': {e}")
                                break
                    else:
                        logger.warning(f"Timed out waiting for Job '{app_name}' to be deleted.")
                return

            # If no specific Job name, use label selector to delete associated Jobs
            jobs = self.batch_v1_api.list_namespaced_job(namespace, label_selector=f"app={app_name}")
            for job in jobs.items:
                self.batch_v1_api.delete_namespaced_job(
                    job.metadata.name,
                    namespace,
                    body=kubernetes.client.V1DeleteOptions(propagation_policy='Foreground')
                )
                logger.debug(f"Deleted Job '{job.metadata.name}' in namespace '{namespace}'.")

                if wait_for_deletion:
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        try:
                            self.batch_v1_api.read_namespaced_job(job.metadata.name, namespace)
                            time.sleep(1)
                        except ApiException as e:
                            if e.status == 404:
                                logger.info(f"Job '{job.metadata.name}' successfully deleted.")
                                break
                            else:
                                logger.error(f"Error checking deletion status for Job '{job.metadata.name}': {e}")
                                break
                    else:
                        logger.warning(f"Timed out waiting for Job '{job.metadata.name}' to be deleted.")

            logger.info(f"Deleted all Jobs associated with app '{app_name}' in namespace '{namespace}'.")

        except ApiException as e:
            logger.error(f"Failed to delete Jobs for '{app_name}' in namespace '{namespace}': {e}")

    def cleanup_cronjobs(self, namespace, app_name):
        """
        Cleanup CronJobs associated with the application.
        """
        if self.cronjob_exists(app_name, namespace):
            self.delete_cronjobs_by_name(namespace, app_name)
        else:
            logger.info(f"No CronJobs found for '{app_name}' in namespace '{namespace}' to clean up.")

    def cleanup_jobs(self, namespace, app_name):
        """
        Cleanup Jobs associated with the application.
        """
        if self.job_exists(app_name, namespace):
            self.delete_jobs_by_name(app_name, namespace)
        else:
            logger.info(f"No Jobs found for '{app_name}' in namespace '{namespace}' to clean up.")

    def delete_services_by_deployment(self, deployment_name, namespace):
        try:
            services = self.core_v1_api.list_namespaced_service(namespace, label_selector=f"app={deployment_name}")
            for service in services.items:
                logger.debug(f"Deleting service: {service.metadata.name}")
                self.core_v1_api.delete_namespaced_service(service.metadata.name, namespace)
                logger.info(
                    f"Deleted service '{service.metadata.name}' associated with deployment '{deployment_name}'.")
        except ApiException as e:
            logger.error(f"Failed to delete Services associated with deployment '{deployment_name}': {e}")

    def delete_routes_by_deployment_name(self, deployment_name, namespace):
        """
        Delete all routes associated with the deployment name.
        """
        try:
            routes = self.custom_objects_api.list_namespaced_custom_object(
                group="route.openshift.io", version="v1", namespace=namespace, plural="routes"
            )
            for route in routes.get('items', []):
                if deployment_name in route['metadata']['name']:
                    self.custom_objects_api.delete_namespaced_custom_object(
                        group="route.openshift.io", version="v1", namespace=namespace,
                        plural="routes", name=route['metadata']['name']
                    )
                    logger.debug(f"Deleted route '{route['metadata']['name']}' in namespace '{namespace}'.")
            logger.info(
                f"Deleted all routes associated with deployment '{deployment_name}' in namespace '{namespace}'.")
        except ApiException as e:
            logger.error(f"Failed to delete routes associated with deployment '{deployment_name}': {e}")

    def delete_configmap_by_deployment(self, deployment_name, namespace):
        try:
            logger.debug(f"Deleting configmap: {deployment_name}-config")
            self.core_v1_api.delete_namespaced_config_map(f"{deployment_name}-config", namespace)
            logger.info(f"Deleted ConfigMap '{deployment_name}-config'.")
        except ApiException as e:
            if e.status == 404:
                logger.warning(f"ConfigMap '{deployment_name}-config' not found in namespace '{namespace}'.")
            else:
                logger.error(f"Failed to delete ConfigMap '{deployment_name}-config': {e}")

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

    def delete_resources_starting_with(self, prefix, namespace):
        """
        Deletes all resources in the given namespace that have names starting with the specified prefix.
        """
        try:
            # Delete pods starting with the prefix
            pods = self.core_v1_api.list_namespaced_pod(namespace=namespace)
            for pod in pods.items:
                if pod.metadata.name.startswith(prefix):
                    logger.debug(f"Deleting pod: {pod.metadata.name}")
                    self.core_v1_api.delete_namespaced_pod(pod.metadata.name, namespace)
                    logger.info(f"Deleted pod '{pod.metadata.name}' in namespace '{namespace}'")

            # Delete services starting with the prefix
            services = self.core_v1_api.list_namespaced_service(namespace=namespace)
            for service in services.items:
                if service.metadata.name.startswith(prefix):
                    logger.debug(f"Deleting service: {service.metadata.name}")
                    self.core_v1_api.delete_namespaced_service(service.metadata.name, namespace)
                    logger.info(f"Deleted service '{service.metadata.name}' in namespace '{namespace}'")

            # Delete configmaps starting with the prefix
            configmaps = self.core_v1_api.list_namespaced_config_map(namespace=namespace)
            for configmap in configmaps.items:
                if configmap.metadata.name.startswith(prefix):
                    logger.debug(f"Deleting configmap: {configmap.metadata.name}")
                    self.core_v1_api.delete_namespaced_config_map(configmap.metadata.name, namespace)
                    logger.info(f"Deleted configmap '{configmap.metadata.name}' in namespace '{namespace}'")

            # Delete deployments starting with the prefix
            deployments = self.apps_v1_api.list_namespaced_deployment(namespace=namespace)
            for deployment in deployments.items:
                if deployment.metadata.name.startswith(prefix):
                    logger.debug(f"Deleting deployment: {deployment.metadata.name}")
                    self.apps_v1_api.delete_namespaced_deployment(deployment.metadata.name, namespace)
                    logger.info(f"Deleted deployment '{deployment.metadata.name}' in namespace '{namespace}'")

            # Delete cronjobs starting with the prefix
            cronjobs = self.batch_v1_api.list_namespaced_cron_job(namespace=namespace)
            for cronjob in cronjobs.items:
                if cronjob.metadata.name.startswith(prefix):
                    logger.debug(f"Deleting cronjob: {cronjob.metadata.name}")
                    self.batch_v1_api.delete_namespaced_cron_job(cronjob.metadata.name, namespace)
                    logger.info(f"Deleted cronjob '{cronjob.metadata.name}' in namespace '{namespace}'")

            # Delete jobs starting with the prefix
            jobs = self.batch_v1_api.list_namespaced_job(namespace=namespace)
            for job in jobs.items:
                if job.metadata.name.startswith(prefix):
                    logger.debug(f"Deleting job: {job.metadata.name}")
                    self.batch_v1_api.delete_namespaced_job(job.metadata.name, namespace)
                    logger.info(f"Deleted job '{job.metadata.name}' in namespace '{namespace}'")

            # Add more resource deletion if needed

        except ApiException as e:
            logger.error(f"Failed to delete resources starting with '{prefix}': {e}")

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

    # Homepage_url=f"http://{route['host']}"
    def get_route_urls(self, namespace):
        # Fetching all routes in the namespace
        routes = self.get_routes_info(namespace)
        route_urls = []
        for route in routes:
            if 'home-page' in route['host']:
                route_urls.append(f"home-page: http://{route['host']}\n")
            if 'flask' in route['host']:
                route_urls.append(f"flask serve: http://{route['host']}\n")
        final_route_urls = ''.join(route_urls)
        # return final_route_urls
        return route_urls

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
        import shlex
        from kubernetes.stream import stream
        """
        Execute a command in a pod.

        :param namespace: The namespace of the pod.
        :param pod_name: The name of the pod where the command will be executed.
        :param command: The command to execute inside the pod. This can be a string or a list of strings.
        :return: The output from the command execution.
        """
        # If command is a string, use shlex to split it into a list
        if isinstance(command, str):
            command_list = shlex.split(command)
        else:
            command_list = command

        try:
            # Executing the command
            response = stream(self.core_v1_api.connect_get_namespaced_pod_exec,
                              pod_name,
                              namespace,
                              command=command_list,
                              stderr=True,
                              stdin=False,
                              stdout=True,
                              tty=False)
            return response
        except Exception as e:
            logger.error(f"Failed to execute command in pod {pod_name}: {str(e)}")
            raise

    def get_pod_info(self, pod_name, namespace):
        """Retrieve information about a specific pod."""
        try:
            return self.core_v1_api.read_namespaced_pod(name=pod_name, namespace=namespace or self.namespace)
        except ApiException as e:
            logger.error(f"Failed to get pod info for {pod_name} in {namespace}: {e}")
            return None

    def get_pod_ip(self, pod_name, namespace):
        try:
            pod = self.core_v1_api.read_namespaced_pod(name=pod_name, namespace=namespace)
            return pod.status.pod_ip
        except ApiException as e:
            logger.error(f"Failed to fetch IP for pod {pod_name} in namespace {namespace}: {str(e)}")
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

    @staticmethod
    def send_email(subject, body, to_email, from_email, from_email_password):
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email
        # msg.attach(MIMEText(body, 'plain'))
        msg.attach(MIMEText(body, 'html'))

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.set_debuglevel(1)
            server.starttls()
            server.login(from_email, from_email_password)
            server.sendmail(from_email, to_email, msg.as_string())
            server.quit()
            logger.info("Email sent successfully!")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    # def delete_route(self, route_name, namespace):
    #     from openshift.dynamic.exceptions import NotFoundError
    #     from openshift.dynamic import DynamicClient
    #
    #     dyn_client = DynamicClient(self.api_client)
    #
    #     # Get the Route resource from the OpenShift API
    #     route_resource = dyn_client.resources.get(api_version='route.openshift.io/v1', kind='Route')
    #
    #     try:
    #         # Try to get the existing route
    #         existing_route = route_resource.get(name=route_name, namespace=namespace)
    #         # If the route exists, delete it
    #         existing_route.delete()
    #         logger.info(f"Deleted route '{route_name}' from namespace '{namespace}'.")
    #     except NotFoundError:
    #         # If the route doesn't exist, log a message
    #         logger.info(f"Route '{route_name}' does not exist in namespace '{namespace}'.")
    #     except Exception as e:
    #         # Handle other exceptions
    #         logger.error(f"Error deleting route '{route_name}' in namespace '{namespace}': {e}")
    #
    # def delete_ingress(self, service_name):
    #     from kubernetes.client import NetworkingV1Api, V1DeleteOptions
    #
    #     try:
    #         # Initialize the NetworkingV1Api
    #         networking_v1_api = NetworkingV1Api(self.api_client)
    #
    #         # Use the NetworkingV1Api to delete the Ingress
    #         networking_v1_api.delete_namespaced_ingress(
    #             name=f"{service_name}-ingress",
    #             namespace=self.namespace,
    #             body=V1DeleteOptions()
    #         )
    #         logger.info(f"Ingress for service {service_name} deleted successfully.")
    #     except Exception as e:
    #         logger.error(f"Failed to delete Ingress for service {service_name}: {e}")
    #
    # def delete_service_account(self, app_name, namespace):
    #     try:
    #         self.core_v1_api.delete_namespaced_service_account(app_name, namespace)
    #         logger.info(f"Deleted ServiceAccount '{app_name}' in namespace '{namespace}'.")
    #     except ApiException as e:
    #         logger.error(f"Failed to delete ServiceAccount '{app_name}': {e}")
