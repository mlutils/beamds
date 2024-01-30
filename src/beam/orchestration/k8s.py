from ..path import beam_path
from ..core import Processor
from ..utils import lazy_property
from kubernetes import client, config
from kubernetes.client import Configuration
from kubernetes.client.rest import ApiException
from openshift.dynamic import DynamicClient
from ..logger import beam_logger as logger
import json


class BeamPod(Processor):
    pass


# #!/bin/bash
#
# IMAGE=$1
# NAME=$2
# INITIALS=$3
# INITIALS=$(printf '%03d' $(echo $INITIALS | rev) | rev)
# HOME_DIR=$4
# MORE_ARGS=${@:5}
#
# echo "Running a new container named: $NAME, Based on image: $IMAGE"
# echo "Jupyter port will be available at: ${INITIALS}88"
#
# echo $MORE_ARGS
#
# # Get total system memory in kilobytes (kB)
# total_memory_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
# # Calculate 90% backoff of total memory
# backoff_memory_kb=$(awk -v x=$total_memory_kb 'BEGIN {printf "%.0f", x * 0.9}')
# # Convert to megabytes for Docker
# backoff_memory_mb=$(awk -v x=$backoff_memory_kb 'BEGIN {printf "%.0f", x / 1024}')
#
# # -e USER_HOME=${HOME}
# echo "Home directory: ${HOME_DIR}"
# docker run -p ${INITIALS}00-${INITIALS}99:${INITIALS}00-${INITIALS}99 --cap-add=NET_ADMIN --gpus=all --shm-size=8g --memory=${backoff_memory_mb}m --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v ${HOME_DIR}:${HOME_DIR} -v /mnt/:/mnt/ ${MORE_ARGS} --name ${NAME} --hostname ${NAME} ${IMAGE} ${INITIALS}
# # docker run -p 28000-28099:28000-28099 --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v /home/:/home/ -v /mnt/:/mnt/ --name <name> beam:<date> 28


class BeamDeploy(Processor):

    def __init__(self, k8s=None, project_name=None, namespace=None, replicas=None, labels=None, image_name=None,
                 deployment_name=None, ports=None, service_type=None, *entrypoint_args, **entrypoint_envs):
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
        self.ports = ports
        self.service_type = service_type
        self.service_account_name = f"svc{deployment_name}"

    def launch(self, replicas=None):
        if replicas is None:
            replicas = self.replicas

        self.k8s.create_service_account(self.service_account_name, self.namespace)

        self.k8s.create_service(
            name=self.deployment_name,
            namespace=self.namespace,
            ports=self.ports,
            labels=self.labels,
            service_type=self.service_type
        )

        deployment = self.k8s.create_deployment(image_name=self.image_name, labels=self.labels,
                                                deployment_name=self.deployment_name, namespace=self.namespace,
                                                project_name=self.project_name, replicas=replicas, ports=self.ports,
                                                service_account_name=self.service_account_name, *self.entrypoint_args,
                                                **self.entrypoint_envs)

        pod = self.k8s.apply_deployment(deployment, namespace=self.namespace)
        return BeamPod(pod)


class BeamK8S(Processor):  # processor is a another class and the BeamK8S inherits the method of processor
    """BeamK8S is a class that provides a simple interface to the Kubernetes API."""

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

    @staticmethod
    def create_container(image_name, deployment_name=None, project_name=None, ports=None, *entrypoint_args,
                         **envs):

        container_name = f"{project_name}-{deployment_name}-container"
        if ports is None:
            ports = []
        return client.V1Container(
            name=container_name,
            image=image_name,
            ports=BeamK8S.create_container_ports(ports),
            args=entrypoint_args,
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
                          replicas=None,
                          ports=None, *entrypoint_args, **envs):
        if namespace is None:
            namespace = self.namespace

        if project_name is None:
            project_name = self.project_name

        if deployment_name is None:
            import coolname
            name = coolname.generate_slug(2)
            deployment_name = f"{image_name.split(':')[0]}-{name}"

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

    def create_service(self, name, namespace, ports, labels, service_type='ClusterIP'):
        metadata = client.V1ObjectMeta(name=name, labels=labels)
        # Update here to include port names
        port_specs = [client.V1ServicePort(name=f"port-{p}", port=p, target_port=p) for p in ports]
        spec = client.V1ServiceSpec(ports=port_specs, selector=labels, type=service_type)
        service = client.V1Service(api_version="v1", kind="Service", metadata=metadata, spec=spec)
        try:
            response = self.core_v1_api.create_namespaced_service(namespace=namespace, body=service)
            logger.info(f"Service {name} created in namespace {namespace}. Response: {response}")
        except ApiException as e:
            logger.error(f"Failed to create service {name} in namespace {namespace}: {e}")
            raise

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
