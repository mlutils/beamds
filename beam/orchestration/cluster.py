from typing import List, Union, Dict
import time

from kubernetes.client import ApiException

from ..logging import beam_logger as logger
from ..base import BeamBase
from .k8s import BeamK8S
from .pod import BeamPod
from .deploy import BeamDeploy
from .dataclasses import (ServiceConfig, StorageConfig, RayPortsConfig, UserIdmConfig,
                          MemoryStorageConfig, SecurityContextConfig)


# BeamCluster class now inherits from BeamBase
class BeamCluster(BeamBase):

    def __init__(self, deployment: Union[BeamDeploy, Dict[str, BeamDeploy], None], config, pods: List[BeamPod] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pods = pods
        self.deployment = deployment
        self.config = config
        self.k8s = BeamK8S(
            api_url=config['api_url'],
            api_token=config['api_token'],
            project_name=config['project_name'],
            namespace=config['project_name'],
        )

        self.security_context_config = SecurityContextConfig(**config.get('security_context_config', {}))
        self.memory_storage_configs = [MemoryStorageConfig(**v) for v in config.get('memory_storage_configs', [])]
        self.service_configs = [ServiceConfig(**v) for v in config.get('service_configs', [])]
        self.storage_configs = [StorageConfig(**v) for v in config.get('storage_configs', [])]
        self.ray_ports_configs = [RayPortsConfig(**v) for v in config.get('ray_ports_configs', [])]
        self.user_idm_configs = [UserIdmConfig(**v) for v in config.get('user_idm_configs', [])]
        self.entrypoint_args = config['entrypoint_args']
        self.entrypoint_envs = config['entrypoint_envs']



class ServeCluster(BeamCluster):

    def __init__(self, deployment, config, pods=None,  *args, **kwargs):
        super().__init__(deployment, config, *args,  pods=pods,  **kwargs)

        self.subject = config['subject']
        self.body = config['body']
        self.to_email = config['to_email']
        self.from_email = config['from_email']
        self.from_email_password = config['from_email_password']

        if pods is None:
            self.replicas = config['replicas']
        else:
            self.replicas = len(pods)
        self.labels = config.get('labels', None)

        if not self.labels:
            raise ValueError("Labels must be provided in the configuration.")


    @classmethod
    def  _deploy_and_launch(cls, bundle_path=None, obj=None, image_name=None, config=None, k8s=None):

        from ..auto import AutoBeam

        logger.info(f"base_image: {config['base_image']}")
        if config['requirements_blacklist']:
            logger.warning(f"Will not install the blacklisted requirements: {config['requirements_blacklist']}")

        if image_name is None:
            image_name = AutoBeam.to_docker(obj=obj, bundle_path=bundle_path, base_image=config.base_image,
                                            image_name=config.alg_image_name, copy_bundle=config.copy_bundle,
                                            beam_version=config.beam_version, beam_ds_path=config.beam_ds_path,
                                            base_url=config.base_url,
                                            registry_url=config.registry_url, username=config.registry_username,
                                            password=config.registry_password, serve_config=config,
                                            registry_project_name=config.registry_project_name,
                                            entrypoint=config.entrypoint, dockerfile=config.dockerfile,
                                            requirements_blacklist=config.requirements_blacklist, path_to_state=config.path_to_state,
                                            )
            logger.info(f"Image {image_name} created successfully")

        if k8s is None:
            # to deployment
            k8s = BeamK8S(
                api_url=config['api_url'],
                api_token=config['api_token'],
                project_name=config['project_name'],
                namespace=config['project_name'],
            )

        config.set('image_name', image_name)
        deployment = BeamDeploy(config, k8s)

        try:
            pods = deployment.launch(replicas=config['replicas'])

            # Ensure pods is always a list of BeamPod objects
            if isinstance(pods, BeamPod):
                pods = [pods]
            elif isinstance(pods, list):
                for pod in pods:
                    if not isinstance(pod, BeamPod):
                        raise TypeError(f"Expected a BeamPod object, but got {type(pod)}")

            if pods:
                logger.info("Pod deployment successful")

            if config.send_email is True:

                subject = "Cluster Deployment Information"
                # body = f"{config['body']}\n{get_cluster_info}"
                body = f"{config['body']}<br>{deployment.cluster_info}"
                to_email = config['to_email']
                from_email = config['from_email']
                from_email_password = config['from_email_password']
                k8s.send_email(subject, body, to_email, from_email, from_email_password)
            else:

                logger.debug(f"Skipping email - printing Cluster info: {deployment.cluster_info}")
                logger.info(f"Route Urls: {deployment.k8s.get_route_urls(namespace=config['project_name'])}")
            # logger.debug(f"Cluster info: {deployment.cluster_info}")
            logger.info(f"Route Urls: {deployment.k8s.get_route_urls(namespace=config['project_name'])}")

            if not pods:
                logger.error("Pod deployment failed")

                return None  # Or handle the error as needed

            # Create the ServeCluster instance first, without pods
            cluster_instance = cls(deployment=deployment, pods=pods, config=config)

            # Assign pods after instantiation
            cluster_instance.pods = pods

            return cluster_instance

        except Exception as e:
            logger.error(f"Error during deployment: {str(e)}")
            from ..utils import beam_traceback
            logger.debug(beam_traceback())
            raise e

    @classmethod
    def deploy_from_bundle(cls, bundle_path, config, k8s=None):
        return cls._deploy_and_launch(bundle_path=bundle_path, obj=None, config=config, k8s=k8s)

    @classmethod
    def deploy_from_algorithm(cls, alg, config, k8s=None):
        return cls._deploy_and_launch(bundle_path=None, obj=alg, config=config, k8s=k8s)

    @classmethod
    def deploy_from_image(cls, image_name, config, k8s=None):
        return cls._deploy_and_launch(bundle_path=None, obj=None, image_name=image_name, config=config, k8s=k8s)

    def get_cluster_status(self):
        pod_statuses = [pod.get_pod_status() for pod in self.pods]
        for status in pod_statuses:
            if status[0][1] != "Running":
                return f"Pod {status[0][0]} status: {status[0][1]}"
        return "healthy"

    def monitor_cluster(self):

        try:
            while True:
                if not self.labels:
                    raise AttributeError("No labels provided to filter pods. Please ensure labels are set correctly.")

                # Ensure the namespace is a string before using it
                namespace = self.config['project_name']
                if not isinstance(namespace, str):
                    logger.error(f"Expected namespace to be a string, but got {type(namespace).__name__}.")
                    return

                # Retrieve updated pods list based on labels
                updated_pods = self.k8s.get_pods_by_label(self.labels, namespace)

                if not updated_pods:
                    logger.error("No pods were found with the provided labels.")
                else:
                    logger.debug(f"Retrieved {len(updated_pods)} pods with type: {type(updated_pods[0])}")

                self.pods = []
                for pod in updated_pods:
                    try:
                        pod_info = BeamPod.extract_pod_info(pod)
                        logger.debug(f"Extracted pod info: {pod_info}")
                        beam_pod = BeamPod(
                            pod_infos=[pod_info],
                            namespace=namespace,
                            k8s=self.k8s
                        )
                        self.pods.append(beam_pod)
                    except Exception as e:
                        logger.error(f"Error initializing BeamPod for pod {pod.metadata.name}: {str(e)}")

                for pod in self.pods:
                    try:
                        statuses = pod.get_pod_status()
                        for pod_name, status in statuses:
                            if status != "Running":
                                logger.warning(f"Pod {pod_name} is not running. Status: {status}")
                            else:
                                logger.info(f"Pod {pod_name} is running smoothly.")
                    except Exception as e:
                        logger.error(f"Error retrieving status for pod: {str(e)}")
                time.sleep(30)  # Adjust sleep duration as needed
        except Exception as e:
            logger.error(f"Error occurred while monitoring the cluster: {str(e)}")

    def get_cluster_logs(self):
        logger.info("Getting logs from ServeCluster pods...")

        if not isinstance(self.pods, list) or not self.pods:
            logger.error("Pods list is not initialized or is not a list.")
            return None

        cluster_logs = {}
        try:
            for pod in self.pods:
                if isinstance(pod, BeamPod):
                    logs = pod.get_logs()
                    pod_name = pod.pod_infos[0].name if pod.pod_infos else "unknown_pod"
                    cluster_logs[pod_name] = logs
                    logger.info(f"--- Logs from pod {pod_name} ---")
                    for log_entry in logs:
                        _, log_content = log_entry
                        logger.info(f"Logs for {pod_name}:")
                        # Split the log content by lines and log each line individually
                        for line in log_content.splitlines():
                            if line.strip():  # Only log non-empty lines
                                logger.info(line.strip())
                else:
                    logger.error(f"Expected BeamPod object, but got {type(pod)}")
            return cluster_logs
        except Exception as e:
            logger.exception("Failed to retrieve or process cluster logs", exc_info=e)
            return None


class RayCluster(BeamCluster):
    def __init__(self, deployment, n_pods, config, head=None, workers=None,  *args, **kwargs):
        super().__init__(deployment, config, *args, n_pods, **kwargs)
        self.workers =  []
        self.n_pods = config['n_pods']

    @classmethod
    def _deploy_and_launch(cls, n_pods=None, config=None):

        k8s = BeamK8S(
            api_url=config['api_url'],
            api_token=config['api_token'],
            project_name=config['project_name'],
            namespace=config['project_name']
        )

        deployment = BeamDeploy(config, k8s)

        try:
            pod_instances = deployment.launch(replicas=config['n_pods'])
            if not pod_instances:
                raise Exception("Pod deployment failed")

            head = pod_instances[0]
            workers = pod_instances[1:]
            # TODO: kill current ray processes before starting new ones
            head_command = "ray start --head --port=6379 --disable-usage-stats --dashboard-host=0.0.0.0"
            head.execute(head_command)

            # TODO: implement reliable method that get ip from head pod when its ready instead of relying to "sleep"
            time.sleep(10)

            head_pod_ip = cls.get_head_pod_ip(head, k8s, config['project_name'])

            worker_command = "ray start --address={}:6379".format(head_pod_ip)

            for pod_instance in pod_instances[1:]:
                pod_instance.execute(worker_command)

            logger.info(deployment.cluster_info)

            return cls(deployment=deployment, n_pods=n_pods, config=config, head=head, workers=workers)

        except Exception as e:
            logger.error(f"Error during deployment: {str(e)}")
            from ..utils import beam_traceback
            logger.debug(beam_traceback())
            raise e

    @classmethod
    def deploy_ray_cluster_s_deployment(cls, config, n_pods):
        return cls._deploy_and_launch(n_pods=n_pods, config=config)

    @classmethod
    def deploy_ray_cluster_m_deployments(cls, config, n_pods):
        return cls._deploy_and_launch(n_pods=n_pods, config=config)

    def deploy_ray_head(cls, config):
        pass
        # return cls._deploy_and_launch(n_pods=1, config=config)

    @classmethod
    def get_head_pod_ip(cls, head_pod_instance, k8s, project_name):
        head_pod_status = head_pod_instance.get_pod_status()
        head_pod_name = head_pod_instance.pod_infos[0].name

        if head_pod_status[0][1] == "Running":
            pod_info = k8s.get_pod_info(head_pod_name, namespace=project_name)
            if pod_info and pod_info.status:
                return pod_info.status.pod_ip
            else:
                raise Exception(f"Failed to get pod info or pod status for {head_pod_name}")
        else:
            raise Exception(f"Head pod {head_pod_name} is not running. Current status: {head_pod_status[0][1]}")
    # def connect_cluster(self):
    #     # example how to connect to head node
    #     for w in self.workers:
    #         w.execute(f"command to connect to head node with ip: {self.head.ip}")

    # Todo: run over all nodes and get info from pod, if pod is dead, relaunch the pod

    def get_cluster_status(self):
        head_pod_status = self.head.get_pod_status()
        if head_pod_status[0][1] == "Running":
            return "healthy"
        else:
            return f"Head pod {self.head.pod_infos[0].name} status: {head_pod_status[0][1]}"

    # TODO: adjust the monitor_cluster as it works in rnd_cluster
    def monitor_cluster(self):
        while True:
            try:
                if self.head is None:
                    logger.error("Head pod is not initialized.")
                    break  # Exit the loop if head pod is not set
                head_pod_status = self.head.get_pod_status()
                if head_pod_status[0][1] != "Running":
                    logger.info(f"Head pod {self.head.pod_infos[0].name} is not running. Restarting...")
                    self.deploy_cluster()
                time.sleep(3)
            except KeyboardInterrupt:
                break

    @staticmethod
    def stop_monitoring():
        logger.info("Stopped monitoring the Ray cluster.")

    def get_cluster_logs(self):
        logger.info("Getting logs from head and worker nodes...")
        head_logs = self.head.get_logs()  # Retrieve head node logs
        worker_logs = self.workers[0].get_logs()  # Retrieve worker node logs
        try:
            logger.info("Logs from head node:")
            for pod_name, log_entries in head_logs:
                logger.info(f"Logs for {pod_name}:")
                for line in log_entries.split('\n'):
                    if line.strip():
                        logger.info(line.strip())

            logger.info("Logs from worker node:")
            for pod_name, log_entries in worker_logs:
                logger.info(f"Logs for {pod_name}:")
                for line in log_entries.split('\n'):
                    if line.strip():
                        logger.info(line.strip())

        except Exception as e:
            logger.exception("Failed to retrieve or process cluster logs", exception=e)

        return head_logs, worker_logs

    def add_nodes(self, n=1):
        raise NotImplementedError
        # new_pods = self.deployment.launch(replicas=n)
        # for pod_instance in new_pods:
        #     self.workers.append(pod_instance)
        #     worker_command = "ray start --address={}:6379".format(self.get_head_pod_ip(self.head))
        #     pod_instance.execute(worker_command)
        #     pod_suffix = pod_instance.pod_infos[0].name.split('-')[-1]
        #     # Re-use BeamDeploy to create services and routes for new worker nodes
        #     for svc_config in self.service_configs:
        #         service_name = f"{svc_config.service_name}-{svc_config.port}-{pod_suffix}"
        #         self.deployment.k8s.create_service(
        #             base_name=service_name,
        #             namespace=self.config['project_name'],
        #             ports=[svc_config.port],
        #             labels=self.config['labels'],
        #             service_type='ClusterIP'
        #         )
        #
        #         # Create routes and ingress if configured
        #         if svc_config.create_route:
        #             self.deployment.k8s.create_route(
        #                 service_name=service_name,
        #                 namespace=self.config['project_name'],
        #                 protocol=svc_config.route_protocol,
        #                 port=svc_config.port
        #             )
        #         if svc_config.create_ingress:
        #             self.deployment.k8s.create_ingress(
        #                 service_configs=[svc_config],
        #             )

    def remove_node(self, i):
        pass


class RnDCluster(BeamCluster):
    def __init__(self, deployment, config, *args, pods=None, **kwargs):
        super().__init__(deployment, config,  *args, pods=pods,  **kwargs)
        if pods is None:
            self.replicas = config['replicas']
        else:
            self.replicas = len(pods)
        self.labels = config.get('labels', None)

        if not self.labels:
            raise ValueError("Labels must be provided in the configuration.")


    @classmethod
    def _deploy_and_launch(cls, replicas=None, config=None, k8s=None):
        if k8s is None:
            k8s = BeamK8S(
                api_url=config['api_url'],
                api_token=config['api_token'],
                project_name=config['project_name'],
                namespace=config['project_name']
            )

        deployment = BeamDeploy(config, k8s)

        if replicas is None:
            replicas = config['replicas']

        try:
            pods = deployment.launch(replicas=replicas)

            # Ensure pods is always a list of BeamPod objects
            if isinstance(pods, BeamPod):
                pods = [pods]
            elif isinstance(pods, list):
                for pod in pods:
                    if not isinstance(pod, BeamPod):
                        raise TypeError(f"Expected a BeamPod object, but got {type(pod)}")

            if pods:
                logger.info("Pod deployment successful")

            if config.send_email is True:
                subject = "Cluster Deployment Information"
                # body = f"{config['body']}\n{get_cluster_info}"
                body = f"{config['body']}<br>{deployment.cluster_info}"
                to_email = config['to_email']
                from_email = config['from_email']
                from_email_password = config['from_email_password']
                k8s.send_email(subject, body, to_email, from_email, from_email_password)

            else:
                logger.debug(f"Skipping email - printing Cluster info: {deployment.cluster_info}")
            # logger.debug(f"Cluster info: {deployment.cluster_info}")

            if not pods:
                logger.error("Pod deployment failed")
                return None  # Or handle the error as needed

            # Create the RnDCluster instance first, without pods
            cluster_instance = cls(deployment=deployment, config=config, pods=pods)

            return cluster_instance

        except Exception as e:
            logger.error(f"Error during deployment: {str(e)}")
            from ..utils import beam_traceback
            logger.debug(beam_traceback())
            raise e

    @classmethod
    def deploy_and_launch(cls, replicas, config, k8s=None):
        return cls._deploy_and_launch(replicas=replicas, config=config, k8s=k8s)

    def get_cluster_status(self):
        pod_statuses = [pod.get_pod_status() for pod in self.pods]
        for status in pod_statuses:
            if status[0][1] != "Running":
                return f"Pod {status[0][0]} status: {status[0][1]}"
        return "healthy"

    def monitor_cluster(self):
        try:
            while True:
                if not self.labels:
                    raise AttributeError("No labels provided to filter pods. Please ensure labels are set correctly.")

                # Ensure the namespace is a string before using it
                namespace = self.config['project_name']
                if not isinstance(namespace, str):
                    logger.error(f"Expected namespace to be a string, but got {type(namespace).__name__}.")
                    return

                # Retrieve updated pods list based on labels
                updated_pods = self.k8s.get_pods_by_label(self.labels, namespace)

                if not updated_pods:
                    logger.error("No pods were found with the provided labels.")
                else:
                    logger.debug(f"Retrieved {len(updated_pods)} pods with type: {type(updated_pods[0])}")

                self.pods = []
                for pod in updated_pods:
                    try:
                        pod_info = BeamPod.extract_pod_info(pod)
                        logger.debug(f"Extracted pod info: {pod_info}")
                        beam_pod = BeamPod(
                            pod_infos=[pod_info],
                            namespace=namespace,
                            k8s=self.k8s
                        )
                        self.pods.append(beam_pod)
                    except Exception as e:
                        logger.error(f"Error initializing BeamPod for pod {pod.metadata.name}: {str(e)}")

                for pod in self.pods:
                    try:
                        pod_status = pod.get_pod_status()
                        if pod_status != "Running":
                            logger.warning(f"Pod {pod.pod_infos[0].name} is not running. Status: {pod_status}")
                        else:
                            logger.info(f"Pod {pod.pod_infos[0].name} is running smoothly.")
                    except Exception as e:
                        logger.error(f"Error retrieving status for pod {pod.pod_infos[0].name}: {str(e)}")

                time.sleep(30)  # Adjust sleep duration as needed
        except Exception as e:
            logger.error(f"Error occurred while monitoring the cluster: {str(e)}")

    def deploy_cluster(self):
        logger.info("Redeploying the RnDCluster...")
        self.deployment.delete()
        self.deployment = BeamDeploy(self.config, self.k8s)
        self.pods = self.deployment.launch(replicas=self.replicas)

    @staticmethod
    def stop_monitoring():
        logger.info("Stopped monitoring the Ray cluster.")

    def get_cluster_logs(self):
        logger.info("Getting logs from RnDCluster pods...")

        if not isinstance(self.pods, list) or not self.pods:
            logger.error("Pods list is not initialized or is not a list.")
            return None

        cluster_logs = {}
        try:
            for pod in self.pods:
                if isinstance(pod, BeamPod):
                    logs = pod.get_logs()
                    pod_name = pod.pod_infos[0].name if pod.pod_infos else "unknown_pod"
                    cluster_logs[pod_name] = logs
                    logger.info(f"--- Logs from pod {pod_name} ---")
                    for log_entry in logs:
                        _, log_content = log_entry
                        logger.info(f"Logs for {pod_name}:")
                        # Split the log content by lines and log each line individually
                        for line in log_content.splitlines():
                            if line.strip():  # Only log non-empty lines
                                logger.info(line.strip())
                else:
                    logger.error(f"Expected BeamPod object, but got {type(pod)}")
            return cluster_logs
        except Exception as e:
            logger.exception("Failed to retrieve or process cluster logs", exc_info=e)
            return None




