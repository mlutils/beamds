from .k8s import BeamK8S
from .deploy import BeamDeploy
from .dataclasses import (ServiceConfig, StorageConfig, RayPortsConfig, UserIdmConfig,
                          MemoryStorageConfig, SecurityContextConfig)
from ..logging import beam_logger as logger
import time
from ..processor import Processor


class HTTPServeCluster(Processor):

    def __init__(self, alg, deployment, pods, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alg = alg
        self.pods = pods
        self.deployment = deployment
        self.config = config
        self.k8s = BeamK8S(
            api_url=config['api_url'],
            api_token=config['api_token'],
            project_name=config['project_name'],
            namespace=config['project_name'],
        )
        self.subject = config['subject']
        self.body = config['body']
        self.to_email = config['to_email']
        self.from_email = config['from_email']
        self.from_email_password = config['from_email_password']
        self.security_context_config = SecurityContextConfig(**config.get('security_context_config', {}))
        self.memory_storage_configs = [MemoryStorageConfig(**v) for v in config.get('memory_storage_configs', [])]
        self.service_configs = [ServiceConfig(**v) for v in config.get('service_configs', [])]
        self.storage_configs = [StorageConfig(**v) for v in config.get('storage_configs', [])]
        self.ray_ports_configs = [RayPortsConfig(**v) for v in config.get('ray_ports_configs', [])]
        self.user_idm_configs = [UserIdmConfig(**v) for v in config.get('user_idm_configs', [])]
        self.entrypoint_args = config['entrypoint_args']
        self.entrypoint_envs = config['entrypoint_envs']

    @classmethod
    def deploy_from_algorithm(cls, alg, config):

        from ..auto import AutoBeam

        logger.info(f"base_image: {config['base_image']}")

        full_image_name = (
            AutoBeam.to_docker(obj=alg, base_image=config.base_image, image_name=config.alg_image_name,
                               beam_version=config.beam_version, base_url=config.base_url,
                               push_image=config.push_image, registry_url=config.registry_url,
                               username=config.registry_username, password=config.registry_password,
                               serve_config=config, registry_project_name=config.registry_project_name))

        logger.info(f"Image {full_image_name} created successfully")

        # to deployment
        k8s = BeamK8S(
            api_url=config['api_url'],
            api_token=config['api_token'],
            project_name=config['project_name'],
            namespace=config['project_name'],
        )

        security_context_config = SecurityContextConfig(**config.get('security_context_config', {}))
        memory_storage_configs = [MemoryStorageConfig(**v) for v in config.get('memory_storage_configs', [])]
        service_configs = [ServiceConfig(**v) for v in config.get('service_configs', [])]
        # service_configs = [ServiceConfig(service_name='fake-alg', port=serve_config.port, port_name='http',
        #                                  service_type='NodePort', create_route=True, create_ingress=False)]
        storage_configs = [StorageConfig(**v) for v in config.get('storage_configs', [])]

        deployment = BeamDeploy(
            k8s=k8s,
            project_name=config['project_name'],
            check_project_exists=config['check_project_exists'],
            namespace=config['project_name'],
            replicas=config['replicas'],
            labels=config['labels'],
            image_name=full_image_name,
            deployment_name=config['deployment_name'],
            create_service_account=config['create_service_account'],
            use_scc=config['use_scc'],
            use_node_selector=config['use_node_selector'],
            node_selector=config['node_selector'],
            scc_name=config['scc_name'],
            cpu_requests=config['cpu_requests'],
            cpu_limits=config['cpu_limits'],
            memory_requests=config['memory_requests'],
            memory_limits=config['memory_limits'],
            use_gpu=config['use_gpu'],
            gpu_requests=config['gpu_requests'],
            gpu_limits=config['gpu_limits'],
            service_configs=service_configs,
            storage_configs=storage_configs,
            n_pods=config['n_pods'],
            memory_storage_configs=memory_storage_configs,
            security_context_config=security_context_config,
            entrypoint_args=config['entrypoint_args'],
            entrypoint_envs=config['entrypoint_envs'],
            user_idm_configs=[],
            enable_ray_ports=False
        )

        try:
            pods = deployment.launch(replicas=config['replicas'])
            if pods:
                logger.info("Pod deployment successful")
            get_cluster_info = deployment.cluster_info  # Assume this method returns the formatted cluster info string
            if config.send_email is True:
                subject = "Cluster Deployment Information"
                # body = f"{config['body']}\n{get_cluster_info}"
                body = f"{config['body']}<br>{deployment.cluster_info}"
                to_email = config['to_email']
                from_email = config['from_email']
                from_email_password = config['from_email_password']
                k8s.send_email(subject, body, to_email, from_email, from_email_password)
            else:
                logger.info(f"Skipping email - printing Cluster info: {get_cluster_info}")
            # get_cluster_info = deployment.cluster_info
            # logger.info(f"Cluster info: {get_cluster_info}")
            # logger.info(f"Cluster info: {deployment.cluster_info}")
            if not pods:
                logger.error("Pod deployment failed")
                return None  # Or handle the error as needed
            return cls(alg=alg, get_cluster_info=get_cluster_info, deployment=deployment,
                       pods=pods, config=config)
        except Exception as e:
            logger.error(f"Error during deployment: {str(e)}")
            return None


class RayCluster(Processor):
    def __init__(self, deployment, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deployment = deployment
        self.workers = []
        self.n_pods = config['n_pods']
        self.head = None
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

        self.deployment = BeamDeploy(
            k8s=self.k8s,
            project_name=self.config['project_name'],
            check_project_exists=self.config['check_project_exists'],
            namespace=self.config['project_name'],
            replicas=self.config['replicas'],
            labels=self.config['labels'],
            image_name=self.config['image_name'],
            deployment_name=self.config['deployment_name'],
            create_service_account=self.config['create_service_account'],
            use_scc=self.config['use_scc'],
            use_node_selector=self.config['use_node_selector'],
            node_selector=self.config['node_selector'],
            scc_name=self.config['scc_name'],
            cpu_requests=self.config['cpu_requests'],
            cpu_limits=self.config['cpu_limits'],
            memory_requests=self.config['memory_requests'],
            memory_limits=self.config['memory_limits'],
            use_gpu=self.config['use_gpu'],
            gpu_requests=self.config['gpu_requests'],
            gpu_limits=self.config['gpu_limits'],
            service_configs=self.service_configs,
            storage_configs=self.storage_configs,
            ray_ports_configs=self.ray_ports_configs,
            n_pods=self.config['n_pods'],
            memory_storage_configs=self.memory_storage_configs,
            security_context_config=self.security_context_config,
            entrypoint_args=self.config['entrypoint_args'],
            entrypoint_envs=self.config['entrypoint_envs'],
            user_idm_configs=self.user_idm_configs,
            enable_ray_ports=True
        )

    def deploy_cluster(self):
        pod_instances = self.deployment.launch(replicas=self.n_pods)
        if not pod_instances:
            raise Exception("Pod deployment failed")

        self.head = pod_instances[0]
        self.workers = pod_instances[1:]
        head_command = "ray start --head --port=6379 --disable-usage-stats --dashboard-host=0.0.0.0"
        self.head.execute(head_command)

        # TODO: implement reliable method that get ip from head pod when its ready instead of relying to "sleep"
        time.sleep(10)
        head_pod_ip = self.get_head_pod_ip(self.head)

        worker_command = "ray start --address={}:6379".format(head_pod_ip)

        for pod_instance in pod_instances[1:]:
            pod_instance.execute(worker_command)

    def get_head_pod_ip(self, head_pod_instance):
        head_pod_status = head_pod_instance.get_pod_status()
        head_pod_name = head_pod_instance.pod_infos[0].name

        if head_pod_status[0][1] == "Running":
            pod_info = self.k8s.get_pod_info(head_pod_name, namespace=self.config['project_name'])
            if pod_info and pod_info.status:
                return pod_info.status.pod_ip
            else:
                raise Exception(f"Failed to get pod info or pod status for {head_pod_name}")
        else:
            raise Exception(f"Head pod {head_pod_name} is not running. Current status: {head_pod_status[0][1]}")

    #  TODO: implement connect_cluster live in pycharm for now
    # def connect_cluster(self):
    #     # example how to connect to head node
    #     for w in self.workers:
    #         w.execute(f"command to connect to head node with ip: {self.head.ip}")

    # Todo: run over all nodes and get info from pod, if pod is dead, relaunch the pod

    def monitor_cluster(self):
        while True:
            try:
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
        new_pods = self.deployment.launch(replicas=n)
        for pod_instance in new_pods:
            self.workers.append(pod_instance)
            worker_command = "ray start --address={}:6379".format(self.get_head_pod_ip(self.head))
            pod_instance.execute(worker_command)
            pod_suffix = pod_instance.pod_infos[0].name.split('-')[-1]
            # Re-use BeamDeploy to create services and routes for new worker nodes
            for svc_config in self.service_configs:
                service_name = f"{svc_config.service_name}-{svc_config.port}-{pod_suffix}"
                self.deployment.k8s.create_service(
                    base_name=service_name,
                    namespace=self.config['project_name'],
                    ports=[svc_config.port],
                    labels=self.config['labels'],
                    service_type='ClusterIP'
                )

                # Create routes and ingress if configured
                if svc_config.create_route:
                    self.deployment.k8s.create_route(
                        service_name=service_name,
                        namespace=self.config['project_name'],
                        protocol=svc_config.route_protocol,
                        port=svc_config.port
                    )
                if svc_config.create_ingress:
                    self.deployment.k8s.create_ingress(
                        service_configs=[svc_config],
                    )

    def remove_node(self, i):
        pass
