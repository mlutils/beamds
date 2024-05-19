
from src.beam.orchestration import (BeamK8S, BeamPod, BeamDeploy, ServiceConfig, StorageConfig,
                                    RayPortsConfig, UserIdmConfig, MemoryStorageConfig, SecurityContextConfig,
                                    RayClusterConfig)
from src.beam import resource
import time
import os
from ..processor import Processor


class RayCluster(Processor):
    def __init__(self, deployment, config, *args, n_pods=None, **kwargs):
        super().__init__(*args, n_pods=n_pods, **kwargs)
        self.deployment = deployment
        self.pods = []
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

    def deploy_head_node(self):
        head_deployment = BeamDeploy(
            k8s=self.k8s,
            project_name=self.config['project_name'],
            check_project_exists=self.config['check_project_exists'],
            namespace=self.config['project_name'],
            replicas=1,
            labels=self.config['labels'],
            image_name=self.config['image_name'],
            deployment_name=f"{self.config['deployment_name']}-head",
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
            memory_storage_configs=self.memory_storage_configs,
            security_context_config=self.security_context_config,
            entrypoint_args=self.config['entrypoint_args'],
            entrypoint_envs=self.config['entrypoint_envs'],
            user_idm_configs=self.user_idm_configs,
        )
        return head_deployment.launch(replicas=1)

    def deploy_worker_nodes(self, head_pod_ip):
        worker_deployment = BeamDeploy(
            k8s=self.k8s,
            project_name=self.config['project_name'],
            check_project_exists=self.config['check_project_exists'],
            namespace=self.config['project_name'],
            replicas=self.config['replicas'],
            labels=self.config['labels'],
            image_name=self.config['image_name'],
            deployment_name=f"{self.config['deployment_name']}-worker",
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
            memory_storage_configs=self.memory_storage_configs,
            security_context_config=self.security_context_config,
            entrypoint_args=self.config['entrypoint_args'],
            entrypoint_envs=self.config['entrypoint_envs'],
            user_idm_configs=self.user_idm_configs,
        )
        worker_deployment.launch(replicas=self.config['replicas'])

        # Command to join the head node
        command = f"ray start --address={head_pod_ip}:6379"

        # Execute the command on each worker pod
        worker_pod_instances = worker_deployment.launch(replicas=self.config['replicas'])
        if isinstance(worker_pod_instances, list):
            for pod_instance in worker_pod_instances:
                pod_instance.execute(command)
        elif isinstance(worker_pod_instances, BeamPod):
            worker_pod_instances.execute(command)

    def get_head_pod_ip(self, head_deployment):
        head_pod_instance = head_deployment[0] if isinstance(head_deployment, list) else head_deployment
        head_pod_status = head_pod_instance.get_pod_status()
        head_pod_name = head_pod_instance.pod_infos[0].metadata.name

        if head_pod_status[0][1] == "Running":
            return self.k8s.get_pod_ip(head_pod_name, namespace=self.config['project_name'])
        else:
            raise Exception(f"Head pod {head_pod_name} is not running. Current status: {head_pod_status[0][1]}")

    def deploy_cluster(self):
        print("Deploying head node...")
        head_deployment = self.deploy_head_node()
        time.sleep(10)  # Wait for the head node to be ready
        head_pod_ip = self.get_head_pod_ip(head_deployment)
        print(f"Head Pod IP: {head_pod_ip}")
        print("Deploying worker nodes...")
        self.deploy_worker_nodes(head_pod_ip)

    # def run_head(self, pod):
    #     pod.execute("command to run ray head node")
    #
    # def run_worker(self, pod):
    #     pod.execute("command to run ray worker")

    def launch(self):
        self.add_nodes(self.n_pods)
        self.run_head(self.pods[0])
        for pod in self.pods[1:]:
            self.run_worker(pod)

    def monitor(self):
        # Todo: run over all nodes and get info from pod, if pod is dead, relaunch the pod
        for p in self.pods:
            p.execute("command to verify if ray is on")
            # if not on, relaunch the pod

    def add_nodes(self, n=1):
        self.pods.append(self.deployment.launch(replicas=n))
        # dynamically add nodes after starting the cluster: first add pod and then connect to the cluster (with ray)

    def remove_node(self, i):
        pass
        # dynamically remove nodes after starting the cluster: first remove pod and then connect to the cluster (with ray)


