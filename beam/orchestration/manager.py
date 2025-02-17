from threading import Thread
import os
import time
import atexit
import threading
import namegenerator

# from examples.orchestration.orchestration_beampod import project_name, deployment_name
from ..base import BeamBase
from ..orchestration import BeamK8S
from ..logging import beam_logger as logger
from ..resources import resource
from ..orchestration import BeamManagerConfig, RayClusterConfig, ServeClusterConfig, RnDClusterConfig

from .resource import deploy_server, deploy_job


# BeamManager class
class BeamManager(BeamBase):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, hparams=config, **kwargs)
        self.k8s = BeamK8S(
            api_url=config['api_url'],
            api_token=config['api_token'],
            project_name=config['project_name'],
            namespace=config['project_name']
        )
        self.clusters = {}
        self.jobs = {}
        self._stop_monitoring = threading.Event()
        # add monitor thread
        # self.monitor_thread = Thread(target=self._monitor)
        # self.monitor_thread.start()
        atexit.register(self._cleanup)

    def cleanup_existing_resources(self, namespace, deployment_name, app_name):
        """
        Cleanup existing Kubernetes/OpenShift resources before deploying a new one.
        """
        logger.info(f"Cleaning up existing resources for deployment '{deployment_name}' in namespace '{namespace}'")

        # Scale deployment to zero if it exists
        if self.k8s.deployment_exists(deployment_name, namespace):
            self.k8s.scale_deployment_to_zero(deployment_name, namespace)
        else:
            logger.info(f"Deployment '{deployment_name}' not found in namespace '{namespace}'. Skipping scaling to 0.")

        self.k8s.delete_resources_starting_with(prefix=deployment_name, namespace=namespace)
        # Delete all resources labeled with the app name
        self.k8s.delete_all_resources_by_app_label(app_name, deployment_name=deployment_name, namespace=namespace)

        # Delete CronJobs, Jobs, ConfigMaps, Services, Routes
        self.k8s.cleanup_cronjobs(namespace, app_name)
        self.k8s.cleanup_jobs(namespace, app_name)
        self.k8s.delete_services_by_deployment(deployment_name, namespace)
        self.k8s.delete_routes_by_deployment_name(deployment_name, namespace)
        self.k8s.delete_configmap_by_deployment(deployment_name, namespace)

        # Delete service accounts associated with the app
        self.k8s.delete_service_account(app_name, namespace)

        logger.info(f"Cleanup completed for deployment '{deployment_name}' in namespace '{namespace}'")

    def _monitor(self):
        try:
            while True:
                for cluster in self.clusters.values():
                    # This could be used to log status or perform other checks
                    print(f"Monitoring {cluster.name}: {cluster.get_cluster_status()}")
                time.sleep(10)  # Adjust the sleep time as necessary
        except KeyboardInterrupt:
            # Handle cleanup if the program is stopped
            for cluster in self.clusters.values():
                cluster.stop_monitoring()

    @staticmethod
    def get_cluster_status():
        # Placeholder method to get the cluster status
        # This could interact with Kubernetes to check the status of pods, services, etc.
        return "healthy"  # Or return different statuses based on actual checks

    def _cleanup(self):
        for cluster in self.clusters:
            if hasattr(self.clusters[cluster], 'cleanup'):
                self.clusters[cluster].cleanup()
            else:
                logger.warning(f"Cluster {cluster} does not have a cleanup method.")

        if hasattr(self.k8s, 'cleanup'):
            self.k8s.cleanup()
        else:
            logger.warning("K8S object does not have a cleanup method.")

        # Kill monitor thread
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join()

    def info(self):
        return {cluster: self.clusters[cluster].info() for cluster in self.clusters}

    def launch_job(self, config, **kwargs):
        # If config is a string (path), resolve it and load the configuration
        if isinstance(config, str):
            # Resolve the configuration path relative to the script's directory
            script_dir = os.path.dirname(os.path.realpath(__file__))
            conf_path = os.path.join(script_dir, config)

            # Convert the path to a BeamManagerConfig object
            config = BeamManagerConfig(resource(conf_path).str)

        name = self.get_cluster_name(config)

        from .jobs import BeamJob
        # Cleanup any existing job before deploying the new one
        self.k8s.cleanup_jobs(namespace=config['project_name'], app_name=config['job_name'])

        # Deploy a new job
        job = BeamJob.deploy(config=config, k8s=self.k8s)
        self.clusters[name] = job

        # Start monitoring the job
        monitor_thread = Thread(target=job.monitor_job)
        monitor_thread.start()

        return name

    def launch_cron_job(self, config, **kwargs):
        # If config is a string (path), resolve it and load the configuration
        if isinstance(config, str):
            # Resolve the configuration path relative to the script's directory
            script_dir = os.path.dirname(os.path.realpath(__file__))
            conf_path = os.path.join(script_dir, config)

            # Convert the path to a BeamManagerConfig object
            config = BeamManagerConfig(resource(conf_path).str)

        project_name = config.get('project_name', self.hparams.project_name)
        app_name = config.get('cron_job_name', self.hparams.cron_job_name)
        name = config.get('cron_job_name', self.hparams.cron_job_name)
        # Cleanup any existing job before deploying the new one
        self.k8s.cleanup_cronjobs(namespace=project_name, app_name=app_name)

        # # Deploy a new cron job
        # from .jobs import BeamCronJob
        # cron_job = BeamCronJob.deploy(config=config, k8s=self.k8s)
        # self.jobs[name] = config['alg']

        self.jobs[name] = deploy_job(config=config)

        # # Start monitoring the cron job
        # monitor_thread = Thread(target=cron_job.monitor_cron_job)
        # monitor_thread.start()

        return name

    def launch_ray_cluster(self, config, **kwargs):
        # If config is a string (path), resolve it and load the configuration
        if isinstance(config, str):
            # Resolve the configuration path relative to the script's directory
            script_dir = os.path.dirname(os.path.realpath(__file__))
            conf_path = os.path.join(script_dir, config)

            # Convert the path to a BeamManagerConfig object
            config = RayClusterConfig(resource(conf_path).str)

        name = self.get_cluster_name(config)

        from .cluster import RayCluster
        ray_cluster = RayCluster(config=config, k8s=self.k8s, n_pods=config['n_pods'], deployment=name, **kwargs)
        self.clusters[name] = ray_cluster.deploy_ray_cluster_s_deployment(config=config, n_pods=config['n_pods'])

        # Start monitoring the cluster
        monitor_thread = Thread(target=ray_cluster.monitor_cluster)
        monitor_thread.start()

        return name

    def launch_serve_cluster(self, config, **kwargs):


        if isinstance(config, str):
            # Resolve the configuration path relative to the script's directory
            script_dir = os.path.dirname(os.path.realpath(__file__))
            conf_path = os.path.join(script_dir, config)

            # Convert the path to a BeamManagerConfig object
            config = ServeClusterConfig(resource(conf_path).str)

        project_name = config.get('project_name', self.hparams.project_name)
        deployment_name = config.get('deployment_name', namegenerator.gen())

        app = None
        if 'labels' in config:
             if 'app' in config['labels']:
                app = config['labels']['app']

        app = app or deployment_name

        self.cleanup_existing_resources(project_name, deployment_name, app)

        name = self.get_cluster_name(config)
        from .cluster import ServeCluster

        # serve_cluster = ServeCluster(config=config, pods=[], k8s=self.k8s, deployment=name, **kwargs)
        # self.clusters[name] = serve_cluster.deploy_from_image(config=config, image_name=config['image_name'])
        self.clusters[name] = deploy_server(obj=config.alg, config=config)

        # # Start monitoring the cluster
        # monitor_thread = Thread(target=serve_cluster.monitor_cluster)
        # monitor_thread.start()

        return name

    def launch_rnd_cluster(self, config,  **kwargs):
        if isinstance(config, str):
            # Resolve the configuration path relative to the script's directory
            script_dir = os.path.dirname(os.path.realpath(__file__))
            conf_path = os.path.join(script_dir, config)

            # Convert the path to a BeamManagerConfig object
            config = RnDClusterConfig(resource(conf_path).str)

        self.cleanup_existing_resources(config['project_name'], config['deployment_name'], config['labels']['app'])

        name = self.get_cluster_name(config)
        from .cluster import RnDCluster

        # rnd_cluster = RnDCluster(config=config, replicas=config['replicas'], k8s=self.k8s, deployment=name, **kwargs)

        self.clusters[name] = RnDCluster.deploy_and_launch(replicas=config['replicas'], config=config,
                                                           k8s=self.k8s)

        if self.clusters[name] and isinstance(self.clusters[name], RnDCluster):
            # Start monitoring the cluster
            monitor_thread = Thread(target=self.clusters[name].monitor_cluster)
            monitor_thread.start()

            cluster_logs = self.clusters[name].get_cluster_logs()
            if cluster_logs:
                for pod_name, logs in cluster_logs.items():
                    logger.info(f"--- Logs from pod {pod_name} ---")
                    for log_entry in logs:
                        pod_name, log_content = log_entry
                        # Split the log content by lines and log each line individually
                        for line in log_content.splitlines():
                            if line.strip():  # Only log non-empty lines
                                logger.info(line.strip())

        else:
            logger.error("Failed to initialize or launch the RnDCluster.")

        return name

    def get_cluster_name(self, config):
        # TODO: implement a method to generate a unique cluster name (or get it from the config)
        # return random name for now, (docker style)
        # import randomname
        import namegenerator
        return namegenerator.gen()

    def get_cluster_service(self, deployment_name, namespace, port):
        """
        Retrieve the URL of a pod associated with a specific deployment and port.
        :param deployment_name: Name of the deployment.
        :param namespace: Namespace of the deployment.
        :param port: Target port to match.
        :return: URL of the pod (http://<pod_name>:<port>) if found, otherwise None.
        """
        try:
            # Get the deployment details
            deployment = self.k8s.apps_v1_api.read_namespaced_deployment(name=deployment_name, namespace=namespace)
            selector = deployment.spec.selector.match_labels

            # Get all pods matching the deployment's selector
            label_selector = ",".join([f"{key}={value}" for key, value in selector.items()])
            pods = self.k8s.core_v1_api.list_namespaced_pod(namespace=namespace, label_selector=label_selector).items

            for pod in pods:
                if pod.status.phase == "Running":
                    # Check if the pod has a container with the specific port
                    for container in pod.spec.containers:
                        if container.ports:
                            for container_port in container.ports:
                                if container_port.container_port == port:
                                    # Construct the URL
                                    # pod_name = pod.metadata.name
                                    # url = f"http://{pod_name}:{port}"
                                    pod_ip = pod.status.pod_ip
                                    url = f"http://{pod_ip}:{port}"
                                    return url

            logger.info(f"No pod with port {port} found for deployment '{deployment_name}' in namespace '{namespace}'.")
            return None

        except self.k8s.client.exceptions.ApiException as e:
            logger.error(f"Error retrieving pod URL for deployment '{deployment_name}': {e}")
            return None

        except self.k8s.client.exceptions.ApiException as e:
            logger.error(f"Error retrieving pod hostname for deployment '{deployment_name}': {e}")
            return None


    def retrieve_cluster_logs(self, cluster_name):
        if cluster_name not in self.clusters:
            logger.error(f"Cluster '{cluster_name}' not found.")
            return

        cluster = self.clusters[cluster_name]
        cluster_logs = cluster.get_cluster_logs()

        for pod_name, logs in cluster_logs.items():
            logger.info(f"--- Logs from pod {pod_name} ---")
            logger.info(logs)


    def scale_up(self, cluster, n):
        # add n pods to the cluster
        self.clusters[cluster].scale_up(n)

    def scale_down(self, cluster, n):
        # remove n pods from the cluster
        self.clusters[cluster].scale_down(n)

    def stop_monitoring(self):
        # Signal the monitoring thread to stop
        self._stop_monitoring.set()
        self.monitor_thread.join()

    def kill_cluster(self, cluster):
        self.clusters[cluster].cleanup()
        del self.clusters[cluster]

    def cluster_info(self, cluster):
        return self.clusters[cluster].info()

