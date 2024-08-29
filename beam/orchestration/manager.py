from beam.base import BeamBase
from beam.orchestration import BeamK8S
from threading import Thread
import os
from beam.resources import resource
from beam.orchestration import BeamManagerConfig, RayClusterConfig, ServeClusterConfig, RnDClusterConfig
import time
import atexit
import threading


# BeamManager class
class BeamManager(BeamBase):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, hparams=config, **kwargs)
        self.k8s = BeamK8S(config)
        self.clusters = {}
        self._stop_monitoring = threading.Event()
        # add monitor thread
        self.monitor_thread = Thread(target=self._monitor)
        self.monitor_thread.start()
        atexit.register(self._cleanup)

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
            self.clusters[cluster].cleanup()
        self.k8s.cleanup()
        # kill monitor thread
        self.monitor_thread.join()

    def info(self):
        return {cluster: self.clusters[cluster].info() for cluster in self.clusters}

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

    def launch_serve_cluster(self, config, **kwargs):
        if isinstance(config, str):
            # Resolve the configuration path relative to the script's directory
            script_dir = os.path.dirname(os.path.realpath(__file__))
            conf_path = os.path.join(script_dir, config)

            # Convert the path to a BeamManagerConfig object
            config = ServeClusterConfig(resource(conf_path).str)

        name = self.get_cluster_name(config)
        from .cluster import ServeCluster
        serve_cluster = ServeCluster(config=config, pods=[], k8s=self.k8s, deployment=name, **kwargs)

        self.clusters[name] = serve_cluster.deploy_from_image(config=config, image_name=config['image_name'])

        # Start monitoring the cluster
        monitor_thread = Thread(target=serve_cluster.monitor_cluster)
        monitor_thread.start()

    def launch_rnd_cluster(self, config, **kwargs):
        if isinstance(config, str):
            # Resolve the configuration path relative to the script's directory
            script_dir = os.path.dirname(os.path.realpath(__file__))
            conf_path = os.path.join(script_dir, config)

            # Convert the path to a BeamManagerConfig object
            config = RnDClusterConfig(resource(conf_path).str)

        name = self.get_cluster_name(config)
        from .cluster import RnDCluster

        rnd_cluster = RnDCluster(config=config, replicas=config['replicas'], k8s=self.k8s, deployment=name, **kwargs)
        self.clusters[name] = rnd_cluster.deploy_rnd_cluster_deployment(config=config, replicas=config['replicas'])

        # Start monitoring the cluster
        monitor_thread = Thread(target=rnd_cluster.monitor_cluster)
        monitor_thread.start()

    def get_cluster_name(self, config):
        # TODO: implement a method to generate a unique cluster name (or get it from the config)
        # return random name for now, (docker style)
        # import randomname
        import namegenerator
        return namegenerator.gen()


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

