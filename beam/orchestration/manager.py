from ..base import BeamBase
from .k8s import BeamK8S
from threading import Thread
import time
import atexit

# BeamManager class
class BeamManager(BeamBase):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, hparams=config, **kwargs)
        self.k8s = BeamK8S(config)
        self.clusters = {}
        # add monitor thread
        self.monitor_thread = Thread(target=self._monitor)
        self.monitor_thread.start()
        atexit.register(self._cleanup)

    def _monitor(self):
        while True:
            for cluster in self.clusters:
                self.clusters[cluster].monitor()
            time.sleep(1)

    def _cleanup(self):
        for cluster in self.clusters:
            self.clusters[cluster].cleanup()
        self.k8s.cleanup()
        # kill monitor thread
        self.monitor_thread.join()

    def info(self):
        return {cluster: self.clusters[cluster].info() for cluster in self.clusters}

    def launch_ray_cluster(self, config, **kwargs):
        name = self.get_cluster_name(config)
        from .cluster import RayCluster
        self.clusters[name] = RayCluster(config, self.k8s, name=name, **kwargs)

    def launch_serve_cluster(self, config, **kwargs):
        name = self.get_cluster_name(config)
        from .cluster import ServeCluster
        self.clusters[name] = ServeCluster(config, self.k8s, name=name, **kwargs)

    def launch_rnd_cluster(self, config, **kwargs):
        name = self.get_cluster_name(config)
        from .cluster import RnDCluster
        self.clusters[name] = RnDCluster(config, self.k8s, name=name, **kwargs)

    def get_cluster_name(self, config):
        # TODO: implement a method to generate a unique cluster name (or get it from the config)
        # return random name for now, (docker style)
        import randomname
        return randomname.generate_name()

    def scale_up(self, cluster, n):
        # add n pods to the cluster
        self.clusters[cluster].scale_up(n)

    def scale_down(self, cluster, n):
        # remove n pods from the cluster
        self.clusters[cluster].scale_down(n)

    def kill_cluster(self, cluster):
        self.clusters[cluster].cleanup()
        del self.clusters[cluster]

    def cluster_info(self, cluster):
        return self.clusters[cluster].info()

