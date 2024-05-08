
from ..core import Processor

# class BeamCluster:
#
#
#     def launch(self):
#         raise NotImplementedError
#
#     @property
#     def status(self):
#         return self.get_cluster_status()
#
#     # def get_cluster_status(self):
#
#
# class DevCluster:
#     pass

class RayCluster(Processor):

    def __init__(self, deployment, *args, n_pods=None, **kwargs):
        super().__init__(*args, n_pods=n_pods, **kwargs)
        self.deployment = deployment
        self.pods = []

    def run_head(self, pod):
        pod.execute("command to run ray head node")

    def run_worker(self, pod):
        pod.execute("command to run ray worker")

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


