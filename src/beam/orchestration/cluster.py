
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

class RayCluster:

    def launch(self):
        pass

    def add_node(self):
        pass
        # dynamically add nodes after starting the cluster: first add pod and then connect to the cluster (with ray)
