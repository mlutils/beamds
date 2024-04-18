from ..config import BeamConfig, BeamParam


# api_url = "https://api.kh-dev.dt.local:6443"
# api_token = "sha256~CUfUK_8toDmCmLRBkdwcS3qQXbCjaHdqWK-tZw_mGds"
# project_name = "kh-dev"
# image_name = "harbor.dt.local/public/beam:openshift-20.02.6"
# labels = {"app": "kh"}
# deployment_name = "kh"
# # namespace = "ben-guryon"
# namespace = project_name
# replicas = 1
# entrypoint_args = ["63"]  # Container arguments
# entrypoint_envs = {"TEST": "test"}  # Container environment variables
# use_scc = True  # Pass the SCC control parameter
# scc_name = "anyuid"  # privileged , restricted, anyuid, hostaccess, hostmount-anyuid, hostnetwork, node-exporter-scc
# security_context_config = (
#     SecurityContextConfig(add_capabilities=["SYS_CHROOT", "CAP_AUDIT_CONTROL",
#                                             "CAP_AUDIT_WRITE"], enable_security_context=False))
# # node_selector = {"gpu-type": "tesla-a100"} # Node selector in case of GPU scheduling
# node_selector = None
# cpu_requests = "4"  # 0.5 CPU
# cpu_limits = "4"       # 1 CPU
# memory_requests = "12"
# memory_limits = "12"
# gpu_requests = "1"
# gpu_limits = "1"


class K8SConfig(BeamConfig):

    parameters = [
        BeamParam('api_url', str, None, 'URL of the Kubernetes API server'),
        BeamParam('api_token', str, None, 'API token for the Kubernetes API server'),
        BeamParam('project_name', str, None, 'Name of the project'),
        BeamParam('os_namespace', str, None, 'Namespace for the deployment'),
        BeamParam('replicas', int, 1, 'Number of replicas for the deployment'),
        BeamParam('entrypoint_args', list, [], 'Arguments for the container entrypoint'),
        BeamParam('entrypoint_envs', dict, {}, 'Environment variables for the container entrypoint'),
        BeamParam('use_scc', bool, True, 'Use SCC control parameter'),
        BeamParam('scc_name', str, 'anyuid', 'SCC name'),
        BeamParam('security_context_config', dict, {}, 'Security context configuration'),
        BeamParam('node_selector', dict, None, 'Node selector for GPU scheduling'),
        BeamParam('cpu_requests', str, '4', 'CPU requests'),
        BeamParam('cpu_limits', str, '4', 'CPU limits'),
        BeamParam('memory_requests', str, '12', 'Memory requests'),
        BeamParam('memory_limits', str, '12', 'Memory limits'),
        BeamParam('gpu_requests', str, '1', 'GPU requests'),
        BeamParam('gpu_limits', str, '1', 'GPU limits'),
        BeamParam('storage_configs', list, [], 'Storage configurations'),
        BeamParam('memory_storage_configs', list, [], 'Memory storage configurations'),
        BeamParam('service_configs', list, [], 'Service configurations'),
        BeamParam('ray_ports_configs', list, [], 'Ray ports configurations'),
        BeamParam('user_idm_configs', list, [], 'User IDM configurations'),

    ]


class RayClusterConfig(K8SConfig):

    parameters = [
        BeamParam('n-workers', int, 1, 'Number of Ray workers'),
    ]