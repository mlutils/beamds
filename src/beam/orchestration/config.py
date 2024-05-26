from ..config import BeamConfig, BeamParam


# {
#     "api_url": "https://api.kh-dev.dt.local:6443",
#     "api_token": "sha256~J2Dc93HHMiCHYUwRqDtL1ng9O9TTYj-AVVF1qbTyrnw",
#     "check_project_exists": true,
#     "project_name": "kh-dev",
#     "create_service_account": true,
#     "image_name": "harbor.dt.local/public/beam:openshift-180524a",
#     "labels": {"app": "kh-dev"},
#     "deployment_name": "kh-dev",
#     "replicas": 1,
#     "entrypoint_args": ["63"],
#     "entrypoint_envs": {"TEST": "test"},
#     "use_scc": true,
#     "scc_name": "anyuid",
#     "use_node_selector": false,
#     "node_selector": {"gpu-type": "tesla-a100"},
#     "cpu_requests": "4",
#     "cpu_limits": "4",
#     "memory_requests": "12",
#     "memory_limits": "12",
#     "use_gpu": false,
#     "gpu_requests": "1",
#     "gpu_limits": "1",
#     "enable_ray_ports": true,
#   "ray_ports_configs": [{"ray_ports": [6379, 8265]}],
#   "user_idm_configs": [{"user_name": "yos", "role_name": "admin", "role_binding_name": "yos",
#                          "create_role_binding": false, "project_name": "ben-guryon"},
#                         {"user_name": "asafe", "role_name": "admin", "role_binding_name": "asafe",
#                          "create_role_binding": false, "project_name": "ben-guryon"}],
#   "security_context_config": {"add_capabilities": ["SYS_CHROOT", "CAP_AUDIT_CONTROL", "CAP_AUDIT_WRITE"], "enable_security_context": false},
#   "storage_configs": [{"pvc_name": "data-pvc", "pvc_mount_path": "/data-pvc",
#                        "pvc_size": "500", "pvc_access_mode": "ReadWriteMany", "create_pvc": true}],
#   "memory_storage_configs": [{"name": "dshm", "mount_path": "/dev/shm", "size_gb": 8, "enabled": true}],
#   "service_configs": [{"port":  2222, "service_name":  "ssh", "service_type": "NodePort",
#                        "port_name": "ssh-port", "create_route": false, "create_ingress": false,
#                        "ingress_host": "ssh.example.com" },
#                      {"port": 8888, "service_name": "jupyter", "service_type": "ClusterIP",
#                        "port_name": "jupyter-port", "create_route": true, "create_ingress": false,
#                        "ingress_host": "jupyter.example.com"},
#                      {"port": 8265, "service_name": "ray-dashboard", "service_type": "ClusterIP", "port_name": "ray-dashboard-port",
#                       "create_route": true, "create_ingress": false, "ingress_host": "jupyter.example.com"},
#                      {"port": 6379, "service_name": "ray-gcs", "service_type": "ClusterIP", "port_name": "ray-gcs-port",
#                         "create_route": false, "create_ingress": false, "ingress_host": "jupyter.example.com"}]
# }


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
        BeamParam('memory_configs', list, [], 'Memory storage configurations'),
        BeamParam('service_configs', list, [], 'Service configurations'),
        BeamParam('user_idm_configs', list, [], 'User IDM configurations'),
        BeamParam('ray_ports_configs', list, [], 'Ray ports configurations'),
        BeamParam('check_project_exists', bool, True, 'Check if project exists'),

    ]


class RayClusterConfig(K8SConfig):
    parameters = [
        BeamParam('n-pods', int, 1, 'Number of Ray worker pods'),
    ]
