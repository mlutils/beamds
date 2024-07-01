from ..config import BeamConfig, BeamParam
from ..serve import BeamServeConfig


class K8SConfig(BeamConfig):

    parameters = [
        BeamParam('api_url', str, None, 'URL of the Kubernetes API server'),
        BeamParam('api_token', str, None, 'API token for the Kubernetes API server'),
        BeamParam('project_name', str, None, 'Name of the project'),
        BeamParam('deployment_name', str, None, 'Name of the deployment'),
        BeamParam('labels', dict, {}, 'Labels for the deployment'),
        BeamParam('image_name', str, None, 'Name of the image to deploy'),
        BeamParam('command', dict, {}, 'Command configuration for the deployment'),
        BeamParam('os_namespace', str, None, 'Namespace for the deployment'),
        BeamParam('replicas', int, 1, 'Number of replicas for the deployment'),
        BeamParam('entrypoint_args', list, [], 'Arguments for the container entrypoint'),
        BeamParam('entrypoint_envs', dict, {}, 'Environment variables for the container entrypoint'),
        BeamParam('use_scc', bool, True, 'Use SCC control parameter'),
        BeamParam('scc_name', str, 'anyuid', 'SCC name'),
        BeamParam('create_service_account', bool, True, 'Create service account'),
        BeamParam('security_context_config', dict, {}, 'Security context configuration'),
        BeamParam('use_node_selector', bool, False, 'Use node selector'),
        BeamParam('node_selector', dict, {"gpu-type": "tesla-a100"}, 'Node selector'),
        BeamParam('cpu_requests', str, '4', 'CPU requests'),
        BeamParam('cpu_limits', str, '4', 'CPU limits'),
        BeamParam('memory_requests', str, '12', 'Memory requests'),
        BeamParam('memory_limits', str, '12', 'Memory limits'),
        BeamParam('gpu_requests', str, '1', 'GPU requests'),
        BeamParam('gpu_limits', str, '1', 'GPU limits'),
        BeamParam('use_gpu', bool, False, 'Use GPU'),
        BeamParam('n_pods', int, 1, 'Number of pods'),
        BeamParam('enable_ray_ports', bool, False, 'Enable Ray ports'),
        BeamParam('storage_configs', list, [], 'Storage configurations'),
        BeamParam('memory_configs', list, [], 'Memory storage configurations'),
        BeamParam('service_configs', list, [], 'Service configurations'),
        BeamParam('user_idm_configs', list, [], 'User IDM configurations'),
        BeamParam('ray_ports_configs', list, [], 'Ray ports configurations'),
        BeamParam('check_project_exists', bool, True, 'Check if project exists')
    ]


class RayClusterConfig(K8SConfig):
    parameters = [
        BeamParam('n-pods', int, 1, 'Number of Ray worker pods'),
    ]


class HTTPServeClusterConfig(K8SConfig, BeamServeConfig):

    defaults = dict(n_threads=16)

    parameters = [
        BeamParam('alg_image_name', str, 'fake-alg-http-server:latest', 'Algorithm image name'),
        BeamParam('base_image', str, 'eladsar/beam:20240605', 'Base image'),
        BeamParam('base_url', str, 'tcp://10.0.7.55:2375', 'Base URL'),
        BeamParam('beam_version', str, '2.5.11', 'Beam version'),
        BeamParam('send_email', bool, False, 'Send email'),
        BeamParam('body', str, 'Here is the cluster information:', 'Email body'),
        BeamParam('from_email', str, 'dayotech2018@gmail.com', 'From email address'),
        BeamParam('from_email_password', str, 'mkhdokjqwwmazyrf', 'From email password'),
        BeamParam('to_email', str, None, 'To email address'),
        BeamParam('smtp_server', str, 'smtp.gmail.com', 'SMTP server'),
        BeamParam('smtp_port', int, 587, 'SMTP port'),
        BeamParam('registry_project_name', str, 'public1', 'Registry project name'),
        BeamParam('subject', str, 'Cluster Deployment Information', 'Email subject'),
        BeamParam('registry_url', str, 'harbor.dt.local', 'Registry URL'),
        BeamParam('registry_username', str, 'admin', 'Registry username'),
        BeamParam('registry_password', str, 'Har@123', 'Registry password'),
        BeamParam('push_image', bool, True, 'Push image to registry'),
        BeamParam('pods', list, [], 'List of pods'),
        BeamParam('alg', object, None, 'Algorithm object')
    ]


#
# alg_image_name: fake-alg-http-server:latest
# api_token: sha256~ya9nNwLC_tY6nTGY4WgrDP5llXBbtPlOAKngFL2l4J0
# api_url: https://api.kh-dev.dt.local:6443
# base_image: eladsar/beam:20240605
# base_url: tcp://10.0.7.55:2375
# # client = docker.APIClient(base_url='unix://var/run/docker.sock')
# #client = docker.APIClient(base_url='tcp://10.0.7.55:2375')
# # client = docker.APIClient(base_url='unix:///home/beam/.docker/run/docker.sock')
# # client = docker.APIClient(base_url='unix:////home/beam/runtime/docker.sock')
# beam_version: 2.5.11
# body: 'Here is the cluster information:'
# check_project_exists: true
# command:
#   arguments:
#   - -c
#   - sleep infinity
#   executable: /bin/bash
# cpu_limits: '4'
# cpu_requests: '4'
# create_service_account: true
# deployment_name: oneclick
# enable_ray_ports: true
# entrypoint_args:
# - '63'
# entrypoint_envs:
#   TEST: test
# from_email: dayotech2018@gmail.com
# from_email_password: mkhdokjqwwmazyrf
# gpu_limits: '1'
# gpu_requests: '1'
# image_name: harbor.dt.local/public/beam:openshift-180524a
# labels:
#   app: oneclick
# memory_limits: '12'
# memory_requests: '12'
# memory_storage_configs:
# - enabled: true
#   mount_path: /dev/shm
#   name: dshm
#   size_gb: 8
# n_pods: '2'
# node_selector:
#   gpu-type: tesla-a100
# project_name: oneclick
# push_image: true
# ray_ports_configs:
# - ray_ports:
#   - 6379
#   - 8265
# registry_password: Har@123
# registry_url: harbor.dt.local
# registry_username: admin
# replicas: '1'
# scc_name: anyuid
# security_context_config:
#   add_capabilities:
#   - SYS_CHROOT
#   - CAP_AUDIT_CONTROL
#   - CAP_AUDIT_WRITE
#   enable_security_context: false
# service_configs:
# - create_ingress: false
#   create_route: false
#   ingress_host: ssh.example.com
#   port: 2222
#   port_name: ssh-port
#   service_name: ssh
#   service_type: NodePort
# - create_ingress: false
#   create_route: true
#   ingress_host: jupyter.example.com
#   port: 8888
#   port_name: jupyter-port
#   service_name: jupyter
#   service_type: ClusterIP
# - create_ingress: false
#   create_route: true
#   ingress_host: jupyter.example.com
#   port: 44044
#   port_name: fake-alg-http
#   service_name: fake-alg
#   service_type: ClusterIP
# smtp_port: 587
# smtp_server: smtp.gmail.com
# storage_configs:
# - create_pvc: false
#   pvc_access_mode: ReadWriteMany
#   pvc_mount_path: /data-pvc
#   pvc_name: data-pvc
#   pvc_size: '500'
# subject: Cluster Deployment Information
# to_email: yossi@dayo-tech.com
# use_gpu: false
# use_node_selector: false
# use_scc: true
# user_idm_configs:
# - create_role_binding: false
#   project_name: ben-guryon
#   role_binding_name: yos
#   role_name: admin
#   user_name: yos
# - create_role_binding: false
#   project_name: ben-guryon
#   role_binding_name: asafe
#   role_name: admin
#   user_name: asafe



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
