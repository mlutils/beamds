from ..config import BeamConfig, BeamParam
from ..serve import BeamServeConfig


class K8SConfig(BeamConfig):

    parameters = [
        BeamParam('api_url', str, 'https://api.kh-dev.dt.local:6443', 'URL of the Kubernetes API server'),
        BeamParam('api_token', str, 'eyJhbGciOiJSUzI1NiIsImtpZCI6Imhtdk5nbTRoenVRenhkd0lWdnBWMUI0MmV2ZGpxMk8wQ0NaMlhmejZBc1UifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJkZXYiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlY3JldC5uYW1lIjoieW9zLXRva2VuLWQycDUyIiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQubmFtZSI6InlvcyIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50LnVpZCI6ImNlOWUzNzkyLThmZTAtNDgxNC05YTVlLWNlMTdmODJjOGU5MiIsInN1YiI6InN5c3RlbTpzZXJ2aWNlYWNjb3VudDpkZXY6eW9zIn0.mKgXusdiiEVN3MzjQ6mZOTfjoY8LFz-1RxVCrDcq38V5AcxaEiTvOGm-6-Vd4ZTV15DR7ds2OqBqZcpcdeuD_eSqZofsfF_dFM8483mXsA8obzBjXiOw0sLeUAq7ZCzb0sTVOySfz4v84MGHgCbMOfD92sfVsfbXhvAXYY2HLX2Vh5og6spjz0P__BBpL--8rfaR1bpua8bMhR5gOreuednJ8hTFPsxTtgZkNppBdHC6WO0j6rm5APDLhu0CMj1_Dwdee4KL0xtt5vKK1YDqy2fdq4ApFP5kYIZu0YnIsliI-msGgX1ioT_eqj_7oz6Hdi5gdSiNDVGnXbhwkdYchslB4evLCEGXAEI2uFQ0d2wVkCcFjGqiVjHdpQa6JCxWClXBveap8o78eM_c59WV343YQri2pfiGthAZUYxIz5mXddV9237OHUh6YwUFyosaKv853c_W-py8rCsxUVFA_o7PFkfHnVogPETjJw-ZzVTxk_PYzxGl9Dh8kEVhJCiPrFBlNtoJVnaEcdNKD_z8I2hr3ca6DB6k6Ws-ABIYWOKO3yu07wp6RdTYeoS3wjWB9GkcjW52UHBi1hQ2qrR1m-X0DsdTrg_PTuw-9KgXz5LnekPJwMrzRn2DFaswOmXOynTEM_PbvlsQ55DBntix_r2df2rWnCWgxbw9MuFog44', 'API token for the Kubernetes API server'),
        BeamParam('beam_version', str, 'latest', 'Beam version (can be overridden with bea_ds_path for local installation'),
        # BeamParam('beam_ds_path', str, '/app/docker-tools/beam-ds.whl', 'Beam DS local path to compiled package'),
        BeamParam('beam_ds_path', str, None, 'Beam DS local path to compiled package'),
        BeamParam('project_name', str, 'dev', 'Name of the project'),
        BeamParam('deployment_name', str, 'demo', 'Name of the deployment'),
        BeamParam('debug_sleep', bool, False, 'Sleep Infinity deployment'),
        BeamParam('labels', dict, {"runai/node-pool": "cpu-only"}, 'Labels for the deployment'),
        BeamParam('image_name', str, 'harbor.dt.local/public/beam:20240801', 'Name of the image to deploy'),
        BeamParam('command', dict, {}, 'Command configuration for the deployment'),
        BeamParam('os_namespace', str, 'dev', 'Namespace for the deployment'),
        BeamParam('replicas', int, 1, 'Number of replicas for the deployment'),
        BeamParam('entrypoint_args', list, [], 'Arguments for the container entrypoint'),
        BeamParam('entrypoint_envs', dict, {}, 'Environment variables for the container entrypoint'),
        BeamParam('use_scc', bool, True, 'Use SCC control parameter'),
        BeamParam('scc_name', str, 'anyuid', 'SCC name'),
        BeamParam('create_service_account', bool, True, 'Create service account'),
        BeamParam('security_context_config', dict, {"add_capabilities": ["SYS_CHROOT", "CAP_AUDIT_CONTROL", "CAP_AUDIT_WRITE"],  "enable_security_context": False, "privileged": False}, 'Security context configuration'),
        BeamParam('use_node_selector', bool, False, 'Use node selector'),
        BeamParam('node_selector', dict, {"gpu-type": "tesla-a100"}, 'Node selector'),
        BeamParam('cpu_requests', str, '4', 'CPU requests'),
        BeamParam('cpu_limits', str, '4', 'CPU limits'),
        BeamParam('memory_requests', str, '0.4', 'Memory requests [e.g.: 0.5Gi, 1.5, 400m]'),
        BeamParam('memory_limits', str, '0.4', 'Memory limits [e.g.: 0.5Gi, 1.5, 400m]'),
        BeamParam('gpu_requests', str, '1', 'GPU requests'),
        BeamParam('gpu_limits', str, '1', 'GPU limits'),
        BeamParam('use_gpu', bool, False, 'Use GPU'),
        BeamParam('n_pods', int, 1, 'Number of pods'),
        BeamParam('storage_configs', list, [{"create_pvc": False, "pvc_access_mode": "ReadWriteMany", "pvc_mount_path": "/data-pvc", "pvc_name": "data-pvc", "pvc_size": "500"}], 'Storage configuration for the deployment'),
        BeamParam('service_configs', list, [
            {"create_ingress": False, "create_route": True, "ingress_host": "home-page.example.com", "port": 35000,
             "port_name": "flask-port", "service_name": "flask", "service_type": "ClusterIP",
             "annotations": {"haproxy.router.openshift.io/timeout": "599", "aproxy.router.openshift.io/balance:": "roundrobin"}}],
                  'Service configuration for the deployment'),
        # BeamParam('service_configs', list, [{"create_ingress": False, "create_route": True, "ingress_host": "home-page.example.com", "port": 35000, "port_name": "flask-port", "service_name": "flask", "service_type": "ClusterIP"}], 'Service configuration for the deployment'),
        BeamParam('user_idm_configs', list, [{"create_role_binding": False, "project_name": "ben-guryon", "role_binding_name": "yos", "role_name": "admin", "user_name": "yos"}], 'User IDM configurations'),
        BeamParam('route_timeout', int, 599, 'Route timeout'),
        BeamParam('memory_storage_configs', list, [{"enabled": True, "mount_path": "/dev/shm", "name": "dshm", "size_gb": 8}], 'Memory storage configuration for the deployment'),
        BeamParam('restart_policy_configs', dict, {"condition": "Always", "delay": "5s", "active_deadline_seconds": 300, "max_attempts": 3, "window": "120s"}, 'Restart policy configuration for the deployment'),
        BeamParam('check_project_exists', bool, True, 'Check if project exists'),
        BeamParam('entrypoint', str, None, 'Entrypoint for the container'),
        BeamParam('dockerfile', str, None, 'Dockerfile for the container'),
        BeamParam('docker_kwargs', dict, {"version": "1.0.0", "author": "user@example.com"}, 'Auxiliary Docker arguments (for the build process)'),

    ]


class RayClusterConfig(K8SConfig):
    parameters = [
        BeamParam('n-pods', int, 1, 'Number of Ray worker pods'),
    ]


class RnDClusterConfig(K8SConfig):
    parameters = [
        BeamParam('replicas', int, 1, 'Number of replica pods'),
        BeamParam('send_email', bool, False, 'Send email'),
        BeamParam('body', str, 'Here is the cluster information:', 'Email body'),
        BeamParam('from_email', str, 'dayotech2018@gmail.com', 'From email address'),
        BeamParam('from_email_password', str, 'mkhdokjqwwmazyrf', 'From email password'),
        BeamParam('to_email', str, None, 'To email address'),
        BeamParam('send_email', bool, False, 'Send email or not'),
        BeamParam('smtp_server', str, 'smtp.gmail.com', 'SMTP server'),
        BeamParam('smtp_port', int, 587, 'SMTP port'),
        BeamParam('subject', str, 'Cluster Deployment Information', 'Email subject'),

    ]


class ServeClusterConfig(K8SConfig, BeamServeConfig):

    defaults = dict(n_threads=16,alg=None)

    parameters = [
        BeamParam('alg', str, '/tmp/bundle', 'Algorithm object can be - bundle, object, image'),
        BeamParam('alg_image_name', str, 'alg-demo:latest', 'Algorithm image name'),
        BeamParam('deployment_method', str, 'bundle', 'Deploy - bundle, object, image'),
        BeamParam('base_image', str, 'harbor.dt.local/public/beam:20240801', 'Base image'),
        BeamParam('base_url', str, 'tcp://10.0.7.55:2375', 'Base URL'),
        BeamParam('requirements_blacklist', list, ['sklearn'], 'Requirements blacklist'),
        BeamParam('send_email', bool, False, 'Send email'),
        BeamParam('body', str, 'Here is the cluster information:', 'Email body'),
        BeamParam('from_email', str, 'dayotech2018@gmail.com', 'From email address'),
        BeamParam('from_email_password', str, 'mkhdokjqwwmazyrf', 'From email password'),
        BeamParam('to_email', str, 'cluster@demo.com', 'To email address'),
        BeamParam('send_email', bool, False, 'Send email or not'),
        BeamParam('smtp_server', str, 'smtp.gmail.com', 'SMTP server'),
        BeamParam('smtp_port', int, 587, 'SMTP port'),
        BeamParam('subject', str, 'Cluster Deployment Information', 'Email subject'),
        BeamParam('registry_url', str, 'harbor.dt.local', 'Registry URL'),
        BeamParam('registry_username', str, 'admin', 'Registry username'),
        BeamParam('registry_password', str, 'Har@123', 'Registry password'),
        BeamParam('registry_project_name', str, 'public', 'Registry project name'),
        BeamParam('push_image', bool, True, 'Push image to registry'),
        BeamParam('pods', list, [], 'List of pods'), #TODO ??? where it used? the names are generated uniquely
        BeamParam('copy-bundle', bool, False, 'Copy bundle to tmp directory'),
        BeamParam('path_to_state', str, '/tmp', 'Path to bundle'),
    ]


class BeamManagerConfig(K8SConfig):
    parameters = [
        BeamParam('clusters', list, [], 'list of clusters'),
        BeamParam('labels', dict, {"runai/node-pool": "cpu-only"}, 'Labels for the deployment'),
    ]


class CronJobConfig(K8SConfig):

    defaults = dict(cron_job_name='beam-cron-job', active_deadline_seconds=300)

    parameters = [
        BeamParam('cron_job_name', str, 'beam-cron-job', 'Cron job name'),
        BeamParam('job_schedule', str, '*/2 * * * *', 'Cron job schedule'),
        BeamParam('restart_policy_configs', dict, {"condition": "OnFailure", "delay": "5s", "active_deadline_seconds": 300, "max_attempts": 3, "window": "120s"}, 'Restart policy configuration for the deployment'),
    ]


class JobConfig(K8SConfig):

    defaults = dict(job_name='beam-job', active_deadline_seconds=300)

    parameters = [
        BeamParam('job_name', str, 'beam-job', 'Job Name'),
        BeamParam('active_deadline_seconds', int, 86400, 'Active deadline seconds'),
    ]

class StatefulSetConfig(K8SConfig):
    """
    Configuration for StatefulSets
    """
    parameters = [
        BeamParam('statefulset_name', str, 'beam-statefulset', 'StatefulSet name'),
        BeamParam('replicas', int, 1, 'Number of StatefulSet replicas'),
        BeamParam('service_name', str, None, 'Service name associated with the StatefulSet'),
        BeamParam('volume_claims', list, [], 'Volume claims for the StatefulSet'),
        BeamParam('update_strategy', str, 'RollingUpdate', 'Update strategy for the StatefulSet'),
        BeamParam('pod_management_policy', str, 'OrderedReady', 'Pod management policy'),
    ]
