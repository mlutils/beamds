from ..config import BeamConfig, BeamParam



class AirFlowConfig(BeamConfig):
    parameters = [
        BeamParam('airflow_url', str, None, 'Airflow URL'),
        BeamParam('dag-owner', str, 'beam', 'DAG owner'),
        BeamParam('start-date', str, None, 'Start date for the DAGs'),
        BeamParam('email', str, None, 'Email address for notifications'),
        BeamParam('email-on-failure', bool, False, 'Email on failure'),
        BeamParam('email-on-retry', bool, False, 'Email on retry'),

        BeamParam('time-format', str, "%Y-%m-%dT%H:%M:%S", 'Time format to be used'),
        BeamParam('time-zone', str, "UTC", 'Time zone to be used'),
        BeamParam('docker-image', str, None, 'Docker image to be used in all the tasks (Cron format)'),
        BeamParam('schedule-interval', str, None, 'Schedule interval for the DAGs'),
        BeamParam('dag-timeout', int, 60, 'DAG timeout in minutes'),
        BeamParam('dag-retries', int, 0, 'DAG retries'),
        BeamParam('dag-retry-delay', int, 5, 'DAG retry delay in minutes'),
        BeamParam('dag-concurrency', int, 16, 'DAG concurrency'),
        BeamParam('catchup', bool, False, 'Catchup'),
        BeamParam('depends-on-past', bool, False, 'Whether the DAG execution depends on the past'),
        BeamParam('cpu-request', str, 2, 'CPU request for the tasks'),
        BeamParam('cpu-limit', str, None, 'CPU limit for the tasks (if not set, it will be the same as cpu-request)'),
        BeamParam('memory-request', int, 2, 'Memory request for the tasks in GB (can be a float or str as well)'),
        BeamParam('memory-limit', int, None, 'Memory limit for the tasks in GB (can be a float or str as well)'),
        BeamParam('gpu-request', int, 0, 'GPU request for the tasks'),
        BeamParam('gpu-limit', int, 0, 'GPU limit for the tasks (if not set, it will be the same as gpu-request)'),
        BeamParam('gpu-type', str, None, 'GPU type for the tasks'),
        BeamParam('dag-tags', list, [], 'DAG tags'),

    ]