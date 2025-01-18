from airflow_client.client import ApiClient, Configuration
from airflow_client.client.api.dag_run_api import DAGRunApi
from airflow_client.client.api.dag_api import DAGApi
from airflow_client.client.api.config_api import ConfigApi
from airflow_client.client.model.clear_task_instance import ClearTaskInstance
from airflow_client.client.api.task_instance_api import TaskInstanceApi

from ..path import PureBeamPath, normalize_host

class AirflowClient(PureBeamPath):

    def __init__(self, *pathsegments, client=None, hostname=None, port=None, username=None,
                 password=None, tls=False, **kwargs):
        super().__init__(*pathsegments, scheme='airflow', client=client, hostname=hostname, port=port,
                          username=username, password=password, tls=tls, **kwargs)

        if type(tls) is str:
            tls = (tls.lower() == 'true')

        tls = 'https' if tls else 'http'
        url = f'{tls}://{normalize_host(hostname, port)}'

        if client is None:
            client = ApiClient(Configuration(host=url, username=username, password=password))
        self.client = client

        l = len(self.parts[1:])
        self.level = {0: 'root', 1: 'dag', 2: 'dag_run', 3: 'task_instance'}[l]

    @property
    def dag_id(self):
        return self.parts[1]

    @property
    def run_id(self):
        return self.parts[2]

    @property
    def task_id(self):
        return self.parts[3]

    @property
    def dag_api(self):
        return DAGApi(self.client)

    @property
    def dag_run_api(self):
        return DAGRunApi(self.client)

    @property
    def task_instance_api(self):
        return TaskInstanceApi(self.client)

    @property
    def config_api(self):
        return ConfigApi(self.client)

    def iterdir(self):
        if self.level == 'root':
            # iter over all dags
            dags = self.dag_api.get_dags()
            for dag in dags:
                yield self.joinpath(dag.dag_id)
        elif self.level == 'dag':
            # iter over all dag_runs
            dag_runs = self.dag_run_api.get_dag_runs(self.dag_id)
            for dag_run in dag_runs:
                yield self.joinpath(dag_run.run_id)
        elif self.level == 'dag_run':
            # iter over all task_instances
            task_instances = self.task_instance_api.get_task_instances(self.dag_id, self.run_id)
            for task_instance in task_instances:
                yield self.joinpath(task_instance.task_id)
        else:
            raise ValueError(f'Cannot list directory for task_instance level')


    def stat(self):
        if self.level == 'root':
            # get all dags and their status
            info = self.config_api.get_config()
            return info
        elif self.level == 'dag':
            # get current dag info
            info = self.dag_api.get_dag(self.dag_id)
            return info
        elif self.level == 'dag_run':
            # get current dag_run info
            info = self.dag_run_api.get_dag_run(self.dag_id, self.run_id)
            return info
        elif self.level == 'task_instance':
            # get current task_instance info
            info = self.task_instance_api.get_task_instance(self.dag_id, self.run_id, self.task_id)
            return info

    def exists(self):
        return self.stat() is not None

    def unlink(self):
        if self.level == 'root':
            raise ValueError('Cannot delete dags')
        elif self.level == 'dag':
            # delete dag
            self.dag_api.delete_dag(self.dag_id)
        elif self.level == 'dag_run':
            # delete dag_run
            self.dag_run_api.delete_dag_run(self.dag_id, self.run_id)
        else:
            raise ValueError('Cannot delete task_instance')

    def clear(self, upstream=False, downstream=False, future=False, past=False, dry_run=False):

        # clear task_instance
        clear = ClearTaskInstance(upstream=upstream, downstream=downstream, future=future,
                                  past=past, dry_run=dry_run)

        if self.level == 'task_instance':
            self.task_instance_api.clear_task_instance(self.dag_id, self.run_id, self.task_id, clear)


    # set airflow environment variable
    def set_var(self, key, value):
        self.config_api.set_airflow_config(key, value)

    # get airflow environment variable
    def get_var(self, key):
        return self.config_api.get_airflow_config(key)
