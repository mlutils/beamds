from airflow_client.client import ApiClient, Configuration
from airflow_client.client.api.dag_run_api import DAGRunApi
from airflow_client.client.api.dag_api import DAGApi
from airflow_client.client.api.config_api import ConfigApi
from airflow_client.client.model.clear_task_instance import ClearTaskInstance
from airflow_client.client.api.task_instance_api import TaskInstanceApi
from airflow_client.client.model.dag_run import DAGRun
from datetime import datetime

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
        return self.parts[1] if len(self.parts) > 1 else None

    @property
    def run_id(self):
        return self.parts[2] if len(self.parts) > 2 else None

    @property
    def task_id(self):
        return self.parts[3] if len(self.parts) > 3 else None

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

    def time_filter(self, l, start=None, end=None, pattern=None):
        # filter list of items based on time assume start and end are datetime
        # objects
        if start is None and end is None:
            return l
        l = [item for item in l if start <= self.execution_date(item) <= end]
        return l

    def execution_date(self, item):
        # Extract execution time from an item if available
        return getattr(item, 'execution_date', None)


    def get_running(self, time_start=None, time_end=None):
        if self.level == 'root':
            # get all running dags
            dags = self.dag_api.get_dags()
            l = [dag.dag_id for dag in dags if dag.is_paused is False]
        elif self.level == 'dag':
            # get all running dag_runs
            dag_runs = self.dag_run_api.get_dag_runs(self.dag_id)
            l = [dag_run.run_id for dag_run in dag_runs if dag_run.state == 'running']
        elif self.level == 'dag_run':
            # get all running task_instances
            task_instances = self.task_instance_api.get_task_instances(self.dag_id, self.run_id)
            l = [task_instance.task_id for task_instance in task_instances if task_instance.state == 'running']
        else:
            raise ValueError(f'Cannot get running for task_instance level')

        return self.time_filter(l, start=time_start, end=time_end)

    def get_failed(self, time_start=None, time_end=None):
        if self.level == 'root':
            # get all failed dags
            dags = self.dag_api.get_dags()
            l = [dag.dag_id for dag in dags if dag.is_paused is False]
        elif self.level == 'dag':
            # get all failed dag_runs
            dag_runs = self.dag_run_api.get_dag_runs(self.dag_id)
            l = [dag_run.run_id for dag_run in dag_runs if dag_run.state == 'failed']
        elif self.level == 'dag_run':
            # get all failed task_instances
            task_instances = self.task_instance_api.get_task_instances(self.dag_id, self.run_id)
            l = [task_instance.task_id for task_instance in task_instances if task_instance.state == 'failed']
        else:
            raise ValueError(f'Cannot get failed for task_instance level')

        return self.time_filter(l, start=time_start, end=time_end)

    def get_success(self, time_start=None, time_end=None):
        if self.level == 'root':
            # get all success dags
            dags = self.dag_api.get_dags()
            l = [dag.dag_id for dag in dags if dag.is_paused is False]
        elif self.level == 'dag':
            # get all success dag_runs
            dag_runs = self.dag_run_api.get_dag_runs(self.dag_id)
            l = [dag_run.run_id for dag_run in dag_runs if dag_run.state == 'success']
        elif self.level == 'dag_run':
            # get all success task_instances
            task_instances = self.task_instance_api.get_task_instances(self.dag_id, self.run_id)
            l = [task_instance.task_id for task_instance in task_instances if task_instance.state == 'success']
        else:
            raise ValueError(f'Cannot get success for task_instance level')

        return self.time_filter(l, start=time_start, end=time_end)

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

    def rerun_failed(self, time_start=None, time_end=None):
        """ Rerun all failed DAGs or task instances in the given time range """
        failed_items = self.get_failed(time_start, time_end)

        if self.level == 'dag_run':
            for task_id in failed_items:
                self.task_instance_api.clear_task_instance(self.dag_id, self.run_id, task_id, ClearTaskInstance())

        elif self.level == 'dag':
            for run_id in failed_items:
                self.dag_run_api.post_dag_run(self.dag_id, DAGRun(run_id=run_id))

        elif self.level == 'root':
            for dag_id in failed_items:
                self.dag_run_api.post_dag_run(dag_id, DAGRun(run_id=f"rerun_{dag_id}_{datetime.utcnow().isoformat()}"))

    def start(self):
        """ Start a DAG Run or Task """
        if self.level == 'dag_run':
            self.dag_run_api.post_dag_run(self.dag_id, DAGRun(run_id=self.run_id))
        elif self.level == 'dag':
            run_id = f"manual_{datetime.utcnow().isoformat()}"
            self.dag_run_api.post_dag_run(self.dag_id, DAGRun(run_id=run_id))
        elif self.level == 'task_instance':
            self.task_instance_api.clear_task_instance(self.dag_id, self.run_id, self.task_id, ClearTaskInstance())

    def stop(self):
        """ Stop a running DAG Run or Task """
        if self.level == 'dag_run':
            self.dag_run_api.update_dag_run_state(self.dag_id, self.run_id, state="failed")
        elif self.level == 'task_instance':
            self.task_instance_api.update_task_instance_state(self.dag_id, self.run_id, self.task_id, state="failed")

    def stop_all_active(self):
        """ Stop all active DAGs, DAG Runs, or Task Instances """
        running_items = self.get_running()
        if self.level == 'dag_run':
            for run_id in running_items:
                self.dag_run_api.update_dag_run_state(self.dag_id, run_id, state="failed")
        elif self.level == 'dag':
            for dag_id in running_items:
                runs = self.dag_run_api.get_dag_runs(dag_id)
                for run in runs:
                    if run.state == "running":
                        self.dag_run_api.update_dag_run_state(dag_id, run.run_id, state="failed")

    def pause_dag(self):
        """ Pause a DAG """
        if self.level == 'dag':
            self.dag_api.update_dag(self.dag_id, {"is_paused": True})

    def unpause_dag(self):
        """ Unpause a DAG """
        if self.level == 'dag':
            self.dag_api.update_dag(self.dag_id, {"is_paused": False})
