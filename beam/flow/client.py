import json

from airflow_client.client import ApiClient, Configuration
from airflow_client.client.api.dag_run_api import DAGRunApi
from airflow_client.client.api.dag_api import DAGApi
from airflow_client.client.api.config_api import ConfigApi
from airflow_client.client.model.clear_task_instances import ClearTaskInstances
from airflow_client.client.api.task_instance_api import TaskInstanceApi
from airflow_client.client.model.dag_run import DAGRun
from datetime import datetime
from typing import Optional, Union, List

from ..path import PureBeamPath, normalize_host


class AirflowQuery:
    def __init__(
            self,
            state: Optional[Union[str, List[str]]] = None,  # 'success', 'failed', 'running', 'queued', etc.
            execution_date_gte: Optional[datetime] = None,  # filter execution dates greater or equal to this date
            execution_date_lte: Optional[datetime] = None,  # filter execution dates less or equal to this date
            start_date_gte: Optional[datetime] = None,  # filter DAG runs starting after or at this date
            start_date_lte: Optional[datetime] = None,  # filter DAG runs starting before or at this date
            end_date_gte: Optional[datetime] = None,  # filter DAG runs ending after or at this date
            end_date_lte: Optional[datetime] = None,  # filter DAG runs ending before or at this date
            order_by: Optional[str] = None  # e.g., 'execution_date', '-execution_date' for descending
    ):
        self.state = state if isinstance(state, list) else [state] if state else None
        self.execution_date_gte = execution_date_gte
        self.execution_date_lte = execution_date_lte
        self.start_date_gte = start_date_gte
        self.start_date_lte = start_date_lte
        self.end_date_gte = end_date_gte
        self.end_date_lte = end_date_lte
        self.order_by = order_by

    def to_params(self):
        params = {}
        if self.state:
            params['state'] = self.state
        if self.execution_date_gte:
            params['execution_date_gte'] = self.execution_date_gte
        if self.execution_date_lte:
            params['execution_date_lte'] = self.execution_date_lte
        if self.start_date_gte:
            params['start_date_gte'] = self.start_date_gte
        if self.start_date_lte:
            params['start_date_lte'] = self.start_date_lte
        if self.end_date_gte:
            params['end_date_gte'] = self.end_date_gte
        if self.end_date_lte:
            params['end_date_lte'] = self.end_date_lte
        if self.order_by:
            params['order_by'] = self.order_by
        return params

    def __and__(self, other):
        if not isinstance(other, AirflowQuery):
            raise TypeError("Operands must be AirflowQuery instances")

        return AirflowQuery(
            state=list(set(self.state or []).intersection(set(other.state or []))),
            execution_date_gte=max(filter(None, [self.execution_date_gte, other.execution_date_gte]), default=None),
            execution_date_lte=min(filter(None, [self.execution_date_lte, other.execution_date_lte]), default=None),
            start_date_gte=max(filter(None, [self.start_date_gte, other.start_date_gte]), default=None),
            start_date_lte=min(filter(None, [self.start_date_lte, other.start_date_lte]), default=None),
            end_date_gte=max(filter(None, [self.end_date_gte, other.end_date_gte]), default=None),
            end_date_lte=min(filter(None, [self.end_date_lte, other.end_date_lte]), default=None),
            order_by=self.order_by or other.order_by
        )

    def __or__(self, other):
        if not isinstance(other, AirflowQuery):
            raise TypeError("Operands must be AirflowQuery instances")

        return AirflowQuery(
            state=list(set(self.state or []).union(set(other.state or []))),
            execution_date_gte=min(filter(None, [self.execution_date_gte, other.execution_date_gte]), default=None),
            execution_date_lte=max(filter(None, [self.execution_date_lte, other.execution_date_lte]), default=None),
            start_date_gte=min(filter(None, [self.start_date_gte, other.start_date_gte]), default=None),
            start_date_lte=max(filter(None, [self.start_date_lte, other.start_date_lte]), default=None),
            end_date_gte=min(filter(None, [self.end_date_gte, other.end_date_gte]), default=None),
            end_date_lte=max(filter(None, [self.end_date_lte, other.end_date_lte]), default=None),
            order_by=self.order_by or other.order_by
        )

    def __str__(self):
        params = self.to_params()
        return ", ".join(f"{key}={value}" for key, value in params.items())

    @classmethod
    def parser(cls, q):
        if isinstance(q, cls):
            return q
        elif isinstance(q, dict):
            return cls(**q)
        return None


class AirflowClient(PureBeamPath):

    limit = 100

    def __init__(self, *pathsegments, client=None, hostname=None, port=None, username=None,
                 password=None, tls=False, verify=False, q=None, **kwargs):

        if not len(pathsegments):
            pathsegments = ('/',)

        super().__init__(*pathsegments, scheme='airflow', client=client, hostname=hostname, port=port,
                          username=username, password=password, tls=tls, verify=verify, **kwargs)

        if type(tls) is str:
            tls = (tls.lower() == 'true')

        tls = 'https' if tls else 'http'
        url = f'{tls}://{normalize_host(hostname, port)}/api/v1'
        self._url = url

        if client is None:
            configuration = Configuration(host=url, username=username, password=password)
            client = ApiClient(configuration)
            if not verify:
                client.configuration.verify_ssl = False
        self.client = client

        self._q: AirflowQuery | None = self.parse_query(q)

        l = len(self.parts[1:])
        self._level = {0: 'root', 1: 'dag', 2: 'dag_run', 3: 'task_instance'}[l]

    @property
    def level(self):
        l = len(self.parts[1:])
        if self.q is None:
            return {0: 'root', 1: 'dag', 2: 'dag_run', 3: 'task_instance'}[l]

        return {0: 'root', 1: 'filtered_dag_runs', 2: 'filtered_tasks', 3: 'task_instance'}[l]

    def parse_query(self, q):
        return AirflowQuery.parser(q) if q else None

    def health(self):
        try:
            res = self.client.rest_client.GET(f"{self._url}/health")
            res = json.loads(res.data)
            return res['metadatabase']['status']
        except Exception as e:
            return str(e)

    def ping(self):
        return self.health() == 'healthy'

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

    @property
    def q(self):
        return self._q

    # Enhanced methods inside AirflowClient:

    def iterdir(self, query: AirflowQuery = None):

        if query is None:
            query = self.q

        query_params = query.to_params() if query else {}

        if self.level == 'root':
            dags = self.dag_api.get_dags()
            for dag in dags.dags:
                yield self.joinpath(dag.dag_id)

        elif self.level == 'dag':
            i = 0
            while True:
                dag_runs = self.dag_run_api.get_dag_runs(
                    self.dag_id, offset=self.limit * i, limit=self.limit, **query_params
                )
                for dag_run in dag_runs.dag_runs:
                    yield self.joinpath(dag_run.dag_run_id)

                if len(dag_runs.dag_runs) < self.limit:
                    break
                i += 1

        elif self.level == 'dag_run':
            task_instances = self.task_instance_api.get_task_instances(
                self.dag_id, self.run_id, **query_params
            )
            for task_instance in task_instances:
                yield self.joinpath(task_instance.task_id)

        else:
            raise ValueError('Cannot iterate at task_instance level')

    def start(self, query: AirflowQuery = None):

        if query is None:
            query = self.q

        query_params = query.to_params() if query else {}

        if self.level == 'dag':
            dag_runs = self.dag_run_api.get_dag_runs(self.dag_id, **query_params)
            for dag_run in dag_runs.dag_runs:
                self.dag_run_api.post_dag_run(
                    self.dag_id, DAGRun(run_id=dag_run.dag_run_id)
                )

        elif self.level == 'dag_run':
            tasks = self.task_instance_api.get_task_instances(self.dag_id, self.run_id, **query_params)
            for task in tasks:
                self.task_instance_api.clear_task_instance(
                    self.dag_id, self.run_id, task.task_id, ClearTaskInstances()
                )

        elif self.level == 'task_instance':
            self.task_instance_api.clear_task_instance(
                self.dag_id, self.run_id, self.task_id, ClearTaskInstances()
            )

        else:
            raise ValueError('Starting operation not supported at root level')

    def stop(self, query: AirflowQuery = None):

        if query is None:
            query = self.q

        query_params = query.to_params() if query else {}

        if self.level == 'dag':
            dag_runs = self.dag_run_api.get_dag_runs(self.dag_id, **query_params)
            for dag_run in dag_runs.dag_runs:
                self.dag_run_api.update_dag_run_state(
                    self.dag_id, dag_run.dag_run_id, state="failed"
                )

        elif self.level == 'dag_run':
            tasks = self.task_instance_api.get_task_instances(self.dag_id, self.run_id, **query_params)
            for task in tasks:
                self.task_instance_api.update_task_instance_state(
                    self.dag_id, self.run_id, task.task_id, state="failed"
                )

        elif self.level == 'task_instance':
            self.task_instance_api.update_task_instance_state(
                self.dag_id, self.run_id, self.task_id, state="failed"
            )

        else:
            raise ValueError('Stopping operation not supported at root level')

    def unlink(self, query: AirflowQuery = None):

        if query is None:
            query = self.q

        query_params = query.to_params() if query else {}

        if self.level == 'dag':
            dag_runs = self.dag_run_api.get_dag_runs(self.dag_id, **query_params)
            for dag_run in dag_runs.dag_runs:
                self.dag_run_api.delete_dag_run(self.dag_id, dag_run.dag_run_id)

        elif self.level == 'dag_run':
            tasks = self.task_instance_api.get_task_instances(self.dag_id, self.run_id, **query_params)
            for task in tasks:
                raise ValueError('Deleting task instances individually is not supported')

        elif self.level == 'task_instance':
            raise ValueError('Cannot delete individual task_instance directly')

        else:
            raise ValueError('Cannot delete all dags at root level')

    def execution_date(self, item):
        # Extract execution time from an item if available
        return getattr(item, 'execution_date', None)

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
        return None

    def exists(self):
        return self.stat() is not None

    # set airflow environment variable
    def set_var(self, key, value):
        self.config_api.set_airflow_config(key, value)

    # get airflow environment variable
    def get_var(self, key):
        return self.config_api.get_airflow_config(key)

    def gen(self, path, **kwargs):
        hostname = kwargs.pop('hostname', self.hostname)
        port = kwargs.pop('port', self.port)
        username = kwargs.pop('username', self.username)
        password = kwargs.pop('password', self.password)
        fragment = kwargs.pop('fragment', self.fragment)
        params = kwargs.pop('params', self.params)
        query = kwargs.pop('query', {})
        q = kwargs.pop('q', self.q)

        # must be after extracting all other kwargs
        query = {**query, **kwargs}
        PathType = type(self)
        return PathType(path, client=self.client, hostname=hostname, port=port, username=username,
                        password=password, fragment=fragment, params=params, q=q, **query)

    def __repr__(self):
        s = str(self.url)
        if self.q is not None:
            fixed_q = str(self.q)
            if len(fixed_q) > 50:
                fixed_q = fixed_q[:50] + "..."

            s = f"{s} | query: {fixed_q}"
        return s

    def __and__(self, other):
        q = self.q
        if type(other) is AirflowClient:
            self._assert_other_type(other)
            query = other.q
        else:
            query = self.parse_query(other)

        if q is not None:
            query = q & query

        return self.gen(self.path, q=query)

    def iter_success(self, **kwargs):
        query = AirflowQuery(state='success', **kwargs)
        yield from self.iterdir(query)

    def iter_failed(self, **kwargs):
        query = AirflowQuery(state='failed', **kwargs)
        yield from self.iterdir(query)

    def iter_running(self, **kwargs):
        query = AirflowQuery(state='running', **kwargs)
        yield from self.iterdir(query)

    def start_failed(self, **kwargs):
        query = AirflowQuery(state='failed', **kwargs)
        self.start(query)

    def stop_running(self, **kwargs):
        query = AirflowQuery(state='running', **kwargs)
        self.stop(query)

    def unlink_failed(self, **kwargs):
        query = AirflowQuery(state='failed', **kwargs)
        self.unlink(query)

    def pause_dag(self):
        """ Pause a DAG """
        if self.level == 'dag':
            self.dag_api.update_dag(self.dag_id, {"is_paused": True})

    def unpause_dag(self):
        """ Unpause a DAG """
        if self.level == 'dag':
            self.dag_api.update_dag(self.dag_id, {"is_paused": False})

    # def iterdir(self):
    #     if self.level == 'root':
    #         # iter over all dags
    #         dags = self.dag_api.get_dags()
    #         for dag in dags.dags:
    #             yield self.joinpath(dag.dag_id)
    #
    #     elif self.level == 'dag':
    #         # iter over all dag_runs
    #         i = 0
    #         while True:
    #             dag_runs = self.dag_run_api.get_dag_runs(self.dag_id, offset=self.limit * i)
    #
    #             for dag_run in dag_runs.dag_runs:
    #                 yield self.joinpath(dag_run.dag_run_id)
    #
    #             if len(dag_runs.dag_runs) < self.limit:
    #                 break
    #
    #             i += 1
    #
    #     elif self.level == 'dag_run':
    #         # iter over all task_instances
    #         task_instances = self.task_instance_api.get_task_instances(self.dag_id, self.run_id)
    #         for task_instance in task_instances:
    #             yield self.joinpath(task_instance.task_id)
    #     else:
    #         raise ValueError(f'Cannot list directory for task_instance level')

    # def time_filter(self, l, start=None, end=None, pattern=None):
    #     # filter list of items based on time assume start and end are datetime
    #     # objects
    #     if start is None and end is None:
    #         return l
    #     l = [item for item in l if start <= self.execution_date(item) <= end]
    #     return l

    # def get_running(self, time_start=None, time_end=None):
    #     if self.level == 'root':
    #         # get all running dags
    #         dags = self.dag_api.get_dags()
    #         l = [dag.dag_id for dag in dags if dag.is_paused is False]
    #     elif self.level == 'dag':
    #         # get all running dag_runs
    #         dag_runs = self.dag_run_api.get_dag_runs(self.dag_id)
    #         l = [dag_run.run_id for dag_run in dag_runs if dag_run.state == 'running']
    #     elif self.level == 'dag_run':
    #         # get all running task_instances
    #         task_instances = self.task_instance_api.get_task_instances(self.dag_id, self.run_id)
    #         l = [task_instance.task_id for task_instance in task_instances if task_instance.state == 'running']
    #     else:
    #         raise ValueError(f'Cannot get running for task_instance level')
    #
    #     return self.time_filter(l, start=time_start, end=time_end)

    # def get_failed(self, time_start=None, time_end=None):
    #     if self.level == 'root':
    #         # get all failed dags
    #         dags = self.dag_api.get_dags()
    #         l = [dag.dag_id for dag in dags if dag.is_paused is False]
    #     elif self.level == 'dag':
    #         # get all failed dag_runs
    #         dag_runs = self.dag_run_api.get_dag_runs(self.dag_id)
    #         l = [dag_run.run_id for dag_run in dag_runs if dag_run.state == 'failed']
    #     elif self.level == 'dag_run':
    #         # get all failed task_instances
    #         task_instances = self.task_instance_api.get_task_instances(self.dag_id, self.run_id)
    #         l = [task_instance.task_id for task_instance in task_instances if task_instance.state == 'failed']
    #     else:
    #         raise ValueError(f'Cannot get failed for task_instance level')
    #
    #     return self.time_filter(l, start=time_start, end=time_end)

    # def get_success(self, time_start=None, time_end=None):
    #     if self.level == 'root':
    #         # get all success dags
    #         dags = self.dag_api.get_dags()
    #         l = [dag.dag_id for dag in dags if dag.is_paused is False]
    #     elif self.level == 'dag':
    #         # get all success dag_runs
    #         dag_runs = self.dag_run_api.get_dag_runs(self.dag_id)
    #         l = [dag_run.run_id for dag_run in dag_runs if dag_run.state == 'success']
    #     elif self.level == 'dag_run':
    #         # get all success task_instances
    #         task_instances = self.task_instance_api.get_task_instances(self.dag_id, self.run_id)
    #         l = [task_instance.task_id for task_instance in task_instances if task_instance.state == 'success']
    #     else:
    #         raise ValueError(f'Cannot get success for task_instance level')
    #
    #     return self.time_filter(l, start=time_start, end=time_end)

    # def unlink(self):
    #     if self.level == 'root':
    #         raise ValueError('Cannot delete all dags')
    #     elif self.level == 'dag':
    #         # delete dag
    #         self.dag_api.delete_dag(self.dag_id)
    #     elif self.level == 'dag_run':
    #         # delete dag_run
    #         self.dag_run_api.delete_dag_run(self.dag_id, self.run_id)
    #     else:
    #         raise ValueError('Cannot delete task_instance')

    # def clear(self, upstream=False, downstream=False, future=False, past=False, dry_run=False):
    #
    #     # clear task_instance
    #     clear = ClearTaskInstances(upstream=upstream, downstream=downstream, future=future,
    #                               past=past, dry_run=dry_run)
    #
    #     if self.level == 'task_instance':
    #         self.task_instance_api.clear_task_instance(self.dag_id, self.run_id, self.task_id, clear)

    # def rerun_failed(self, time_start=None, time_end=None):
    #     """ Rerun all failed DAGs or task instances in the given time range """
    #     failed_items = self.get_failed(time_start, time_end)
    #
    #     if self.level == 'dag_run':
    #         for task_id in failed_items:
    #             self.task_instance_api.clear_task_instance(self.dag_id, self.run_id, task_id, ClearTaskInstances())
    #
    #     elif self.level == 'dag':
    #         for run_id in failed_items:
    #             self.dag_run_api.post_dag_run(self.dag_id, DAGRun(run_id=run_id))
    #
    #     elif self.level == 'root':
    #         for dag_id in failed_items:
    #             self.dag_run_api.post_dag_run(dag_id, DAGRun(run_id=f"rerun_{dag_id}_{datetime.utcnow().isoformat()}"))

    # def start(self):
    #     """ Start a DAG Run or Task """
    #     if self.level == 'dag_run':
    #         self.dag_run_api.post_dag_run(self.dag_id, DAGRun(run_id=self.run_id))
    #     elif self.level == 'dag':
    #         run_id = f"manual_{datetime.utcnow().isoformat()}"
    #         self.dag_run_api.post_dag_run(self.dag_id, DAGRun(run_id=run_id))
    #     elif self.level == 'task_instance':
    #         self.task_instance_api.clear_task_instance(self.dag_id, self.run_id, self.task_id, ClearTaskInstances())
    #
    # def stop(self):
    #     """ Stop a running DAG Run or Task """
    #     if self.level == 'dag_run':
    #         self.dag_run_api.update_dag_run_state(self.dag_id, self.run_id, state="failed")
    #     elif self.level == 'task_instance':
    #         self.task_instance_api.update_task_instance_state(self.dag_id, self.run_id, self.task_id, state="failed")

    # def stop_all_active(self):
    #     """ Stop all active DAGs, DAG Runs, or Task Instances """
    #     running_items = self.get_running()
    #     if self.level == 'dag_run':
    #         for run_id in running_items:
    #             self.dag_run_api.update_dag_run_state(self.dag_id, run_id, state="failed")
    #     elif self.level == 'dag':
    #         for dag_id in running_items:
    #             runs = self.dag_run_api.get_dag_runs(dag_id)
    #             for run in runs:
    #                 if run.state == "running":
    #                     self.dag_run_api.update_dag_run_state(dag_id, run.run_id, state="failed")
