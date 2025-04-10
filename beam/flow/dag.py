from ..base import BeamBase
from .config import AirFlowConfig
from airflow import DAG
from datetime import timedelta


class BeamDag(BeamBase):

    def __init__(self, name, *args, build_function=None, graph_function=None, **kwargs):
        super().__init__(*args, _config_scheme=AirFlowConfig, **kwargs)
        self._name = name
        self.tasks = {}
        self.build_function = build_function
        self.graph_function = graph_function
        self.default_tasks_args = {
            'owner': self.hparams.dag_owner,
            'depends_on_past': self.hparams.depends_on_past,
            'start_date': self.hparams.start_date,
            'email_on_failure': self.hparams.email_on_failure,
            'email_on_retry': self.hparams.email_on_retry,
            'retries': self.hparams.dag_retries,
            'retry_delay': timedelta(minutes=self.hparams.dag_retry_delay),
        }

    def add_task(self, task):
        self.tasks[task.name] = task

    def add_tasks(self, tasks):
        for task in tasks:
            self.add_task(task)

    def _build(self, *args, **kwargs):
        pass

    def _graph(self, *args, **kwargs):
        pass

    def build(self, *args, **kwargs):
        # this is where the tasks are built
        if self.build_function:
            return self.build_function(*args, **kwargs)
        return self._build(*args, **kwargs)

    def graph(self, *args, **kwargs):
        # this is where the tasks are graphed
        if self.graph_function:
            return self.graph_function(*args, **kwargs)
        return self._graph(*args, **kwargs)

    def __getitem__(self, item):
        return self.tasks[item]

    def execute(self, *args, **kwargs):
        with DAG(dag_id=self._name, default_args=self.default_tasks_args,
                 ) as dag:
            self.build(*args, **kwargs)
            self.graph(*args, **kwargs)