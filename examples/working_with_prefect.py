import numpy as np
import torch
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
import datetime
from dataclasses import dataclass


from src.beam.core import Processor
from src.beam.transformer import Transformer
from src.beam.utils import check_type
from src.beam import beam_path, beam_hash


class DummyTransformer(Transformer):
    def transform_callback(self, x, key=None, is_chunk=False, fit=False, path=None, **kwargs):
        res = []
        for xi in x:
            res.append(check_type(xi))
        return res


@dataclass
class CronSchedule:
    hour: int = 0           # Hour of the day (0-23)
    minute: int = 0         # Minute of the hour (0-59)
    day_of_month: str = '*' # Day of the month (1-31) or '*' for every day of the month
    month: str = '*'        # Month of the year (1-12) or '*' for every month
    day_of_week: str = '*'  # Day of the week (0-6, where 0 is Sunday) or '*' for every day of the week

    def __str__(self):
        """
        Return a cron expression string based on the schedule parameters.
        """
        return f"{self.minute} {self.hour} {self.day_of_month} {self.month} {self.day_of_week}"

    @property
    def str(self):
        return self.__str__()


class BeamFlow(Processor):

    def __init__(self, obj, *args, description=None, log_prints=True, retries=None, retry_delay_seconds=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj = obj
        self.description = description
        self.log_prints = log_prints
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self.name = obj.name if hasattr(obj, 'name') else self.name
        self.is_transformer = isinstance(obj, Transformer)
        self.run = task(description=self.description, name=self.name)(self._run)
        self.flow = flow(name=self.name, log_prints=self.log_prints,
                         retries=self.retries, retry_delay_seconds=self.retry_delay_seconds)(self._flow)
        self.monitored_paths = {}

    def schedule(self, method, hour: int = 0, minute: int = 0, day_of_month: str = '*',
                 month: str = '*', day_of_week: str = '*', cron_schedule: CronSchedule = None):

        if cron_schedule is None:
            cron_schedule = CronSchedule(hour=hour, minute=minute, day_of_month=day_of_month,
                                         month=month, day_of_week=day_of_week)

        self.flow.serve(method, schedule=cron_schedule.str)

    def monitor_path(self, path, lazy=False):
        path = beam_path(path)
        walk = path.walk()
        walk_hash = beam_hash(walk)

        path_uri = path.as_uri()
        if path_uri not in self.monitored_paths:
            self.monitored_paths[walk_hash] = path
            return not lazy
        elif self.monitored_paths[path_uri] != walk_hash:
            self.monitored_paths[path_uri] = walk_hash
            return True
        return False

    def _run(self, method, *args, **kwargs):

        method = getattr(self.obj, method)
        result = method(*args, **kwargs)
        return result

    def _flow(self, method, args, kwargs):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        return self.run(method, args=args, kwargs=kwargs)


def beam_flow(obj, method, *args, **kwargs):

    bf = BeamFlow(obj)

    @flow(name=obj.name, log_prints=True)
    def _beam_flow(obj, method, args=None, kwargs=None):

        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        return bf.run(obj, method, args, kwargs)

    _beam_flow(obj=obj, method=method, args=args, kwargs=kwargs)


if __name__ == '__main__':
    beam_flow(DummyTransformer(name='dummy_transformer'), 'transform_callback',
              x=[np.random.randn(100), ['asdf', 'dsf','erer'], torch.arange(10)])

