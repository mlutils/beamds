import numpy as np
import torch
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
import datetime
from dataclasses import dataclass


from src.beam.core import Processor
from src.beam.transformer import Transformer
from src.beam.utils import check_type


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

    def str(self):
        return self.__str__()


class BeamScheduler(Processor):

    def __init__(self, obj, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj = obj
        self.is_transformer = isinstance(obj, Transformer)

    def run(self, method, *args, **kwargs):

        method = getattr(self.obj, method)
        result = method(*args, **kwargs)
        return result

    @task
    def execute(self, method, *args, **kwargs):
        return self.run(method, *args, **kwargs)


def beam_scheduler(obj, method, *args, **kwargs):

    scheduler = BeamScheduler(obj)

    @flow(name=obj.name, log_prints=True)
    def beam_scheduler_flow(obj, method, args=None, kwargs=None):

        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        return scheduler.execute(obj, method, args, kwargs)

    beam_scheduler_flow(obj=obj, method=method, args=args, kwargs=kwargs)


if __name__ == '__main__':
    beam_scheduler(DummyTransformer(name='dummy_transformer'), 'transform_callback',
                   x=[np.random.randn(100), ['asdf', 'dsf','erer'], torch.arange(10)])


    # @flow(task_runner=SequentialTaskRunner(), name=obj.name, log_prints=True)
    # def beam_scheduler_flow(obj, method, args=None, kwargs=None):
    #
    #     if args is None:
    #         args = ()
    #     if kwargs is None:
    #         kwargs = {}
    #
    #     return execute_method(obj, method, args, kwargs)
    #
    # beam_scheduler_flow.serve(obj=obj, method=method, args=args, kwargs=kwargs)
