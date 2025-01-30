from dataclasses import dataclass
from prefect import flow, task

from ..path import beam_path
from ..utils import beam_hash
from ..base import BeamBase
from ..transformer import Transformer


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


class BeamFlow(BeamBase):

    def __init__(self, obj, *args, description=None, log_prints=True, retries=None, retry_delay_seconds=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj = obj
        self.description = description
        self.log_prints = log_prints
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self._name = obj.name if hasattr(obj, 'name') else self._name
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





# @task
# def execute_method(obj, method, *args, **kwargs):
#     scheduler = BeamFlow(obj)
#     return scheduler.run(method, *args, **kwargs)
#
#
# def beam_scheduler(obj, method, *args, **kwargs):
#     @flow(task_runner=SequentialTaskRunner(), name=obj.name, log_prints=True)
#     def beam_scheduler_flow(obj, method, args=None, kwargs=None):
#
#         if args is None:
#             args = ()
#         if kwargs is None:
#             kwargs = {}
#
#         return execute_method(obj, method, args, kwargs)
#
#     beam_scheduler_flow.serve(obj=obj, method=method, args=args, kwargs=kwargs)


# Example usage
# flow.run(obj=your_object, method='your_method', args=your_args, kwargs=your_kwargs)




# def schedule(obj, method, ...scheduling args...):
#     scheduler = BeamScheduler(obj)
#     ...run this with prefect scheduler...

