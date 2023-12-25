from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
import datetime

from ..core import Processor
from ..transformer import Transformer


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
def execute_method(obj, method, *args, **kwargs):
    scheduler = BeamScheduler(obj)
    return scheduler.run(method, *args, **kwargs)


def beam_scheduler(obj, method, *args, **kwargs):
    @flow(task_runner=SequentialTaskRunner(), name=obj.name, log_prints=True)
    def beam_scheduler_flow(obj, method, args=None, kwargs=None):

        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        return execute_method(obj, method, args, kwargs)

    beam_scheduler_flow.serve(obj=obj, method=method, args=args, kwargs=kwargs)


# Example usage
# flow.run(obj=your_object, method='your_method', args=your_args, kwargs=your_kwargs)




# def schedule(obj, method, ...scheduling args...):
#     scheduler = BeamScheduler(obj)
#     ...run this with prefect scheduler...

