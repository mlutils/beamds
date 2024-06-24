import datetime
import time
from typing import Union

from ray.tune.stopper import Stopper

from ..logging import beam_logger as logger


class TimeoutStopper(Stopper):
    """Stops all trials after a certain timeout.

    This stopper is automatically created when the `time_budget_s`
    argument is passed to `tune.run()`.

    Args:
        timeout: Either a number specifying the timeout in seconds, or
            a `datetime.timedelta` object.
    """

    def __init__(self, timeout: Union[int, float, datetime.timedelta]):
        from datetime import timedelta

        if isinstance(timeout, timedelta):
            self._timeout_seconds = timeout.total_seconds()
        elif isinstance(timeout, (int, float)):
            self._timeout_seconds = timeout
        else:
            raise ValueError(
                "`timeout` parameter has to be either a number or a "
                "`datetime.timedelta` object. Found: {}".format(type(timeout))
            )

        self._budget = self._timeout_seconds

        self.start_time = {}

    def stop_all(self):
        return False

    def __call__(self, trial_id, result):
        now = time.time()

        if trial_id in self.start_time:
            if now - self.start_time[trial_id] >= self._budget:
                logger.info(
                    f"Reached timeout of {self._timeout_seconds} seconds. "
                    f"Stopping this trials."
                )
                return True
        else:
            self.start_time[trial_id] = now

        return False