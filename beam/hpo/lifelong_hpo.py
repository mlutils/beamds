import datetime
from ..utils import cached_property
from typing import Container

from optuna import Study
from optuna.trial import TrialState, FrozenTrial

from .optuna import OptunaBase
from .params import HPOConfig
from ..processor import Processor


class FiniteMemoryStudy(Study):

    def __init__(self, *args, replay_buffer_size=None, time_window=None,  **kwargs):
        super().__init__(*args, **kwargs)
        assert replay_buffer_size is not None or time_window is not None, \
            "Either replay_buffer_size or time_window must be specified"
        self.replay_buffer_size = replay_buffer_size
        self.time_window = time_window

    def _get_trials(
        self,
        deepcopy: bool = True,
        states: Container[TrialState] | None = None,
        use_cache: bool = False,
    ) -> list[FrozenTrial]:

        trials = super()._get_trials(deepcopy=deepcopy, states=states, use_cache=use_cache)

        if self.replay_buffer_size is not None:
            trials = trials[-self.replay_buffer_size:]
        elif self.time_window is not None:
            trials = [trial for trial in trials if trial.datetime_start > datetime.datetime.now() - self.time_window]
        else:
            raise ValueError("Either replay_buffer_size or time_window must be specified")

        return trials


class LifelongHPO(Processor, OptunaBase):

    def __init__(self, *args, **kwargs):

        hpo_config = HPOConfig(*args, **kwargs)
        super().__init__(hparams=hpo_config)
        self.time_window = self.hparams.get('time_window')
        self.replay_buffer_size = self.hparams.get('replay_buffer_size')
        self.direction = self.hparams.get('mode')




    @cached_property
    def sampler(self):
        return self.get_sampler()

    @cached_property
    def study(self) -> FiniteMemoryStudy:
        return FiniteMemoryStudy(storage=self.storage, sampler=self.sampler, pruner=self.pruner,
                                 study_name=self.name, direction=self.direction,
                                 load_study=self.load_study, replay_buffer_size=self.replay_buffer_size,
                                 time_window=self.time_window)