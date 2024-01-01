from ..config import BeamConfig, BeamParam
import os


class RayConfig(BeamConfig):
    parameters = [
        BeamParam('include-dashboard', bool, True, 'include ray-dashboard', ),
        BeamParam('runtime-env', str, None, 'runtime environment for ray', ),
        BeamParam('dashboard-port', int, None, 'dashboard port for ray', ),
        BeamParam('ray-address', str, 'auto', 'whether to link to existing ray cluster (auto/ip) or to '
                                              'set up a local ray instance (local/ip)', ),
    ]


class HPOConfig(RayConfig, BeamConfig):

    parameters = [
        BeamParam('gpus-per-trial', int, 1, 'number of gpus per trial',),
        BeamParam('cpus-per-trial', int, 4, 'number of cpus per trial',),
        BeamParam('num-trials', int, 1000, 'number of HPO trails',),
        BeamParam('n-jobs', int, 1, 'number of parallel HPO jobs',),
        BeamParam('time-budget-s', int, None, 'time budget in seconds',),
        BeamParam('print-results', bool, False, 'print the intermediate results during training',),
        BeamParam('enable-tqdm', bool, False, 'enable tqdm progress bar',),
        BeamParam('print-hyperparameters', bool, True, 'print the hyperparameters before training',),
        BeamParam('verbose', bool, True, 'verbose mode in hyperparameter optimization',),
        BeamParam('track-results', bool, False, 'track the results of each trial',),
        BeamParam('track-algorithms', bool, False, 'track the algorithms of each trial',),
        BeamParam('track-hparams', bool, True, 'track the hyperparameters of each trial',),
        BeamParam('track-suggestion', bool, True, 'track the suggestions of each trial',),
        BeamParam('hpo-path', str, os.path.join(os.path.expanduser('~'), 'beam_projects', 'hpo'),
                  'Root directory for Logs and results of Hyperparameter optimizations and the associated experiments'),
        BeamParam('stop', str, None, 'stop criteria for the HPO',),
        BeamParam('get-port-from-beam-port-range', bool, True, 'get port from beam port range',),
        BeamParam('replay-buffer-size', int, None, 'Maximal size of finite-memory hpo',),
        BeamParam('time-window', int, None, 'Maximal time window of finite-memory hpo',),

    ]

