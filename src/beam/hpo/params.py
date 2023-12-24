from ..config import BeamHparams, BeamParam
import os


class HPOConfig(BeamHparams):

    defaults = {}
    use_basic_parser = False
    parameters = [
        BeamParam('gpus-per-trial', int, 1, 'number of gpus per trial', model=False),
        BeamParam('cpus-per-trial', int, 4, 'number of cpus per trial', model=False),
        BeamParam('num-trials', int, 1000, 'number of HPO trails', model=False),
        BeamParam('n-jobs', int, 1, 'number of parallel HPO jobs', model=False),
        BeamParam('time-budget-s', int, None, 'time budget in seconds', model=False),
        BeamParam('print-results', bool, False, 'print the intermediate results during training', model=False),
        BeamParam('enable-tqdm', bool, False, 'enable tqdm progress bar', model=False),
        BeamParam('print-hyperparameters', bool, True, 'print the hyperparameters before training', model=False),
        BeamParam('verbose', bool, True, 'verbose mode in hyperparameter optimization', model=False),
        BeamParam('track-results', bool, False, 'track the results of each trial', model=False),
        BeamParam('track-algorithms', bool, False, 'track the algorithms of each trial', model=False),
        BeamParam('track-hparams', bool, True, 'track the hyperparameters of each trial', model=False),
        BeamParam('track-suggestion', bool, True, 'track the suggestions of each trial', model=False),
        BeamParam('hpo-path', str, os.path.join(os.path.expanduser('~'), 'beam_projects', 'hpo'),
                  'Root directory for Logs and results of Hyperparameter optimizations and the associated experiments.',
                  model=False),
        BeamParam('stop', str, None, 'stop criteria for the HPO', model=False),
        BeamParam('include-dashboard', bool, True, 'include ray-dashboard', model=False),
        BeamParam('runtime-env', str, None, 'runtime environment for ray', model=False),
        BeamParam('dashboard-port', int, None, 'dashboard port for ray', model=False),
        BeamParam('get-port-from-beam-port-range', bool, True, 'get port from beam port range', model=False),
        BeamParam('replay-buffer-size', int, None, 'Maximal size of finite-memory hpo', model=False),
        BeamParam('time-window', int, None, 'Maximal time window of finite-memory hpo', model=False),

    ]

