if len([]):
    from .experiment import Experiment, nn_algorithm_generator, simple_algorithm_generator
    from .reporter import BeamReport


__all__ = ['Experiment', 'nn_algorithm_generator', 'simple_algorithm_generator', 'BeamReport']


def __getattr__(name):
    if name == 'Experiment':
        from .experiment import Experiment
        return Experiment

    if name == 'nn_algorithm_generator':
        from .experiment import nn_algorithm_generator
        return nn_algorithm_generator

    if name == 'simple_algorithm_generator':
        from .experiment import simple_algorithm_generator
        return simple_algorithm_generator

    if name == 'BeamReport':
        from .reporter import BeamReport
        return BeamReport


