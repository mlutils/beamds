# Explicit imports for IDE
if len([]):
    from .neural_algorithm import NeuralAlgorithm
    from .core_algorithm import Algorithm
    from .catboost_algorithm import CBAlgorithm
    from .group_expansion import TextGroupExpansionAlgorithm, GroupExpansionAlgorithm
    from .config import CatboostConfig, CatboostExperimentConfig

__all__ = ['NeuralAlgorithm', 'Algorithm', 'CBAlgorithm', 'TextGroupExpansionAlgorithm',
           'GroupExpansionAlgorithm', 'CatboostConfig', 'CatboostExperimentConfig']


def __getattr__(name):
    if name == 'NeuralAlgorithm':
        from .neural_algorithm import NeuralAlgorithm
        return NeuralAlgorithm
    elif name == 'Algorithm':
        from .core_algorithm import Algorithm
        return Algorithm
    elif name == 'CBAlgorithm':
        from .catboost_algorithm import CBAlgorithm
        return CBAlgorithm
    elif name == 'TextGroupExpansionAlgorithm':
        from .group_expansion import TextGroupExpansionAlgorithm
        return TextGroupExpansionAlgorithm
    elif name == 'GroupExpansionAlgorithm':
        from .group_expansion import GroupExpansionAlgorithm
        return GroupExpansionAlgorithm
    elif name == 'CatboostConfig':
        from .config import CatboostConfig
        return CatboostConfig
    elif name == 'CatboostExperimentConfig':
        from .config import CatboostExperimentConfig
        return CatboostExperimentConfig
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")

