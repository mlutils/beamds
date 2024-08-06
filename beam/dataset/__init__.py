if len([]):
    from .universal_dataset import UniversalDataset
    from .sampler import UniversalBatchSampler
    from .variants import TransformedDataset, LazyReplayBuffer
    from .tabular_dataset import TabularDataset

__all__ = ['UniversalDataset', 'UniversalBatchSampler', 'TransformedDataset', 'LazyReplayBuffer', 'TabularDataset']


def __getattr__(name):
    if name == 'UniversalDataset':
        from .universal_dataset import UniversalDataset
        return UniversalDataset
    elif name == 'UniversalBatchSampler':
        from .sampler import UniversalBatchSampler
        return UniversalBatchSampler
    elif name == 'TransformedDataset':
        from .variants import TransformedDataset
        return TransformedDataset
    elif name == 'LazyReplayBuffer':
        from .variants import LazyReplayBuffer
        return LazyReplayBuffer
    elif name == 'TabularDataset':
        from .tabular_dataset import TabularDataset
        return TabularDataset
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
