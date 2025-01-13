if len([]):

    from .feature import BeamFeature, FeaturesAggregator, FeaturesCategories, ScalingFeature, BinarizedFeature
    from .feature import CategorizedFeature, DiscretizedFeature, InverseOneHotFeature
    from .text_features import FeaturesCategories, SparseEmbeddingFeature, DenseEmbeddingFeature
    from .node2vec import Node2Vec


__all__ = ['BeamFeature', 'FeaturesAggregator', 'FeaturesCategories', 'ScalingFeature', 'BinarizedFeature',
              'CategorizedFeature', 'DiscretizedFeature', 'InverseOneHotFeature', 'SparseEmbeddingFeature',
              'DenseEmbeddingFeature', 'Node2Vec']

def __getattr__(name):
    if name == "BeamFeature":
        from .feature import BeamFeature
        return BeamFeature
    elif name == "FeaturesAggregator":
        from .feature import FeaturesAggregator
        return FeaturesAggregator
    elif name == "FeaturesCategories":
        from .feature import FeaturesCategories
        return FeaturesCategories
    elif name == "ScalingFeature":
        from .feature import ScalingFeature
        return ScalingFeature
    elif name == "BinarizedFeature":
        from .feature import BinarizedFeature
        return BinarizedFeature
    elif name == "CategorizedFeature":
        from .feature import CategorizedFeature
        return CategorizedFeature
    elif name == "DiscretizedFeature":
        from .feature import DiscretizedFeature
        return DiscretizedFeature
    elif name == "InverseOneHotFeature":
        from .feature import InverseOneHotFeature
        return InverseOneHotFeature
    elif name == "SparseEmbeddingFeature":
        from .text_features import SparseEmbeddingFeature
        return SparseEmbeddingFeature
    elif name == "DenseEmbeddingFeature":
        from .text_features import DenseEmbeddingFeature
        return DenseEmbeddingFeature
    elif name == "Node2Vec":
        from .node2vec import Node2Vec
        return Node2Vec
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
