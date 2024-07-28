__all__ = ['SparseSimilarity', 'TFIDF', 'DenseSimilarity', 'Similarities', 'TextSimilarity',
           'SparnnSimilarity', 'SimilarityConfig', 'TFIDFConfig', 'ChromaSimilarity']


def __getattr__(name):
    if name == 'SparseSimilarity':
        from .sparse import SparseSimilarity
        return SparseSimilarity
    elif name == 'TFIDF':
        from .tfidf import TFIDF
        return TFIDF
    elif name == 'DenseSimilarity':
        from .dense import DenseSimilarity
        return DenseSimilarity
    elif name == 'Similarities':
        from .core import Similarities
        return Similarities
    elif name == 'TextSimilarity':
        from .text import TextSimilarity
        return TextSimilarity
    elif name == 'SparnnSimilarity':
        from .sparnn import SparnnSimilarity
        return SparnnSimilarity
    elif name == 'SimilarityConfig':
        from .config import SimilarityConfig
        return SimilarityConfig
    elif name == 'TFIDFConfig':
        from .config import TFIDFConfig
        return TFIDFConfig
    elif name == 'ChromaSimilarity':
        from .chroma import ChromaSimilarity
        return ChromaSimilarity
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")


# Explicit imports for IDE
if len([]):
    from .sparse import SparseSimilarity
    from .tfidf import TFIDF
    from .dense import DenseSimilarity
    from .core import Similarities
    from .text import TextSimilarity
    from .sparnn import SparnnSimilarity
    from .config import SimilarityConfig, TFIDFConfig
    from .chroma import ChromaSimilarity
