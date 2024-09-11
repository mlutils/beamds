if len([]):
    from .safe_lazy_importer import lazy_importer
    from .safe_imports.torch import torch
    from .safe_imports.scipy import scipy
    from .safe_imports.torchvision import transforms


__all__ = ['lazy_importer', 'torch', 'scipy', 'transforms']


def __getattr__(name):
    if name == 'lazy_importer':
        from .safe_lazy_importer import lazy_importer
        return lazy_importer
    elif name == 'torch':
        from .safe_imports.torch import torch
        return torch
    elif name == 'scipy':
        from .safe_imports.scipy import scipy
        return scipy
    elif name == 'transforms':
        from .safe_imports.torchvision import transforms
        return transforms
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
