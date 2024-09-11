if len([]):
    from .safe_lazy_importer import lazy_importer


__all__ = ['lazy_importer']


def __getattr__(name):
    if name == 'lazy_importer':
        from .safe_lazy_importer import lazy_importer
        return lazy_importer
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
