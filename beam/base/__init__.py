__all__ = ['BeamBase', 'beam_cache', 'BeamURL', 'BeamResource', 'resource_names']


# Explicit imports for IDE
if len([]):
    from .base_class import BeamBase
    from .base_cache import beam_cache
    from .beam_url import BeamURL
    from .beam_resource import BeamResource
    from .beam_resource import resource_names


def __getattr__(name):
    if name == 'BeamBase':
        from .base_class import BeamBase
        return BeamBase
    elif name == 'beam_cache':
        from .base_cache import beam_cache
        return beam_cache
    elif name == 'BeamURL':
        from .beam_url import BeamURL
        return BeamURL
    elif name == 'BeamResource':
        from .beam_resource import BeamResource
        return BeamResource
    elif name == 'resource_names':
        from .beam_resource import resource_names
        return resource_names
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")

