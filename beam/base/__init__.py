__all__ = ['BeamBase', 'beam_cache', 'BeamURL', 'BeamResource', 'resource_names', 'base_paths', 'tmp_paths',
           'Iloc', 'Loc', 'Key', 'Groups', 'return_none']


# Explicit imports for IDE
if len([]):
    from .base_class import BeamBase
    from .base_cache import beam_cache
    from .beam_url import BeamURL
    from .beam_resource import BeamResource
    from .beam_resource import resource_names
    from .consts import base_paths
    from .consts import tmp_paths
    from .elements import Iloc, Loc, Key, Groups, return_none


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
    elif name == 'base_paths':
        from .consts import base_paths
        return base_paths
    elif name == 'tmp_paths':
        from .consts import tmp_paths
        return tmp_paths
    elif name == 'Iloc':
        from .elements import Iloc
        return Iloc
    elif name == 'Loc':
        from .elements import Loc
        return Loc
    elif name == 'Key':
        from .elements import Key
        return Key
    elif name == 'Groups':
        from .elements import Groups
        return Groups
    elif name == 'return_none':
        from .elements import return_none
        return return_none
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")

