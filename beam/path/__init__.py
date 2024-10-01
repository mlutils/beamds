

__all__ = ['beam_path', 'in_memory_storage', 'local_copy', 'FileSystem', 'beam_key', 'PureBeamPath',
           'BeamURL', 'normalize_host', 'BeamResource', 'BeamFile', 'prioritized_extensions']


def __getattr__(name):
    if name == 'beam_key':
        from .resource import beam_key
        try:
            from ..config import KeysConfig
            beam_key.set_hparams(KeysConfig(silent=True, strict=True, load_config_files=False))
        except ImportError:
            pass
        return beam_key
    elif name == 'beam_path':
        from .resource import beam_path
        return beam_path
    elif name == 'in_memory_storage':
        from .resource import in_memory_storage
        return in_memory_storage
    elif name == 'local_copy':
        from .utils import local_copy
        return local_copy
    elif name == 'FileSystem':
        from .utils import FileSystem
        return FileSystem
    elif name == 'PureBeamPath':
        from .core import PureBeamPath
        return PureBeamPath
    elif name == 'BeamURL':
        from .core import BeamURL
        return BeamURL
    elif name == 'normalize_host':
        from .core import normalize_host
        return normalize_host
    elif name == 'BeamResource':
        from .core import BeamResource
        return BeamResource
    elif name == 'BeamFile':
        from .core import BeamFile
        return BeamFile
    elif name == 'BeamPath':
        from .models import BeamPath
        return BeamPath
    elif name == 'prioritized_extensions':
        from .core import prioritized_extensions
        return prioritized_extensions
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")


# Explicit imports for IDE
if len([]):
    from .resource import beam_path, in_memory_storage
    from .utils import local_copy, FileSystem
    from .core import PureBeamPath, BeamURL, normalize_host, BeamResource, BeamFile, prioritized_extensions
    from .models import BeamPath
    from .resource import beam_key
