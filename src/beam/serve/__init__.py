# Explicit imports for IDE
if len([]):
    from .remote import beam_server, beam_client, triton_client
    from .config import BeamServeConfig

__all__ = ['beam_server', 'beam_client', 'triton_client', 'BeamServeConfig']


def __getattr__(name):
    if name == 'beam_server':
        from .remote import beam_server
        return beam_server
    elif name == 'beam_client':
        from .remote import beam_client
        return beam_client
    elif name == 'triton_client':
        from .remote import triton_client
        return triton_client
    elif name == 'BeamServeConfig':
        from .config import BeamServeConfig
        return BeamServeConfig
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")

