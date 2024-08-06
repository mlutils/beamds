# Explicit imports for IDE
if len([]):

    from .resource import beam_worker, beam_dispatcher_server, beam_dispatcher, async_client, ray_client

    from .celery_dispatcher import CeleryDispatcher
    from .celery_worker import CeleryWorker

    from .ray_dispatcher import RayDispatcher, RayClient
    from .pool_dispatcher import ThreadedDispatcher, ProcessDispatcher

    from .async_client import AsyncClient
    from .async_server import AsyncRayServer, AsyncCeleryServer


__all__ = ['beam_worker', 'beam_dispatcher_server', 'beam_dispatcher', 'async_client', 'ray_client',
           'CeleryDispatcher', 'CeleryWorker', 'RayDispatcher', 'RayClient', 'ThreadedDispatcher', 'AsyncClient',
           'AsyncRayServer', 'AsyncCeleryServer', 'ProcessDispatcher']


def __getattr__(name):
    if name == 'beam_worker':
        from .resource import beam_worker
        return beam_worker
    elif name == 'beam_dispatcher_server':
        from .resource import beam_dispatcher_server
        return beam_dispatcher_server
    elif name == 'beam_dispatcher':
        from .resource import beam_dispatcher
        return beam_dispatcher
    elif name == 'async_client':
        from .resource import async_client
        return async_client
    elif name == 'ray_client':
        from .resource import ray_client
        return ray_client
    elif name == 'CeleryDispatcher':
        from .celery_dispatcher import CeleryDispatcher
        return CeleryDispatcher
    elif name == 'CeleryWorker':
        from .celery_worker import CeleryWorker
        return CeleryWorker
    elif name == 'RayDispatcher':
        from .ray_dispatcher import RayDispatcher
        return RayDispatcher
    elif name == 'RayClient':
        from .ray_dispatcher import RayClient
        return RayClient
    elif name == 'ThreadedDispatcher':
        from .pool_dispatcher import ThreadedDispatcher
        return ThreadedDispatcher
    elif name == 'ProcessDispatcher':
        from .pool_dispatcher import ProcessDispatcher
        return ProcessDispatcher
    elif name == 'AsyncClient':
        from .async_client import AsyncClient
        return AsyncClient
    elif name == 'AsyncRayServer':
        from .async_server import AsyncRayServer
        return AsyncRayServer
    elif name == 'AsyncCeleryServer':
        from .async_server import AsyncCeleryServer
        return AsyncCeleryServer
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
