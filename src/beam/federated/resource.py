import os

from ..distributed import ThreadedDispatcher
from ..distributed import RayDispatcher
from .core import BeamFederated
from ..utils import find_port


class RayGPUAllocator:
    def get_devices(self):
        return os.environ['CUDA_VISIBLE_DEVICES']


def federated_learner(func, world_size, framework='ddp', distributed_backend='nccl', host=None,
                 port=None, func_args=None, func_kwargs=None, done_event=None, kv_store='tcp', kv_store_path=None,
                 kv_store_timeout=300, kv_store_port=None, ray_address=None, ray_kwargs=None, num_gpus=1,
                 num_cpus=4, remote_kwargs=None, **kwargs):

    if host is None:
        host = 'localhost'

    if port is None:
        port = find_port(application='distributed')

    if kv_store_port is None and kv_store == 'tcp':
        kv_store_port = find_port(application='distributed')

    if kv_store_path is None and kv_store == 'file':
        kv_store_path = '/tmp/beam_kv_store'

    if kv_store_timeout is None:
        kv_store_timeout = 300

    remote_kwargs = remote_kwargs if remote_kwargs is not None else {}
    if num_cpus is not None:
        remote_kwargs['num_cpus'] = num_cpus
    if num_gpus is not None:
        remote_kwargs['num_gpus'] = num_gpus

    if num_cpus is not None:
        RayGPUAllocatorRemote = RayDispatcher(RayGPUAllocator,
                                              remote_kwargs={'num_gpus': num_cpus, 'resources': {"local_node": 1}},
                                              asynchronous=False)
        gpu_allocator = RayGPUAllocatorRemote()
        local_devices = gpu_allocator.get_devices()

    LocalWorkeClass = ThreadedDispatcher(BeamFederated)
    RemoteWorkerClass = RayDispatcher(BeamFederated, address=ray_address, ray_kwargs=ray_kwargs,
                                      remote_kwargs=remote_kwargs)

    remote_workers = [LocalWorkeClass(func=func, rank=0, world_size=world_size, framework=framework,
                                      distributed_backend=distributed_backend, host=host, port=port,
                                      func_args=func_args, func_kwargs=func_kwargs, done_event=done_event,
                                      kv_store=kv_store, kv_store_path=kv_store_path,
                                      kv_store_timeout=kv_store_timeout, kv_store_port=kv_store_port, **kwargs)]

    for rank in range(1, world_size):
        remote_workers.append(RemoteWorkerClass(func=func, rank=rank, world_size=world_size, framework=framework,
                                                distributed_backend=distributed_backend, host=host, port=port,
                                                func_args=func_args, func_kwargs=func_kwargs, done_event=done_event,
                                                kv_store=kv_store, kv_store_path=kv_store_path,
                                                kv_store_timeout=kv_store_timeout, kv_store_port=kv_store_port,
                                                **kwargs))

    return remote_workers




