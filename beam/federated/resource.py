import socket

from ..distributed import ThreadedDispatcher
from ..distributed import RayDispatcher, RayClient
from .federated import BeamFederated
from ..utils import find_port, GPUManager
from ..logging import beam_logger as logger
from ..base import tmp_paths


def federated_executor(func=None, world_size=1, framework='ddp', distributed_backend='nccl', host=None,
                       port=None, func_args=None, func_kwargs=None, kv_store='tcp', kv_store_path=None,
                       kv_store_timeout=300, kv_store_port=None, ray_address=None, ray_kwargs=None, num_gpus=1,
                       num_cpus=4, remote_kwargs=None, **kwargs):

    # ray_cluster = RayCluster(address=ray_address, ray_kwargs=ray_kwargs)
    # if host is None:
    #     host = ray_cluster.head_node_ip
    #     logger.info(f'Host is not specified, using the head node IP: {host}')

    if port is None:
        port = find_port(application='distributed')

    if kv_store_port is None and kv_store == 'tcp':
        kv_store_port = find_port(application='distributed', blacklist=[port])

    logger.info(f'Multiprocessing port is: {port}, KV store port is: {kv_store_port}')

    if kv_store_path is None and kv_store == 'file':
        kv_store_path = tmp_paths.beam_kv_store

    if kv_store_timeout is None:
        kv_store_timeout = 300

    remote_kwargs = remote_kwargs if remote_kwargs is not None else {}
    if num_cpus is not None:
        remote_kwargs['num_cpus'] = num_cpus
    if num_gpus is not None:
        remote_kwargs['num_gpus'] = num_gpus

    logical_devices = None
    if num_gpus is not None:

        hostname = socket.gethostname()

        RayGPUAllocator = RayDispatcher(GPUManager, remote_kwargs={'num_gpus': num_gpus,
                                                             'resources': {f"hostname_{hostname}": 1}},
                                              ray_kwargs=ray_kwargs, asynchronous=False)
        gpu_allocator = RayGPUAllocator()
        logical_devices = GPUManager.logical_devices(physical_devices=gpu_allocator.physical_devices())

    LocalWorkeClass = ThreadedDispatcher(BeamFederated, asynchronous=True)

    remote_workers = [LocalWorkeClass(func=func, rank=0, world_size=world_size, framework=framework,
                                      distributed_backend=distributed_backend, host=host, port=port,
                                      func_args=func_args, func_kwargs=func_kwargs,
                                      kv_store=kv_store, kv_store_path=kv_store_path, devices=logical_devices,
                                      kv_store_timeout=kv_store_timeout, kv_store_port=kv_store_port, **kwargs)]

    if world_size > 1:
        RemoteWorkerClass = RayDispatcher(BeamFederated, address=ray_address, ray_kwargs=ray_kwargs,
                                          remote_kwargs=remote_kwargs, asynchronous=True)

        for rank in range(1, world_size):
            remote_workers.append(RemoteWorkerClass(func=func, rank=rank, world_size=world_size, framework=framework,
                                                    distributed_backend=distributed_backend, host=host, port=port,
                                                    func_args=func_args, func_kwargs=func_kwargs,
                                                    kv_store=kv_store, kv_store_path=kv_store_path,
                                                    kv_store_timeout=kv_store_timeout, kv_store_port=kv_store_port,
                                                    **kwargs))

    return remote_workers




