import json
import socket
from datetime import timedelta

import torch.distributed as dist
import torch
import os

from ..logging import beam_logger as logger
from ..processor import Processor
from ..utils import has_kwargs, GPUManager


class BeamFederated(Processor):
    def __init__(self, *args, func=None, rank=0, world_size=1, framework='ddp', distributed_backend='nccl', host=None,
                 port=None, func_args=None, func_kwargs=None, kv_store='tcp', kv_store_path=None,
                 kv_store_timeout=300, kv_store_port=None, devices=None, **kwargs):

        super().__init__(*args, training_framework=framework, mp_port=port, mp_ip=host,
                         kv_store_path=kv_store_path, kv_store_timeout=kv_store_timeout, kv_store_port=kv_store_port,
                         distributed_backend=distributed_backend, kv_store=kv_store, **kwargs)

        self.rank = rank
        self.world_size = world_size

        self.func = func
        self.func_args = func_args if func_args is not None else []
        self.func_kwargs = func_kwargs if func_kwargs is not None else {}

        self.host = self.get_hparam('mp_ip') or 'localhost'

        self.port = str(self.get_hparam('mp_port'))
        self.backend = self.get_hparam('distributed_backend')
        self.framework = self.get_hparam('training_framework')

        if devices is None:
            devices = list(range(torch.cuda.device_count()))
        else:
            if not isinstance(devices, list):
                devices = [int(devices)]
            else:
                devices = [int(d) for d in devices]
        self.devices = devices

        self.kv_store = self.get_hparam('kv_store')
        self.kv_store_path = self.get_hparam('kv_store_path')
        self.kv_store_timeout = self.get_hparam('kv_store_timeout')
        self.kv_store_port = self.get_hparam('kv_store_port')

        logger.info(f"Rank {self.rank}/{self.world_size} connects to the KV store: {self.kv_store}")
        if self.kv_store == 'tcp':
            self.store = dist.TCPStore(host_name=self.host, port=int(self.kv_store_port), world_size=self.world_size,
                                       is_master=(self.rank == 0), timeout=timedelta(seconds=self.kv_store_timeout))
        elif self.kv_store == 'hash':
            self.store = dist.HashStore()
        elif self.kv_store == 'file':
            self.store = dist.FileStore(self.kv_store_path, self.world_size)
        else:
            raise ValueError(f"Unknown kv_store: {self.kv_store}")

        if self.world_size > 1:
            self._init_distributed()

    @property
    def physical_devices(self):
        return GPUManager.physical_devices(self.devices)

    @property
    def hostname(self):
        return socket.gethostname()

    def _init_distributed(self):
        os.environ['MASTER_ADDR'] = self.host
        os.environ['MASTER_PORT'] = self.port

        logger.info(f"Rank {self.rank}/{self.world_size} is initializing distributed training (host={self.host}, port={self.port})")
        logger.info(f"Initializing distributed training with backend={self.backend} and framework={self.framework}")
        if self.framework == 'ddp':
            # initialize the process group
            dist.init_process_group(self.backend, rank=self.rank, world_size=self.world_size, store=self.store)
        elif self.framework == 'deepspeed':

            # make sure that mpi path is in the path variable

            os.environ['LOCAL_RANK'] = str(self.rank)
            os.environ['RANK'] = str(self.rank)
            os.environ['WORLD_SIZE'] = str(self.world_size)
            os.environ['OMPI_COMM_WORLD_SIZE'] = str(self.world_size)
            os.environ['OMPI_COMM_WORLD_RANK'] = str(self.rank)

            import deepspeed
            # deepspeed.init_distributed(dist_backend=backend, auto_mpi_discovery=backend == 'mpi',
            #                            rank=rank, world_size=world_size, distributed_port=port)
            deepspeed.init_distributed(dist_backend=self.backend, auto_mpi_discovery=False,
                                       rank=self.rank, world_size=self.world_size, distributed_port=self.port)

        else:
            raise ValueError(f"Unknown distributed framework: {self.framework}")

    def barrier(self, group=None, async_op=None, device_ids=None):
        kwargs = {}
        if group is not None:
            kwargs['group'] = group
        if async_op is not None:
            kwargs['async_op'] = async_op
        if device_ids is not None:
            kwargs['device_ids'] = device_ids
        if self.framework == 'ddp':
            import torch.distributed as dist
            dist.barrier(**kwargs)
        elif self.framework == 'deepspeed':
            import deepspeed
            deepspeed.comm.barrier(**kwargs)
        else:
            raise ValueError(f"Unknown distributed framework: {self.framework}")

    def __getitem__(self, item):
        value = self.store.get(item)
        return json.loads(value)

    def __setitem__(self, key, value):
        value = json.dumps(value)
        return self.store.set(key, value)

    def __delitem__(self, key):
        return self.store.delete_key(key)

    def add_to_key(self, key, value):
        return self.store.add(key, value)

    def num_keys(self):
        return self.store.num_keys()

    def wait(self, keys):
        if isinstance(keys, str):
            keys = [keys]
        self.store.wait(keys)

    def __call__(self, *args, func=None, **kwargs):
        if func is None:
            func = self.func

        args = list(args) + self.func_args
        kwargs = {**kwargs, **self.func_kwargs}
        kwargs['manager'] = self

        return func(*args, **kwargs)

    def cleanup(self):
        if self.world_size > 1:
            if self.framework == 'ddp':
                dist.destroy_process_group()
            elif self.framework == 'deepspeed':
                pass
            else:
                raise ValueError(f"Unknown distributed framework: {self.framework}")


# For MPI use the spinningup resource:
# https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/utils/mpi_tools.py#L4