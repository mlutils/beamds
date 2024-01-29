import os
from ..logger import beam_logger as logger
from ..core import Processor

import ray


class BeamCoalition(Processor):
    def __init__(self, *args, func=None, rank=0, world_size=1, framework='ddp', distributed_backend='nccl', host=None,
                 port=None, func_args=None, func_kwargs=None, done_event=None, **kwargs):

        super().__init__(*args, training_framework=framework, mp_port=port, mp_ip=host,
                         distributed_backend=distributed_backend, **kwargs)

        self.rank = rank
        self.world_size = world_size
        self.host = self.get_hparam('mp_ip')
        self.port = str(self.get_hparam('mp_port'))
        self.backend = self.get_hparam('distributed_backend')
        self.framework = self.get_hparam('training_framework')

        self.func = func
        self.func_args = func_args if func_args is not None else []
        self.func_kwargs = func_kwargs if func_kwargs is not None else {}
        self.done_event = done_event

        self._init_distributed()

    def _init_distributed(self):
        os.environ['MASTER_ADDR'] = self.host
        os.environ['MASTER_PORT'] = self.port

        logger.info(f"Initializing distributed training with backend={self.backend} and framework={self.framework}")
        if self.framework == 'ddp':
            # initialize the process group
            import torch.distributed as dist
            dist.init_process_group(self.backend, rank=self.rank, world_size=self.world_size)
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

    @property
    def is_done(self):
        return ray.get(self.done_event)

    def set_done(self):
        if self.rank == 0:
            ray.put(True, self.done_event)

    def __call__(self, *args, func=None, **kwargs):
        if func is None:
            func = self.func

        args = list(args) + self.func_args
        kwargs = {**kwargs, **self.func_kwargs}

        return func(self, *args, **kwargs)