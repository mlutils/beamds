import copy
import torch.multiprocessing as mp
import inspect
import traceback
import time
import os

from ..utils import (set_seed, is_notebook,)
from ..path import beam_path
from ..logging import beam_logger as logger
from ..config import get_beam_llm, BeamConfig


def training_closure(rank, world_size, experiment, alg, **kwargs):

    if not rank:
        alg.training_closure(**kwargs)
        checkpoint_file = experiment.checkpoints_dir.joinpath(f'checkpoint_{alg.epoch + 1:06d}')
        alg.save_checkpoint(checkpoint_file)


def worker_executor(experiment, alg, algorithm_generator, manager=None, dataset=None, alg_args=None, alg_kwargs=None,
                    dataset_args=None, dataset_kwargs=None, **kwargs):

    if manager is None:
        rank = 0
        world_size = 1
        manager = {}
    else:
        rank = manager.rank
        world_size = manager.world_size

    if rank == 0:
        manager['is_done'] = False

    experiment.set_rank(rank, world_size, devices=manager.devices)
    set_seed(seed=experiment.hparams.seed, constant=rank + 1, increment=False,
             deterministic=experiment.hparams.deterministic)

    alg = algorithm_generator(experiment, alg, dataset=dataset, alg_args=alg_args, alg_kwargs=alg_kwargs,
                                   dataset_args=dataset_args, dataset_kwargs=dataset_kwargs, rank=rank)

    if rank == 0:
        experiment.writer_control()
    results = {}

    try:
        for i, results in enumerate(iter(alg)):

            if manager['is_done']:
                break

            experiment.save_model_results(copy.deepcopy(results), alg, i)

        if rank == 0:
            logger.info(f"Training is done, Worker terminates.")

    except KeyboardInterrupt as e:

        if rank == 0:
            logger.warning(f"KeyboardInterrupt: Training was interrupted, Worker terminates.")
            logger.debug(f"KeyboardInterrupt: {e}")
            training_closure(rank, world_size, experiment, alg)

    except Exception as e:

        tb = traceback.format_exc()

        llm = get_beam_llm() if experiment.llm is None else experiment.llm

        if llm is not None:
            explain = llm.explain_traceback(tb)
            logger.error(f"LLM Message: {explain}")

        if rank == 0:

            logger.error(f"Exception: {e}")
            logger.error(f"Exception: {tb}")
            logger.error(f"Exception: Training was interrupted, Worker terminates, but checkpoint will be saved.")
            training_closure(rank, world_size, experiment, alg)

        if not is_notebook():
            raise e

    if hasattr(manager, 'cleanup'):
        manager.cleanup()
    experiment.writer_cleanup()
    manager['is_done'] = True

    if rank == 0:
        return alg, results
    return None, None
