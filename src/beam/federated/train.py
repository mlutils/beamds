import copy
import torch.multiprocessing as mp
import inspect
import traceback
import time
import os

from ..utils import (set_seed, is_notebook,)
from ..path import beam_path
from ..logger import beam_logger as logger
from ..config import get_beam_llm, BeamConfig
from .resource import federated_executor


def beam_algorithm_generator(experiment, alg, dataset=None, alg_args=None, alg_kwargs=None,
                             dataset_args=None, dataset_kwargs=None, store_init_args=True, rank=0):

    if alg_args is None:
        alg_args = tuple()
    if alg_kwargs is None:
        alg_kwargs = dict()
    if dataset_args is None:
        dataset_args = tuple()
    if dataset_kwargs is None:
        dataset_kwargs = dict()

    if dataset is not None and not isinstance(dataset, dict):
        datasets = {'dataset': dataset}
    else:
        datasets = dataset

    if datasets is not None:
        for k, v in datasets.items():
            if inspect.isclass(v):
                datasets[k] = v(experiment.hparams, *dataset_args, **dataset_kwargs)
            elif inspect.isfunction(v):
                datasets[k] = v(experiment.hparams, *dataset_args, **dataset_kwargs)

    if inspect.isclass(alg):
        store_init_path = None
        if store_init_args and rank == 0:
            store_init_path = experiment.store_init_path

        alg = alg(experiment.hparams, experiment=experiment, *alg_args, store_init_path=store_init_path, **alg_kwargs)
        # if a new algorithm is generated, we clean the tensorboard writer. If the reload option is True,
        # the algorithm will fix the epoch number s.t. tensorboard graphs will not overlap
        experiment.writer_cleanup()
    else:
        alg.experiment = experiment

    if datasets is not None:
        alg.load_datasets(datasets)

    return alg


def training_closure(rank, world_size, experiment, alg, **kwargs):

    if not rank:
        alg.training_closure(**kwargs)
        checkpoint_file = experiment.checkpoints_dir.joinpath(f'checkpoint_{alg.epoch + 1:06d}')
        alg.save_checkpoint(checkpoint_file)


def beam_train_worker(experiment, alg, manager=None, dataset=None, alg_args=None, alg_kwargs=None,
                      dataset_args=None, dataset_kwargs=None, store_init_args=True, **kwargs):

    if manager is None:
        rank = 0
        world_size = 1
        manager = {}
    else:
        rank = manager.rank
        world_size = manager.world_size

    if rank == 0:
        manager['is_done'] = False

    experiment.set_rank(rank, world_size)
    set_seed(seed=experiment.hparams.seed, constant=rank + 1, increment=False,
             deterministic=experiment.hparams.deterministic)

    alg = beam_algorithm_generator(experiment, alg, dataset=dataset, alg_args=alg_args, alg_kwargs=alg_kwargs,
                                   dataset_args=dataset_args, dataset_kwargs=dataset_kwargs,
                                   store_init_args=store_init_args, rank=rank)

    experiment.writer_control(enable=not (bool(rank)))
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

    return alg, results


# def beam_train(experiment, alg, dataset=None, alg_args=None, alg_kwargs=None,
#
#     workers = federated_executor()