import copy
import torch.multiprocessing as mp
import inspect
import traceback
import os

from ..utils import (set_seed, is_notebook, beam_device)
from ..path import beam_path
from ..logging import beam_logger as logger
from ..config import get_beam_llm, BeamConfig
from ..importer import lazy_importer as lzi


done_training = mp.Event()


def setup_distributed(rank, world_size, port='7463', backend='nccl', framework='ddp', master_addr='localhost'):

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = port
    logger.info(f"Initializing distributed training with backend={backend} and framework={framework}")
    if framework == 'ddp':
        # initialize the process group
        import torch.distributed as dist
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    elif framework == 'deepspeed':

        # make sure that mpi path is in the path variable
        # os.environ['PATH'] = f"/usr/local/mpi/bin:{os.environ['PATH']}"
        # os.environ['LD_LIBRARY_PATH'] = f"/usr/local/mpi/lib:{os.environ['PATH']}"

        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['OMPI_COMM_WORLD_SIZE'] = str(world_size)
        os.environ['OMPI_COMM_WORLD_RANK'] = str(rank)

        import deepspeed
        # deepspeed.init_distributed(dist_backend=backend, auto_mpi_discovery=backend == 'mpi',
        #                            rank=rank, world_size=world_size, distributed_port=port)
        deepspeed.init_distributed(dist_backend=backend, auto_mpi_discovery=False,
                                   rank=rank, world_size=world_size, distributed_port=port)

    else:
        raise ValueError(f"Unknown distributed framework: {framework}")


def cleanup(rank, world_size, framework='ddp'):

    if framework == 'ddp':
        import torch.distributed as dist
        dist.destroy_process_group()
    elif framework == 'deepspeed':
        pass
    elif framework == 'horovod':
        import horovod.torch as hvd
        hvd.shutdown()
    else:
        raise ValueError(f"Unknown distributed framework: {framework}")


def gen_hparams_string(experiment_path):
    experiment_path = beam_path(experiment_path)
    tensorboard_hparams = BeamConfig.from_path(experiment_path.joinpath('args.pkl'))
    tensorboard_hparams_keys = tensorboard_hparams.model_parameter + tensorboard_hparams.tune_parameter
    return '/'.join([f"{k}_{tensorboard_hparams[k]}" for k in tensorboard_hparams_keys])


def path_depth(path):

    if isinstance(path, str):
        path = beam_path(path)

    return len(str(path.resolve()).split(os.sep))


def nn_algorithm_generator(experiment, alg, dataset=None, alg_args=None, alg_kwargs=None, dataset_args=None,
                           dataset_kwargs=None, rank=0, **kwargs):

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
        if rank == 0:
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


def simple_algorithm_generator(experiment, alg, dataset=None, alg_args=None, alg_kwargs=None, dataset_args=None,
                               dataset_kwargs=None, rank=0, **kwargs):

    if alg_args is None:
        alg_args = tuple()
    if alg_kwargs is None:
        alg_kwargs = dict()
    if dataset_args is None:
        dataset_args = tuple()
    if dataset_kwargs is None:
        dataset_kwargs = dict()

    if dataset is not None:
        if inspect.isclass(dataset):
            dataset = dataset(experiment.hparams, *dataset_args, **dataset_kwargs)
        elif inspect.isfunction(dataset):
            dataset = dataset(experiment.hparams, *dataset_args, **dataset_kwargs)

    if inspect.isclass(alg):
        store_init_path = None
        if rank == 0:
            store_init_path = experiment.store_init_path

        alg = alg(experiment.hparams, experiment=experiment, *alg_args, store_init_path=store_init_path, **alg_kwargs)
        # if a new algorithm is generated, we clean the tensorboard writer. If the reload option is True,
        # the algorithm will fix the epoch number s.t. tensorboard graphs will not overlap
        experiment.writer_cleanup()
    else:
        alg.experiment = experiment

    return alg, dataset


def training_closure(rank, world_size, experiment, alg, *args, **kwargs):

    if not rank:
        alg.training_closure(*args, **kwargs)
        checkpoint_file = experiment.checkpoints_dir.joinpath(f'checkpoint_{alg.epoch + 1:06d}')
        alg.save_checkpoint(checkpoint_file)


def default_runner(rank, world_size, experiment, algorithm_generator, *args, tensorboard_arguments=None, **kwargs):

    alg = algorithm_generator(*args, rank=rank, **kwargs)

    if rank == 0:
        experiment.writer_control()
    results = {}

    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()

    try:
        for i, results in enumerate(iter(alg)):

            if done_training.is_set():
                break

            experiment.save_model_results(copy.deepcopy(results), alg, i, argv=tensorboard_arguments)

            if world_size > 1:
                logger.info(f"Worker {rank + 1}/{world_size} finished epoch {i + 1}/{alg.n_epochs}. Waiting for others.")
                dist.barrier()
                logger.info(f"Worker {rank + 1}/{world_size} is continuing.")

        if rank == 0:
            logger.info(f"Training is done, Worker terminates.")

    except KeyboardInterrupt as e:

        if rank == 0:
            logger.warning(f"KeyboardInterrupt: Training was interrupted, Worker terminates.")
            logger.debug(f"KeyboardInterrupt: {e}")
            training_closure(rank, world_size, experiment, alg, *args, **kwargs)

    except Exception as e:

        if lzi.optuna is not None:
            from optuna.exceptions import TrialPruned
            if isinstance(e, TrialPruned):
                logger.warning(f"TrialPruned: Training was interrupted, Worker terminates.")
                logger.debug(f"TrialPruned: {e}")
                raise e

        tb = traceback.format_exc()

        llm = get_beam_llm() if experiment.llm is None else experiment.llm

        if llm is not None:
            explain = llm.explain_traceback(tb)
            logger.error(f"LLM Message: {explain}")

        if rank == 0:

            logger.error(f"Exception: {e}")
            logger.error(f"Exception: {tb}")
            logger.error(f"Exception: Training was interrupted, Worker terminates, but checkpoint will be saved.")
            training_closure(rank, world_size, experiment, alg, *args, **kwargs)

        if not is_notebook():
            raise e

    experiment.writer_cleanup()

    if world_size > 1:
        done_training.set()

    if world_size == 1:
        return alg, results


def simple_runner(rank, world_size, experiment, algorithm_generator, *args, **kwargs):

    alg, dataset = algorithm_generator(*args, rank=rank, **kwargs)

    assert rank == 0, "Simple runner is only supported for single process training."

    results = None

    try:

        results = alg.fit(dataset)
        logger.info(f"Training is done, Worker terminates.")

    except KeyboardInterrupt as e:

        logger.warning(f"KeyboardInterrupt: Training was interrupted, Worker terminates.")
        logger.debug(f"KeyboardInterrupt: {e}")
        training_closure(rank, world_size, experiment, alg, *args, **kwargs)

    except Exception as e:

        if lzi.optuna is not None:
            from optuna.exceptions import TrialPruned
            if isinstance(e, TrialPruned):
                logger.warning(f"TrialPruned: Training was interrupted, Worker terminates.")
                logger.debug(f"TrialPruned: {traceback.format_exc()}")
                raise e

        tb = traceback.format_exc()

        llm = get_beam_llm() if experiment.llm is None else experiment.llm

        if llm is not None:
            explain = llm.explain_traceback(tb)
            logger.error(f"LLM Message: {explain}")

        logger.error(f"Exception: {e}")
        logger.error(f"Exception: {tb}")
        logger.error(f"Exception: Training was interrupted, Worker terminates, but checkpoint will be saved.")
        training_closure(rank, world_size, experiment, alg, *args, **kwargs)

        if not is_notebook():
            raise e

    experiment.writer_cleanup()

    # return alg, results
    return experiment.experiment_dir, results


def run_worker(rank, world_size, results_queue_or_kwargs, job, experiment, *args, **kwargs):

    logger.info(f"Worker: {rank + 1}/{world_size} is running...")

    if world_size > 1:
        backend = experiment.hparams.distributed_backend
        if backend is None:
            backend = 'nccl' if experiment.device.type == 'cuda' else 'gloo'

        setup_distributed(rank, world_size, port=experiment.hparams.mp_port, backend=backend,
                          framework=experiment.distributed_training_framework)

    if world_size > 1 and backend == 'mpi':
        results_queue = None
        kwargs = results_queue_or_kwargs
    else:
        results_queue = results_queue_or_kwargs

    experiment.set_rank(rank, world_size)
    set_seed(seed=experiment.hparams.seed, constant=rank+1, increment=False, deterministic=experiment.hparams.deterministic)

    res = job(rank, world_size, experiment, *args, **kwargs)

    if world_size > 1:

        cleanup(rank, world_size, experiment.distributed_training_framework)

        if results_queue is not None:
            results_queue.put({'rank': rank, 'results': res})

        elif backend == 'mpi' and rank != 0:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            comm.send({'rank': rank, 'results': res}, dest=0)

        done_training.wait()

    else:
        return res


def build_device_list(hparams):

    device = beam_device(hparams.device)
    if 'cpu' in device.type:
        return []

    device_list = hparams.get('device_list', None)
    if device_list is not None:
        device_list = [beam_device(di) for di in device_list]
    else:
        device_list = [beam_device(di + device.index) for di in range(hparams.n_gpus)]

    return device_list

