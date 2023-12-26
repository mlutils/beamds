import copy
import torch.multiprocessing as mp
import inspect
import traceback
import time
import os

from ..utils import (set_seed, is_notebook,)
from ..path import beam_path
from ..logger import beam_logger as logger
from ..config import get_beam_llm, BeamHparams


done_training = mp.Event()


def setup_distributed(rank, world_size, port='7463', backend='nccl', framework='ddp'):

    os.environ['MASTER_ADDR'] = 'localhost'
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

        import deepspeed
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
    elif framework == 'horovbod':
        import horovod.torch as hvd
        hvd.shutdown()
    else:
        raise ValueError(f"Unknown distributed framework: {framework}")


def gen_hparams_string(experiment_path):
    experiment_path = beam_path(experiment_path)
    tensorboard_hparams = BeamHparams.from_path(experiment_path.joinpath('args.pkl'))
    tensorboard_hparams_keys = tensorboard_hparams.model_parameter + tensorboard_hparams.tune_parameter
    return '/'.join([f"{k}_{tensorboard_hparams[k]}" for k in tensorboard_hparams_keys])


def path_depth(path):

    if isinstance(path, str):
        path = beam_path(path)

    return len(str(path.resolve()).split(os.sep))


def beam_algorithm_generator(experiment, alg, dataset=None, alg_args=None, alg_kwargs=None,
                             dataset_args=None, dataset_kwargs=None):

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

        alg = alg(experiment.hparams, experiment=experiment, *alg_args, **alg_kwargs)
        # if a new algorithm is generated, we clean the tensorboard writer. If the reload option is True,
        # the algorithm will fix the epoch number s.t. tensorboard graphs will not overlap
        experiment.writer_cleanup()
    else:
        alg.experiment = experiment

    if datasets is not None:
        alg.load_datasets(datasets)

    return alg


def default_runner(rank, world_size, experiment, algorithm_generator, *args, tensorboard_arguments=None, **kwargs):
    alg = algorithm_generator(*args, **kwargs)

    experiment.writer_control(enable=not (bool(rank)))
    results = {}

    t0 = time.time()

    try:
        for i, results in enumerate(iter(alg)):

            if done_training.is_set():
                break

            total_time = time.time() - t0
            estimated_time = total_time * (alg.n_epochs - i - 1) / (i + 1)

            experiment.save_model_results(copy.deepcopy(results), alg, i,
                                          print_results=experiment.hparams.print_results,
                                          visualize_results=experiment.hparams.visualize_results,
                                          store_results=experiment.hparams.store_results, store_networks=experiment.hparams.store_networks,
                                          visualize_weights=experiment.hparams.visualize_weights,
                                          argv=tensorboard_arguments, total_time=total_time, estimated_time=estimated_time)

    except KeyboardInterrupt as e:

        tb = traceback.format_exc()
        logger.warning(f"KeyboardInterrupt: Training was interrupted, Worker terminates.")
        logger.debug(f"KeyboardInterrupt: {e}")
        logger.debug(f"KeyboardInterrupt: {tb}")

        if rank == 0:
            checkpoint_file = experiment.checkpoints_dir.joinpath(f'checkpoint_{alg.epoch + 1:06d}')
            alg.save_checkpoint(checkpoint_file)

    except Exception as e:

        tb = traceback.format_exc()

        llm = get_beam_llm() if experiment.llm is None else experiment.llm

        if llm is not None:
            explain = llm.explain_traceback(tb)
            logger.error(f"LLM Message: {explain}")

        if not is_notebook():
            raise e

        logger.error(f"Exception: Training was interrupted, Worker terminates, but checkpoint will be saved.")
        logger.error(f"Exception: {e}")
        logger.error(f"Exception: {tb}")

        if rank == 0:
            checkpoint_file = experiment.checkpoints_dir.joinpath(f'checkpoint_{alg.epoch + 1:06d}')
            alg.save_checkpoint(checkpoint_file)

    experiment.writer_cleanup()

    if world_size > 1:
        done_training.set()

    if world_size == 1:
        return alg, results


def run_worker(rank, world_size, results_queue, job, experiment, *args, **kwargs):

    logger.info(f"Worker: {rank + 1}/{world_size} is running...")

    if world_size > 1:
        backend = experiment.hparams.distributed_backend
        if backend is None:
            backend = 'nccl' if experiment.device.type == 'cuda' else 'gloo'

        setup_distributed(rank, world_size, port=experiment.hparams.mp_port, backend=backend,
                          framework=experiment.distributed_training_framework)

    experiment.set_rank(rank, world_size)
    set_seed(seed=experiment.hparams.seed, constant=rank+1, increment=False, deterministic=experiment.hparams.deterministic)

    res = job(rank, world_size, experiment, *args, **kwargs)

    if world_size > 1:

        cleanup(rank, world_size, experiment.distributed_training_framework)
        results_queue.put({'rank': rank, 'results': res})

        done_training.wait()

    else:
        return res