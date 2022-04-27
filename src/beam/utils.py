import os, torch, copy, sys
from collections import defaultdict
import numpy as np
import torch.distributed as dist
from fnmatch import fnmatch, filter
from tqdm import *
import random
import torch
import pandas as pd
import multiprocessing as mp
from .model import BeamOptimizer

from loguru import logger
logger.remove(handler_id=0)
logger.add(sys.stdout, colorize=True, format='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>')


def process_async(func, args, mp_context='spawn', num_workers=10):

    ctx = mp.get_context(mp_context)
    with ctx.Pool(num_workers) as pool:
        res = [pool.apply_async(func, (args,)) for arg in args]
        results = []
        for r in tqdm_beam(res):
            results.append(r.get())

    return results


def check_type(x):
    '''
    return one of the types
    numeric, string, array, tensor

    array type

    '''
    if np.isscalar(x):
        if type(x) is int or type(x) is float:
            return 'numeric'
        elif type(x) is str:
            return 'string'
        else:
            return 'other'
    elif pd.isna(x):
        return 'none'
    else:
        if isinstance(x, torch.Tensor):
            return 'tensor'
        else:
            return 'array'


def include_patterns(*patterns):
    """Factory function that can be used with copytree() ignore parameter.
    Arguments define a sequence of glob-style patterns
    that are used to specify what files to NOT ignore.
    Creates and returns a function that determines this for each directory
    in the file hierarchy rooted at the source directory when used with
    shutil.copytree().
    """
    def _ignore_patterns(path, names):
        keep = set(name for pattern in patterns
                            for name in filter(names, pattern))
        ignore = set(name for name in names
                        if name not in keep and not os.path.isdir(os.path.join(path, name)))
        return ignore

    return _ignore_patterns


def is_notebook():

    return '_' in os.environ and 'jupyter' in os.environ['_']


def setup(rank, world_size):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup(rank, world_size):

    dist.destroy_process_group()


def set_seed(seed=-1, constant=0, increment=False, deterministic=False):

    '''
    :param seed: set -1 to avoid change, set 0 to randomly select seed, set [1, 2**32) to get new seed
    :param constant: a constant to be added to the seed
    :param increment: whether to generate incremental seeds
    :param deterministic: whether to set torch to be deterministic
    :return: None
    '''

    if 'cnt' not in set_seed.__dict__:
        set_seed.cnt = 0
    set_seed.cnt += 1

    if increment:
        constant += set_seed.cnt

    if seed == 0:
        seed = np.random.randint(1, 2**32-constant) + constant

    if seed > 0:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def to_device(data, device='cuda'):

    if type(data) is dict:
        return {k: to_device(v, device) for k, v in data.items()}
    elif type(data) is list:
        return [to_device(s, device) for s in data]
    elif issubclass(type(data), torch.Tensor):
        return data.to(device)
    else:
        return data

def finite_iterations(iterator, n):

    for i, out in enumerate(iterator):

        if i+1 < n:
            yield out
        else:
            return out

def tqdm_beam(x, *args, enable=True, **argv):

    if not enable:
        return x
    else:
        my_tqdm = tqdm_notebook if is_notebook() else tqdm
        return my_tqdm(x, *args, **argv)


def reset_networks_and_optimizers(networks=None, optimizers=None):


    if networks is not None:
        net_iter = networks.keys() if issubclass(type(networks), dict) else range(len(networks))
        for i in net_iter:
            for n, m in networks[i].named_modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()

    if optimizers is not None:
        opt_iter = optimizers.keys() if issubclass(type(optimizers), dict) else range(len(optimizers))
        for i in opt_iter:
            opt = optimizers[i]

            if type(opt) is BeamOptimizer:
                opt.reset()
            else:
                opt.state = defaultdict(dict)

