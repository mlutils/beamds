import os
import sys

def add_beam_to_path():
    beam_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    sys.path.insert(0, beam_path)


add_beam_to_path()

import torch
from torch import distributions
import torchvision
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd

from src.beam import beam_arguments, Experiment
from src.beam import beam_logger as logger
from src.beam import beam_path


def bundle_example():
    from src.beam.tabular import TabularDataset, TabularTransformer, TabularHparams, DeepTabularAlg

    kwargs_base = dict(algorithm='debug_reporter',
                       # path_to_data='/dsi/shared/elads/elads/data/tabular/dataset/data/',
                       path_to_data='/home/dsi/elads/data/tabular/data/',
                       path_to_results='/dsi/shared/elads/elads/data/tabular/results/',
                       copy_code=False, dynamic_masking=False, comet=False, tensorboard=True, n_epochs=2,
                       stop_at=0.98, parallel=1, device=1, n_quantiles=6, label_smoothing=.2)

    kwargs_all = {}

    k = 'california_housing'
    kwargs_all[k] = dict(batch_size=128)

    logger.info(f"Starting a new experiment with dataset: {k}")
    hparams = {**kwargs_base}
    hparams.update(kwargs_all[k])
    hparams['dataset_name'] = k
    hparams['identifier'] = k
    hparams = TabularHparams(hparams)

    # exp = Experiment(hparams)
    # dataset = TabularDataset(hparams)
    net = TabularTransformer(hparams, 10, [4, 4, 4], [0, 0, 1])
    alg = DeepTabularAlg(hparams, networks=net)

    from src.beam.auto import AutoBeam

    ab = AutoBeam(alg)

    # print(ab.module_spec)
    # print(ab.module_walk)
    # print(ab.module_dependencies)
    # print(ab.requirements)
    # print(ab.top_levels)
    # print(ab.module_to_tar('/tmp/beam.tar.gz'))

    tar_path = '/tmp/beam.tar.gz'
    beam_path(tar_path).unlink(missing_ok=True)
    ab.module_to_tar(tar_path)


if __name__ == '__main__':

    bundle_example()