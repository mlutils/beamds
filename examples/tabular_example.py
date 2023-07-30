# from examples.example_utils import add_beam_to_path
from example_utils import add_beam_to_path
add_beam_to_path()

import torch
from torch import distributions
import torchvision
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import os

from src.beam import beam_arguments, Experiment
from src.beam import UniversalDataset, UniversalBatchSampler
from src.beam import Algorithm, PackedFolds, as_numpy
from src.beam import DataTensor, BeamOptimizer
from src.beam.data import BeamData
from src.beam.tabular import TabularDataset, TabularTransformer, TabularHparams, DeepTabularAlg

if __name__ == '__main__':


    kwargs_base = dict(algorithm='run_all_nq_10',
                       path_to_data='/dsi/shared/elads/elads/data/tabular/dataset/data/',
                       path_to_results='/dsi/shared/elads/elads/data/tabular/results/',
                       copy_code=False, tensorboard=False, stop_at=0.98, parallel=1, device=3, n_quantiles=10)

    kwargs_all = {}

    # kwargs_all['california_housing'] = dict(batch_size=256)
    # kwargs_all['adult'] = dict(batch_size=256)
    # kwargs_all['helena'] = dict(batch_size=512)
    # kwargs_all['jannis'] = dict(batch_size=512)
    # kwargs_all['higgs_small'] = dict(batch_size=512)
    # kwargs_all['aloi'] = dict(batch_size=512)
    kwargs_all['year'] = dict(batch_size=1024)
    kwargs_all['covtype'] = dict(batch_size=1024)

    for k in kwargs_all.keys():
        hparams = {**kwargs_base}
        hparams.update(kwargs_all[k])
        hparams['dataset_name'] = k
        hparams['identifier'] = k
        hparams = TabularHparams(hparams)

        exp = Experiment(hparams)

        dataset = TabularDataset(exp.hparams)
        net = TabularTransformer(exp.hparams, dataset.n_classes, dataset.n_tokens, dataset.cat_mask)
        alg = DeepTabularAlg(exp.hparams, networks=net)

        alg = exp.fit(Alg=alg, Dataset=dataset)
