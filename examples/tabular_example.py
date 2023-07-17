from examples.example_utils import add_beam_to_path
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

    hparams = TabularHparams(identifier='spline', path_to_data='/dsi/shared/elads/elads/data/tabular/dataset/data/',
                             path_to_results='/dsi/shared/elads/elads/data/tabular/results/', dataset_name='covtype',
                             copy_code=False, stop_at=0.98, parallel=4)

    exp = Experiment(hparams)

    dataset = TabularDataset(exp.hparams)
    net = TabularTransformer(exp.hparams, dataset.n_classes, dataset.n_tokens, dataset.cat_mask)
    alg = DeepTabularAlg(exp.hparams, networks=net)

    alg = exp.fit(Alg=alg, Dataset=dataset)
