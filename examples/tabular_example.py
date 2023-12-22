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

    device = 3

    hparams = TabularHparams(dataset_name='covtype', algorithm='dynamic_masking', path_to_data='/dsi/shared/elads/elads/data/tabular/dataset/data/',
                             path_to_results='/dsi/shared/elads/elads/data/tabular/results/',
                             copy_code=False, stop_at=0.98, parallel=1, mask_rate=.15, dynamic_delta=0.00,
                             maximal_mask_rate=.2, minimal_mask_rate=.1,
                             dynamic_masking=True, store_data_on_device=True, device=device)

    # hparams = TabularHparams(dataset_name='jannis', algorithm='deep_tabnet',
    #                          path_to_data='/dsi/shared/elads/elads/data/tabular/dataset/data/',
    #                          path_to_results='/dsi/shared/elads/elads/data/tabular/results/', batch_size=256,
    #                          copy_code=False, stop_at=0.98, parallel=1, device=device, n_quantiles=6)

    # hparams = TabularHparams(dataset_name='yahoo', algorithm='deep_tabnet',
    #                          path_to_data='/dsi/shared/elads/elads/data/tabular/dataset/data/',
    #                          path_to_results='/dsi/shared/elads/elads/data/tabular/results/',
    #                          copy_code=False, stop_at=0.98, parallel=1, device=device, n_quantiles=10)

    # hparams = TabularHparams(dataset_name='aloi', algorithm='deep_tabnet',
    #                          path_to_data='/dsi/shared/elads/elads/data/tabular/dataset/data/',
    #                          path_to_results='/dsi/shared/elads/elads/data/tabular/results/',
    #                          copy_code=False, stop_at=0.98, parallel=1, device=device, n_quantiles=6)

    # hparams = TabularHparams(algorithm='deep_tabnet', path_to_data='/dsi/shared/elads/elads/data/tabular/dataset/data/',
    #                          path_to_results='/dsi/shared/elads/elads/data/tabular/results/', dataset_name='year',
    #                          copy_code=False, stop_at=0.98, parallel=1, device=device, n_quantiles=6)

    # hparams = TabularHparams(algorithm='deep_tabnet', path_to_data='/dsi/shared/elads/elads/data/tabular/dataset/data/',
    #                          root_dir='/dsi/shared/elads/elads/data/tabular/results/', dataset_name='adult',
    #                          copy_code=False, stop_at=0.98, parallel=1, batch_size=64, mask_rate=.15, n_rules=64,
    #                          n_transformer_head=4, emb_dim=128, n_encoder_layers=2, n_decoder_layers=4, n_quantiles=6,)

    # hparams = TabularHparams(algorithm='deep_tabnet', path_to_data='/dsi/shared/elads/elads/data/tabular/dataset/data/',
    #                          path_to_results='/dsi/shared/elads/elads/data/tabular/results/', dataset_name='covtype',
    #                          copy_code=False, stop_at=0.98, parallel=1)

    hparams.identifier = hparams.dataset_name
    exp = Experiment(hparams)

    dataset = TabularDataset(exp.hparams)
    net = TabularTransformer(exp.hparams, dataset.n_classes, dataset.n_tokens, dataset.cat_mask)
    alg = DeepTabularAlg(exp.hparams, networks=net)

    alg = exp.fit(Alg=alg, Dataset=dataset)
