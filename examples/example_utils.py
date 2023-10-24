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


def write_bundle_tabular(path):

    from src.beam import beam_arguments, Experiment
    from src.beam import beam_logger as logger
    from src.beam import beam_path

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
    # print(ab.module_to_tar(path))
    print(ab.private_modules)

    # autobeam_path = '/tmp/autobeam'
    # beam_path(autobeam_path).rmtree()

    # ab.module_to_tar(tar_path)
    # AutoBeam.to_bundle(alg, autobeam_path)


def write_bundle_cifar(path):

    from src.beam import beam_arguments, beam_path
    from cifar10_example import CIFAR10Algorithm
    from src.beam.auto import AutoBeam

    path.rmtree()

    args = beam_arguments(
        f"--project-name=cifar10 --algorithm=CIFAR10Algorithm --device=1 --half --lr-d=1e-4 --batch-size=512",
        "--n-epochs=50 --epoch-length-train=50000 --epoch-length-eval=10000 --clip=0 --parallel=1 --accumulate=1 --no-deterministic",
        "--weight-decay=.00256 --momentum=0.9 --beta2=0.999 --temperature=1 --objective=acc --scheduler=one_cycle",
        dropout=.0, activation='gelu', channels=512, label_smoothing=.2, padding=4, scale_down=.7,
        scale_up=1.4, ratio_down=.7, ratio_up=1.4)

    alg = CIFAR10Algorithm(args)
    # ab = AutoBeam(alg)

    # print(ab.requirements)
    # print(ab.private_modules)
    # print(ab.import_statement)
    #
    # ab.modules_to_tar(path.joinpath('modules.tar.gz'))
    AutoBeam.to_bundle(alg, path)
    print('done writing bundle')


def load_bundle(path):
    from src.beam.auto import AutoBeam
    alg = AutoBeam.from_path(path)
    print(alg)
    print(alg.hparams)
    print('done loading bundle')
    return alg


def test_data_apply():



    M = 40000

    nel = 100
    k1 = 20000
    k2 = 20

    def gen_coo_vectors(k):
        r = []
        c = []
        v = []

        for i in range(k):
            r.append(i * torch.ones(nel, dtype=torch.int64))
            c.append(torch.randint(M, size=(nel,)))
            v.append(torch.randn(nel))

        return torch.sparse_coo_tensor(torch.stack([torch.cat(r), torch.cat(c)]), torch.cat(v), size=(k, M))

    s1 = gen_coo_vectors(k1)
    s2 = gen_coo_vectors(k2)
    s3 = gen_coo_vectors(k2)

    from src.beam.similarity import SparseSimilarity
    sparse_sim = SparseSimilarity(similarity='cosine', format='coo', vec_size=10000, device='cuda', k=10)



if __name__ == '__main__':

    from src.beam import beam_arguments, beam_path

    # path = '/home/dsi/elads/sandbox/cifar10_bundle'
    # path = beam_path(path)
    #
    # # write_bundle_cifar(path)
    #
    # alg = load_bundle(path)

    from src.beam.config import BeamHparams
    from src.beam.tabular import TabularHparams

    hparams = BeamHparams(identifier='test', project_name='test', algorithm='test', device=1)

    # hparams = TabularHparams(identifier='test', project_name='test', algorithm='test', device=1)
    hparams = TabularHparams(hparams)
    print(hparams)
    print('done')