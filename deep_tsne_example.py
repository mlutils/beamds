#!/usr/bin/env python
# coding: utf-8

# In[1]:
import collections
import math

import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from src.beam import beam_arguments, Experiment, beam_algorithm_generator
from src.beam import UniversalDataset, UniversalBatchSampler, get_beam_parser
from src.beam import Algorithm
from src.beam import LinearNet
from src.beam import DataTensor, PackedFolds
from functools import partial
import math


# In[2]:

def pairwise_distance(a, b, p=2):

    a = a.unsqueeze(0)
    b = b.unsqueeze(1)

    r = a - b
    if p == 1:
        r = torch.abs(r).sum(dim=-1)
    elif p == 2:
        r = torch.sqrt(torch.pow(r, 2).sum(dim=-1))
    else:
        raise NotImplementedError

    return r


class MNISTDataset(UniversalDataset):

    def __init__(self, hparams):

        path = hparams.path_to_data
        seed = hparams.split_dataset_seed

        super().__init__()
        dataset_train = torchvision.datasets.MNIST(root=path, train=True, transform=torchvision.transforms.ToTensor(), download=True)
        dataset_test = torchvision.datasets.MNIST(root=path, train=False, transform=torchvision.transforms.ToTensor(), download=True)

        self.data = PackedFolds({'train': dataset_train.data, 'test': dataset_test.data})
        self.labels = PackedFolds({'train': dataset_train.targets, 'test': dataset_test.targets})
        self.split(validation=.2, test=self.labels['test'].index, seed=seed)

    def getitem(self, index):
        return {'x': self.data[index].float() / 255, 'y': self.labels[index]}


class DeepTSNE(Algorithm):

    def __init__(self, hparams):

        # choose your network
        net = LinearNet(784, 256, hparams.emb_size, 4)
        super().__init__(hparams, networks=net)

        self.pdist = partial(pairwise_distance, p=hparams.p_norm)
        self.reduction = hparams.reduction
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizers['net'].dense, gamma=0.99)

    # def early_stopping(self, results=None, epoch=None, **kwargs):
    #     acc = np.mean(results['validation']['scalar']['acc'])
    #     return acc > self.stop_at

    def postprocess_epoch(self, sample=None, results=None, epoch=None, subset=None, training=True, **kwargs):

        if not training:
            # self.scheduler.step()
            results['scalar'][f'lr'] = self.optimizers['net'].dense.param_groups[0]['lr']

        return results

    def iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x, y = sample['x'], sample['y']

        x = x.view(len(x), -1)
        lx = x.shape[-1]

        net = self.networks['net']
        opt = self.optimizers['net']

        dx = self.pdist(x, x)
        # rx = torch.norm(x, p=self.hparams.p_norm, dim=-1, keepdim=True)
        # dx / rx

        z = net(x)

        lz = z.shape[-1]
        dz = self.pdist(z, z)

        # w = 1 / (dx + 1e-3)
        # w

        loss_dist = F.smooth_l1_loss(dz / lz, dx / lx, reduction='none')
        loss_dist = loss_dist.sum(dim=-1)

        mu = z.mean(dim=0)
        sig2 = z.var(dim=0)

        loss_reg = torch.pow(mu, 2) + sig2 - .5 * torch.log(sig2)
        loss_reg = loss_reg.sum()

        if self.reduction == 'sum':
            loss_dist = loss_dist.sum()
        else:
            loss_dist = loss_dist.mean()

        loss = loss_dist + self.hparams.reg_weight * loss_reg

        opt.apply(loss, training=training)

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['loss_reg'].append(float(loss_reg / self.hparams.emb_size))
        results['scalar']['loss_dist'].append(float(loss_dist))
        results['scalar']['mu'].append(float(mu.mean()))
        results['scalar']['sig2'].append(float(sig2.mean()))

        return results

    def inference(self, sample=None, results=None, subset=None, predicting=True, **kwargs):

        if predicting:
            x = sample
        else:
            x, y = sample['x'], sample['y']

        x = x.view(len(x), -1)
        net = self.networks['net']

        z = net(x)

        if predicting:
            transforms = z
        else:
            transforms = {'z': z, 'y': y}

        return transforms, results

    def postprocess_inference(self, sample=None, results=None, subset=None, predicting=True, **kwargs):

        return results


def get_deep_tsne_parser():

    parser = get_beam_parser()
    parser.add_argument('--emb-size', type=int, default=2, help='Size of embedding dimension')
    parser.add_argument('--p-norm', type=int, default=1, help='The norm degree')
    parser.add_argument('--reg-weight', type=float, default=200., help='Regularization weight factor')
    parser.add_argument('--reduction', type=str, default='sum', help='The reduction to apply')

    return parser


# ## Training

if __name__ == '__main__':

    # here you put all actions which are performed only once before initializing the workers
    # for example, setting running arguments and experiment:

    path_to_data = '/home/shared/data//dataset/mnist'
    root_dir = '/home/shared/data/results'

    args = beam_arguments(get_deep_tsne_parser(),
        f"--project-name=deep_tsne_mnist --root-dir={root_dir} --algorithm=DeepTSNE --device=cpu",
        "--epoch-length=200000 --n-epochs=10 --parallel=1", path_to_data=path_to_data)

    experiment = Experiment(args)
    experiment.fit(DeepTSNE, MNISTDataset)

