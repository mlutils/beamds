#!/usr/bin/env python
# coding: utf-8

# In[1]:
from examples.example_utils import add_beam_to_path
add_beam_to_path()

import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from src.beam import beam_arguments, Experiment, beam_algorithm_generator
from src.beam import UniversalDataset, UniversalBatchSampler, basic_beam_parser
from src.beam import Algorithm
from src.beam import LinearNet, as_numpy
from src.beam import DataTensor, PackedFolds
from functools import partial
import math
import matplotlib.pyplot as plt

# In[2]:

def pairwise_distance(a, b, p=2):

    a = a.unsqueeze(0)
    b = b.unsqueeze(1)

    r = a - b
    if p == 1:
        r = torch.abs(r).sum(dim=-1)
    elif p == 2:
        # r = torch.sqrt(torch.pow(r, 2).sum(dim=-1))
        r = torch.pow(r, 2).sum(dim=-1)
    else:
        raise NotImplementedError

    return r


class DeepTSNENet(nn.Module):
    def __init__(self, net, eps=1e-05, momentum=0.1):
        super().__init__()
        self.net = net
        # self.norm = nn.LazyBatchNorm1d(eps=eps, momentum=momentum)

    def forward(self, x):

        z = self.net(x)
        # ztag = self.norm(z)
        ztag = z

        return z, ztag


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
        net = LinearNet(784, 256, hparams.emb_size, 6, activation='GELU')
        net = DeepTSNENet(net)
        super().__init__(hparams, networks=net)

        self.pdist = partial(pairwise_distance, p=hparams.p_norm)
        self.reduction = hparams.reduction
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizers['net'].dense,
                                                                    factor=1/math.sqrt(10),
                                                                    patience=2)

    # def early_stopping(self, results=None, epoch=None, **kwargs):
    #     acc = np.mean(results['validation']['scalar']['acc'])
    #     return acc > self.stop_at

    def postprocess_epoch(self, sample=None, results=None, epoch=None, subset=None, training=True, **kwargs):

        if training:
            loss = np.mean(results['scalar']['loss'])
            self.scheduler.step(float(loss))
            results['scalar'][f'lr'] = self.optimizers['net'].dense.param_groups[0]['lr']

        else:
            predictions = self.evaluate('test')

            z = predictions.data['z']
            y = predictions.data['y']

            y = as_numpy(y.detach())
            z = as_numpy(z.detach())

            fig, ax = plt.subplots()
            sc = ax.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')

            size = 30
            lp = lambda i: plt.plot([],color=sc.cmap(sc.norm(i)), ms=np.sqrt(size), mec="none",
                                    label="{:g}".format(i), ls="", marker="o")[0]
            handles = [lp(i) for i in np.arange(10)]
            plt.legend(handles=handles)

            ax.grid(True)
            plt.show()

        return results

    def train_iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x, y = sample['x'], sample['y']

        x = x.view(len(x), -1)
        lx = x.shape[-1]

        net = self.networks['net']
        opt = self.optimizers['net']

        dx = self.pdist(x, x)

        z, z_scaled = net(x)

        dz = self.pdist(z_scaled, z_scaled)

        # w = (1 / (dx + 1)) ** (.2)
        # w = w - torch.diag(w.diag())
        #
        # w = w / w.sum(dim=1, keepdim=True)

        # w = 1

        # w = (dx / lx < .2)

        # loss_dist = (loss_dist * w).sum(dim=-1)

        topk = torch.topk(dx, int(len(x) * self.hparams.perplexity_top), dim=-1, largest=False)
        bottomk = torch.topk(dx, int(len(x) * self.hparams.perplexity_bottom), dim=-1, largest=True)

        dz_local = dz[torch.arange(len(dz)).unsqueeze(1), topk.indices]
        dz_global = dz[torch.arange(len(dz)).unsqueeze(1), bottomk.indices]

        dx_local = topk.values
        # scale = math.sqrt(lx)
        scale = 1

        loss_dist_top = F.l1_loss(dz_local, dx_local / scale, reduction='none')

        th = 4 * float(torch.quantile(dx_local, q=.95)) / scale
        loss_dist_all = torch.clip(th - dz_global, min=0)

        mu = z.mean(dim=0)
        sig2 = z.var(dim=0)

        loss_reg = (torch.pow(mu, 2) + sig2) / 2 - .5 * torch.log(sig2) - 0.5
        loss_reg = loss_reg.sum()

        if self.reduction == 'sum':
            loss_dist = loss_dist_top.sum() + loss_dist_all.sum()
        else:
            loss_dist = loss_dist_top.mean() + loss_dist_all.mean()

        # if self.reduction == 'sum':
        #     loss_dist = loss_dist_top.sum()
        # else:
        #     loss_dist = loss_dist_top.mean()

        loss = loss_dist + self.hparams.reg_weight * loss_reg

        if training:
            opt.apply(loss)

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['loss_reg'].append(float(loss_reg / self.hparams.emb_size))
        results['scalar']['loss_dist'].append(float(loss_dist))
        results['scalar']['loss_dist_top'].append(float(loss_dist_top.mean()))
        results['scalar']['loss_dist_all'].append(float(loss_dist_all.mean()))
        results['scalar']['mu'].append(float(mu.mean()))
        results['scalar']['sig2'].append(float(sig2.mean()))

        return results

    def predict_iteration(self, sample=None, results=None, subset=None, predicting=True, **kwargs):

        if predicting:
            x = sample
        else:
            x, y = sample['x'], sample['y']

        x = x.view(len(x), -1)
        net = self.networks['net']

        z, z_scaled = net(x)

        if predicting:
            transforms = {'z': z, 'z_scaled': z_scaled}
        else:
            transforms = {'z': z, 'y': y, 'z_scaled': z_scaled}

        return transforms, results

    def postprocess_inference(self, sample=None, results=None, subset=None, predicting=True, **kwargs):

        return results


def get_deep_tsne_parser():

    parser = basic_beam_parser()
    parser.add_argument('--emb-size', type=int, default=2, help='Size of embedding dimension')
    parser.add_argument('--p-norm', type=int, default=2, help='The norm degree')
    parser.add_argument('--perplexity-top', type=float, default=.02, help='The number of nearest neighbors that is used in'
                                                                  ' other manifold learning algorithms')
    parser.add_argument('--perplexity-bottom', type=float, default=.9, help='The number of farest neighbors that is used in'
                                                                   ' other manifold learning algorithms')
    parser.add_argument('--reg-weight', type=float, default=.0, help='Regularization weight factor')
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

