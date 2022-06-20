#!/usr/bin/env python
# coding: utf-8

# In[1]:
import collections

import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import os

from src.beam import beam_arguments, Experiment
from src.beam import UniversalDataset, UniversalBatchSampler
from src.beam import Algorithm, PackedFolds
from src.beam import LinearNet
from src.beam import DataTensor, BeamOptimizer
from src.beam.utils import is_notebook
from torchvision import transforms
import torchvision
from ray import tune
from functools import partial
from collections import namedtuple

class ReBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, rezero=True, activation='celu'):
        super(ReBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.rezero = rezero
        if self.rezero:
            self.resweight = self.resweight = nn.Parameter(torch.Tensor([0]), requires_grad=True)

        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'celu':
            self.activation = nn.CELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # With or witout ReZero connection is after the nonlinearity
        if self.rezero == True:
            # ReZero
            out = self.resweight * self.activation(out) + x
        elif self.rezero == False:
            # Nominal
            out = self.activation(out) + x
        return out


class ResBlock(nn.Module):
    """
    Iniialize a residual block with two convolutions followed by batchnorm layers
    """

    def __init__(self, in_size: int, out_size: int, stride=1, activation='celu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=stride, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_size)
        self.batchnorm2 = nn.BatchNorm2d(out_size)

        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'celu':
            self.activation = nn.CELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()

    def convblock(self, x):
        x = self.activation(self.batchnorm1(self.conv1(x)))
        x = self.activation(self.batchnorm2(self.conv2(x)))
        return x

    def forward(self, x):
        return x + self.convblock(x)  # skip connection


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: self.weight.data.fill_(weight_init)
        if bias_init is not None: self.bias.data.fill_(bias_init)
        self.weight.requires_grad = not weight_freeze
        self.bias.requires_grad = not bias_freeze


class Cifar10Network(nn.Module):
    """Simple Convolutional and Fully Connect network."""

    def __init__(self, channels, dropout=.0, weight=0.125, bn_weight_init=1.0, rezero=False, param_knob=None, activation='celu'):

        super().__init__()
        channels = {'prep': channels // 8, 'layer1': channels // 4, 'layer2': channels // 2, 'layer3': channels}

        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'celu':
            self.activation = nn.CELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()

        self.weight = weight
        self.rezero = rezero
        # Layers
        self.conv_prep = nn.Conv2d(3, channels['prep'], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_prep = BatchNorm(channels['prep'], weight_init=bn_weight_init)
        self.conv1 = nn.Conv2d(channels['prep'], channels['layer1'], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm(channels['layer1'], weight_init=bn_weight_init)
        self.layer1_resblock = ReBlock(channels['layer1'], channels['layer1'], rezero=self.rezero)
        self.conv2 = nn.Conv2d(channels['layer1'], channels['layer2'], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(channels['layer2'], weight_init=bn_weight_init)
        self.conv3 = nn.Conv2d(channels['layer2'], channels['layer3'], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = BatchNorm(channels['layer3'], weight_init=bn_weight_init)
        self.layer3_resblock = ReBlock(channels['layer3'], channels['layer3'], rezero=self.rezero)
        self.pool = nn.MaxPool2d(2)
        self.classifier_pool = nn.MaxPool2d(4)
        # self.drop = nn.Dropout(p=drop_p)
        self.classifier_fc = nn.Linear(channels['layer3'], 10, bias=False)

        # self.t_resblock1 = ReBlock(64, 128, rezero=self.rezero)

    def forward(self, x):
        # cuda0 = torch.device('cuda:0')
        """Compute a forward pass."""
        # Prep
        # x = self.drop(x)
        x = self.activation(self.bn_prep(self.conv_prep(x)))
        # Layer1 + ResBlock
        x = self.activation(self.bn1(self.pool(self.conv1(x))))
        x = self.layer1_resblock(x)
        # Layer 2
        x = self.activation(self.bn2(self.pool(self.conv2(x))))
        # Layer3 + ResBlock
        x = self.activation(self.bn3(self.pool(self.conv3(x))))
        x = self.layer3_resblock(x)
        # Classifier
        x = self.classifier_pool(x)
        x = x.view(x.size(0), x.size(1))
        # x = self.drop(x)
        x = self.classifier_fc(x)
        x = x * self.weight
        return x


# class ResBlock(nn.Module):
#     """
#     Iniialize a residual block with two convolutions followed by batchnorm layers
#     """
#
#     def __init__(self, in_size: int, out_size: int, stride=1, bias=False, activation=None):
#         super().__init__()
#
#         if activation is None:
#             activation = nn.GELU()
#
#         self.res = nn.Sequential(nn.BatchNorm2d(in_size), activation,
#                                  nn.Conv2d(in_size, in_size, kernel_size=3, stride=stride, padding=1, bias=bias),
#                                  nn.BatchNorm2d(in_size), activation,
#                                  nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1, bias=bias))
#         self.stride = stride
#         self.repeat = out_size // in_size
#
#         if self.stride > 1:
#             self.identity = nn.MaxPool2d(self.stride)
#         else:
#             self.identity = nn.Identity()
#
#
#
#     def forward(self, x):
#
#         r = self.res(x)
#
#         x = self.identity(x)
#
#         if self.repeat > 1:
#             x = torch.repeat_interleave(x, self.repeat, dim=1)
#
#         return x + r  # skip connection
#
#
# class Cifar10Network(nn.Module):
#     """Simple Convolutional and Fully Connect network."""
#     def __init__(self, channels=256, dropout=.5, activation='gelu', temperature=.125):
#         super().__init__()
#
#         if activation == 'gelu':
#             activation = nn.GELU()
#         elif activation == 'celu':
#             activation = nn.CELU()
#         elif activation == 'relu':
#             activation = nn.ReLU()
#         else:
#             raise NotImplementedError
#
#         self.temperature = temperature
#         self.conv = nn.Sequential(nn.Conv2d(3, channels // 4, kernel_size=3, stride=1, padding=1, bias=True),
#                                   ResBlock(channels // 4, channels // 2, stride=1, bias=False),
#                                   nn.MaxPool2d(2),
#                                   ResBlock(channels // 2, channels, stride=1, bias=False),
#                                   nn.MaxPool2d(2),
#                                   ResBlock(channels, channels, stride=1, bias=False),
#                                   nn.MaxPool2d(2),
#                                   ResBlock(channels, channels, stride=1, bias=False, activation=activation),
#                                   nn.MaxPool2d(2),
#                                   nn.Flatten(),
#                                   nn.BatchNorm1d(channels * 4),
#                                   activation,
#                                   nn.Dropout(dropout),
#                                   nn.Linear(channels * 4, 32, bias=False),
#                                   nn.BatchNorm1d(32),
#                                   activation,
#                                   nn.Linear(32, 10, bias=True),
#                                   )
#
#
#     def forward(self, x):
#
#         x = self.conv(x) * self.temperature
#         return x

class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x[..., y0:y0+self.h, x0:x0+self.w] = 0.0
        return x

    def options(self, shape):
        *_, H, W = shape
        return [{'x0': x0, 'y0': y0} for x0 in range(W+1-self.w) for y0 in range(H+1-self.h)]

class CIFAR10Dataset(UniversalDataset):

    def __init__(self, path, train_batch_size, eval_batch_size,
                 device='cuda', scale=(0.6, 1.1), ratio=(.95, 1.05)):
        super().__init__()

        augmentations = transforms.Compose([transforms.RandomCrop(32),
                                            transforms.RandomHorizontalFlip()])

        self.t_basic =  transforms.Compose([transforms.Lambda(lambda x: (x / 255)),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        self.t_train = transforms.Compose([augmentations, self.t_basic])

        file = os.path.join(path, 'dataset_uint8.pt')
        if os.path.exists(file):
            x_train, x_test, y_train, y_test = torch.load(file, map_location=device)

        else:
            dataset_train = torchvision.datasets.CIFAR10(root=path, train=True,
                                                         transform=torchvision.transforms.PILToTensor(), download=True)
            dataset_test = torchvision.datasets.CIFAR10(root=path, train=False,
                                                        transform=torchvision.transforms.PILToTensor(), download=True)

            x_train = torch.stack([dataset_train[i][0] for i in range(len(dataset_train))]).to(device)
            x_test = torch.stack([dataset_test[i][0] for i in range(len(dataset_test))]).to(device)

            y_train = torch.LongTensor(dataset_train.targets).to(device)
            y_test = torch.LongTensor(dataset_test.targets).to(device)

            torch.save((x_train, x_test, y_train, y_test), file)

        self.data = torch.cat([x_train, x_test])
        self.labels = torch.cat([y_train, y_test])

        # self.data = PackedFolds({'train': x_train, 'test': x_test})
        # self.labels = PackedFolds({'train': y_train, 'test': y_test})

        test_indices = len(x_train) + torch.arange(len(x_test))

        self.split(validation=None, test=test_indices)
        self.build_samplers(train_batch_size, eval_batch_size)
        # self.cutout = Cutout(8, 8)

    def __getitem__(self, index):

        x = self.data[index]

        if self.training:
            x = self.t_train(x)
        else:
            x = self.t_basic(x)

        x = x.to(memory_format=torch.channels_last)

        return {'x': x, 'y': self.labels[index]}

# Maximum learing rate at epoch 5
# linear increase from start and linear decay from epoch 5 to end of epochs
class LRPolicy(object):
    def __init__(self, n_epochs=24):
        if n_epochs < 5:
            n_epochs = 6
        self.n_epochs = n_epochs

    def __call__(self, epoch):

        if epoch == 0:
            return 1

        # steps = np.array([0]*15+ [0.44,0,0,0,0] * int(self.n_epochs/4))
        piecewiselin = np.interp(epoch ,[0,5,self.n_epochs], [0,0.4, 0])
        print(piecewiselin)
        return piecewiselin
        # return np.interp(epoch ,[0,7,9,self.n_epochs], [0.1,0.44,0.01, 0])
        #return np.interp(epoch ,[0, 7, 9, 14, 15, 17, 24, 25, 27, self.n_epochs], [0.1, 0.44, 0.01, 0.008, 0.44, 0.006, 0.04, 0.44, 0.04, 0])


class CIFAR10Algorithm(Algorithm):

    def __init__(self, *args, **argv):
        super().__init__(*args, **argv)
        # self.scheduler = self.optimizers['net'].set_scheduler(torch.optim.lr_scheduler.StepLR,
        #                                                       1, gamma=self.experiment.gamma)

        self.scheduler = self.optimizers['net'].set_scheduler(torch.optim.lr_scheduler.LambdaLR, last_epoch=- 1,
                                                              lr_lambda=LRPolicy(self.n_epochs))

    def postprocess_epoch(self, sample=None, aux=None, results=None, epoch=None, subset=None, training=True):

        x, y = sample['x'], sample['y']

        # results['images']['sample'] = x[:16].view(16, 1, 28, 28).data.cpu()

        if training:
            results['scalar'][f'lr'] = self.optimizers['net'].dense.param_groups[0]['lr']
            # self.optimizers['net'].reset()
            self.scheduler.step()

        aux = {}
        return aux, results

    def iteration(self, sample=None, aux=None, results=None, subset=None, training=True):

        x, y = sample['x'], sample['y']

        net = self.networks['net']
        opt = self.optimizers['net']

        with torch.cuda.amp.autocast(enabled=self.amp):
            y_hat = net(x)
            loss = F.cross_entropy(y_hat, y, reduction='sum', label_smoothing=0.2)

        opt.apply(loss, training=training)

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['acc'].append(float((y_hat.argmax(1) == y).float().mean()))

        return aux, results

    def report(self, results, i):

        acc = np.mean(results['test']['scalar']['acc'])

        if self.hpo == 'tune':
            tune.report(mean_accuracy=acc)
        elif self.hpo == 'optuna':

            self.trial.report(acc, i)
            results['objective'] = acc

        else:
            raise NotImplementedError

        return results

    def inference(self, sample=None, aux=None, results=None, subset=None, with_labels=True):

        if with_labels:
            x, y = sample['x'], sample['y']

        else:
            x = sample

        net = self.networks['net']

        y_hat = net(x)

        # add scalar measurements
        results['predictions']['y_pred'].append(y_hat.detach())

        if with_labels:
            results['scalar']['acc'].append(float((y_hat.argmax(1) == y).float().mean()))
            aux['predictions']['target'].append(y)

        return aux, results

    def postprocess_inference(self, sample=None, aux=None, results=None, subset=None, with_labels=True):
        y_pred = torch.cat(results['predictions']['y_pred'])

        y_pred = torch.argmax(y_pred, dim=1).data.cpu().numpy()

        if with_labels:
            y_true = torch.cat(aux['predictions']['target']).data.cpu().numpy()
            precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)
            results['metrics']['precision'] = precision
            results['metrics']['recall'] = recall
            results['metrics']['fscore'] = fscore
            results['metrics']['support'] = support

        return aux, results


def cifar10_algorithm_generator(experiment):

    dataset = CIFAR10Dataset(experiment.path_to_data,
                           experiment.batch_size_train, experiment.batch_size_eval,
                           scale=(experiment.scale_down, experiment.scale_up),
                           ratio=(experiment.ratio_down, experiment.ratio_up),
                             device=experiment.device)

    dataloader = dataset.build_dataloaders(num_workers=experiment.cpu_workers)

    # choose your network
    net = Cifar10Network(experiment.channels, dropout=experiment.dropout, activation=experiment.activation)

    # net.apply(initialize_weights)
    optimizer = None
    optimizer = partial(BeamOptimizer, dense_args={'lr': experiment.lr_dense, 'weight_decay': experiment.weight_decay,
                                               'momentum': .9, 'nesterov': True}, clip=experiment.clip, accumulate=experiment.accumulate,
                                                                amp=experiment.amp,
                              sparse_args=None, dense_optimizer='SGD', sparse_optimizer='SparseAdam')

    # we recommend using the algorithm argument to determine the type of algorithm to be used
    Alg = globals()[experiment.algorithm]
    alg = Alg(net, dataloader, experiment, optimizers=optimizer)

    return alg


# ## Training

if __name__ == '__main__':

    # here you put all actions which are performed only once before initializing the workers
    # for example, setting running arguments and experiment:

    args = beam_arguments("--project-name=CIFAR10 --root-dir=/home/shared/data/results --algorithm=CIFAR10Algorithm",
                          "--epoch-length=100000 --n-epochs=2 --clip=1 --parallel=1 --parallel=2",
                          path_to_data='/home/elad/projects/CIFAR10')

    experiment = Experiment(args)

    raise NotImplementedError
    # alg = experiment(CIFAR10_algorithm_generator, experiment)

    # here we initialize the workers (can be single or multiple workers, depending on the configuration)
    # alg = experiment.run(run_CIFAR10)

    # ## Inference

    inference = alg('test')

    print('Test inference results:')
    for n, v in inference['metrics'].items():
        print(f'{n}:')
        print(v)

