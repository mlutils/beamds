#!/usr/bin/env python
# coding: utf-8

# In[1]:

from examples.utils import add_beam_to_path
add_beam_to_path()

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
from src.beam import DataTensor, BeamOptimizer, beam_logger

from torchvision import transforms
import torchvision
from ray import tune
import kornia
from kornia.augmentation.container import AugmentationSequential


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
        self.classifier_fc = nn.Linear(channels['layer3'], 10, bias=False)

    def forward(self, x):
        # cuda0 = torch.device('cuda:0')
        """Compute a forward pass."""
        # Prep
        x = self.activation(self.bn_prep(self.conv_prep(x)))
        x = self.activation(self.bn1(self.pool(self.conv1(x))))
        x = self.layer1_resblock(x)
        x = self.activation(self.bn2(self.pool(self.conv2(x))))
        x = self.activation(self.bn3(self.pool(self.conv3(x))))
        x = self.layer3_resblock(x)
        # Classifier
        x = self.classifier_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier_fc(x)
        x = x * self.weight
        return x


class CIFAR10Dataset(UniversalDataset):

    def __init__(self, hparams):
        super().__init__()

        path = hparams.path_to_data
        device = hparams.device
        padding = hparams.padding

        self.augmentations = AugmentationSequential(kornia.augmentation.RandomHorizontalFlip(),
                                                    kornia.augmentation.RandomCrop((32, 32), padding=padding,
                                                                                   padding_mode='reflect'))

        self.mu = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view(1, -1, 1, 1).to(device)
        self.sigma = torch.FloatTensor([0.247, 0.243, 0.261]).view(1, -1, 1, 1).to(device)

        # self.t_basic = transforms.Compose([transforms.Lambda(lambda x: (x.half() / 255)),
        #                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

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

        # self.data = torch.cat([x_train, x_test])
        # self.labels = torch.cat([y_train, y_test])
        # test_indices = len(x_train) + torch.arange(len(x_test))
        # self.split(validation=.2, test=test_indices, seed=hparams.split_dataset_seed)

        self.data = PackedFolds({'train': x_train, 'test': x_test})
        self.labels = PackedFolds({'train': y_train, 'test': y_test})
        self.split(validation=.2, test=self.labels['test'].index, seed=hparams.split_dataset_seed)

    def getitem(self, ind):

        x = self.data[ind]

        # x = self.t_basic(x)

        x = x.half() / 255

        # print(x.shape)
        if self.training:
            x = self.augmentations(x)

        x = (x.float() - self.mu) / self.sigma

        x = x.to(memory_format=torch.channels_last)

        return {'x': x, 'y': self.labels[ind]}


# class CIFAR10Dataset(UniversalDataset):
#
#     def __init__(self, hparams):
#         super().__init__()
#
#         path = hparams.path_to_data
#         device = hparams.device
#         padding = hparams.padding
#
#         augmentations = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                             transforms.RandomCrop(32, padding=padding, padding_mode='edge'),])
#
#         self.t_basic = transforms.Compose([transforms.Lambda(lambda x: (x / 255)),
#                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
#
#         self.t_train = transforms.Compose([augmentations, self.t_basic])
#
#         file = os.path.join(path, 'dataset_uint8.pt')
#         if os.path.exists(file):
#             x_train, x_test, y_train, y_test = torch.load(file, map_location=device)
#
#         else:
#             dataset_train = torchvision.datasets.CIFAR10(root=path, train=True,
#                                                          transform=torchvision.transforms.PILToTensor(), download=True)
#             dataset_test = torchvision.datasets.CIFAR10(root=path, train=False,
#                                                         transform=torchvision.transforms.PILToTensor(), download=True)
#
#             x_train = torch.stack([dataset_train[i][0] for i in range(len(dataset_train))]).to(device)
#             x_test = torch.stack([dataset_test[i][0] for i in range(len(dataset_test))]).to(device)
#
#             y_train = torch.LongTensor(dataset_train.targets).to(device)
#             y_test = torch.LongTensor(dataset_test.targets).to(device)
#
#             torch.save((x_train, x_test, y_train, y_test), file)
#
#         # self.data = torch.cat([x_train, x_test])
#         # self.labels = torch.cat([y_train, y_test])
#         # test_indices = len(x_train) + torch.arange(len(x_test))
#         # self.split(validation=.2, test=test_indices, seed=hparams.split_dataset_seed)
#
#         self.data = PackedFolds({'train': x_train, 'test': x_test})
#         self.labels = PackedFolds({'train': y_train, 'test': y_test})
#         self.split(validation=.2, test=self.labels['test'].index, seed=hparams.split_dataset_seed)
#
#     def getitem(self, ind):
#
#         x = self.data[ind]
#
#         if self.training:
#             x = self.t_train(x)
#         else:
#             x = self.t_basic(x)
#
#         x = x.to(memory_format=torch.channels_last)
#
#         return {'x': x, 'y': self.labels[ind]}


class LRPolicy(object):
    def __init__(self, gain=.4, turn_point=500, final_point=3000, minimal_gain=1e-2):

        self.gain = gain
        self.turn_point = turn_point
        self.final_point = final_point
        self.minimal_gain = minimal_gain

    def __call__(self, epoch):

        if epoch == 0:
            return 1

        piecewiselin = np.interp(epoch ,[0, self.turn_point, self.final_point], [0, self.gain, 0])
        piecewiselin = np.clip(piecewiselin, a_min=self.minimal_gain, a_max=None)

        return piecewiselin


class CIFAR10Algorithm(Algorithm):

    def __init__(self, hparams):

        # choose your network
        net = Cifar10Network(hparams.channels, dropout=hparams.dropout,
                             activation=hparams.activation, weight=hparams.temperature)

        if 'prototype' in hparams and hparams.prototype:
            optimizer = BeamOptimizer.prototype(dense_args={'lr': hparams.lr_dense,
                                                            'weight_decay': hparams.weight_decay,
                                                           'momentum': hparams.beta1, 'nesterov': True},
                                                clip=hparams.clip_gradient, accumulate=hparams.accumulate,
                                                amp=hparams.amp,
                                                sparse_args=None, dense_optimizer='SGD')
        else:
            optimizer = BeamOptimizer(net, dense_args={'lr': hparams.lr_dense,
                                                            'weight_decay': hparams.weight_decay,
                                                           'momentum': hparams.beta1, 'nesterov': True},
                                                clip=hparams.clip_gradient, accumulate=hparams.accumulate,
                                                amp=hparams.amp,
                                                sparse_args=None, dense_optimizer='SGD')

        super().__init__(hparams, networks=net, optimizers=optimizer)
        self.scheduler = self.optimizers['net'].set_scheduler(torch.optim.lr_scheduler.LambdaLR, last_epoch=- 1,
                                                              lr_lambda=LRPolicy(gain=hparams.gain,
                                                                                 turn_point=hparams.turn_point,
                                                                                 final_point=hparams.final_point,
                                                                                 minimal_gain=hparams.minimal_gain))

    def postprocess_epoch(self, sample=None, results=None, epoch=None, subset=None, training=True, **kwargs):

        x, y = sample['x'], sample['y']

        results['images']['sample'] = x[:16].view(16, 3, 32, 32).data.cpu()

        if training:
            results['scalar'][f'lr'] = self.optimizers['net'].dense.param_groups[0]['lr']

        return results

    def iteration(self, sample=None, results=None, counter=None, subset=None, training=True, **kwargs):

        x, y = sample['x'], sample['y']

        net = self.networks['net']
        opt = self.optimizers['net']

        y_hat = net(x)
        loss = F.cross_entropy(y_hat, y, reduction='sum', label_smoothing=self.hparams.label_smoothing)

        if training:
            opt.apply(loss)

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['acc'].append(float((y_hat.argmax(1) == y).float().mean()))

        if training:
            self.scheduler.step()

        return results

    def report(self, results=None, epoch=None, **kwargs):

        acc = np.mean(results['validation']['scalar']['acc'])

        if self.hpo == 'tune':
            tune.report(mean_accuracy=acc)
        elif self.hpo == 'optuna':

            self.trial.report(acc, epoch)
            results['objective'] = acc

        else:
            raise NotImplementedError

        return results

    def inference(self, sample=None, results=None, subset=None, predicting=True, **kwargs):

        if predicting:
            x = sample
        else:
            x, y = sample['x'], sample['y']

        net = self.networks['net']
        y_hat = net(x)

        # add scalar measurements
        results['predictions']['y_pred'].append(y_hat.detach())

        if not predicting:
            results['scalar']['acc'].append(float((y_hat.argmax(1) == y).float().mean()))
            results['predictions']['target'].append(y)
            return {'y': y, 'y_hat': y_hat}, results

        return y_hat, results

    def postprocess_inference(self, sample=None, results=None, subset=None, predicting=True, **kwargs):
        y_pred = torch.cat(results['predictions']['y_pred'])

        y_pred = torch.argmax(y_pred, dim=1).data.cpu().numpy()

        if not predicting:
            y_true = torch.cat(results['predictions']['target']).data.cpu().numpy()
            precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)
            results['metrics']['precision'] = precision
            results['metrics']['recall'] = recall
            results['metrics']['fscore'] = fscore
            results['metrics']['support'] = support

        return results


# ## Training

if __name__ == '__main__':

    # here you put all actions which are performed only once before initializing the workers
    # for example, setting running arguments and experiment:

    path_to_data = '/home/shared/data/dataset/cifar10'
    root_dir = '/home/shared/data/results/cifar10'

    args = beam_arguments(
        f"--project-name=cifar10 --root-dir={root_dir} --algorithm=CIFAR10Algorithm --device=1 --half --lr-d=1e-4 --batch-size=512",
        "--n-epochs=2 --epoch-length-train=50000 --epoch-length-eval=10000 --clip=0 --parallel=1 --accumulate=1 --cudnn-benchmark",
        "--weight-decay=.00256 --beta1=0.9 --beta2=0.9",
        path_to_data=path_to_data, dropout=.0, activation='celu',
        channels=512, label_smoothing=.2, padding=4, scale_down=.7, scale_up=1.4, ratio_down=.7, ratio_up=1.4)

    logger = beam_logger()

    experiment = Experiment(args)
    alg = experiment.fit(CIFAR10Algorithm, CIFAR10Dataset, tensorboard_arguments={'images': {'sample': {'dataformats': 'NCHW'}}})

    # ## Inference
    inference = alg('test')

    logger.info('Test inference results:')
    for n, v in inference.statistics['metrics'].items():
        logger.info(f'{n}:')
        logger.info(v)

