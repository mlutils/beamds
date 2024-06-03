#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import os

from src.beam import beam_arguments, Experiment
from src.beam import UniversalDataset
from src.beam import NeuralAlgorithm, as_numpy
from src.beam import BeamOptimizer
from src.beam.data import BeamData

import torchvision
import kornia
from kornia.augmentation.container import AugmentationSequential
from src.beam import beam_logger as logger


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

        path = hparams.data_path
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

        self.data = BeamData.simple({'train': x_train, 'test': x_test}, label={'train': y_train, 'test': y_test})
        self.labels = self.data.label
        self.split(validation=.2, test=self.data['test'].index, seed=hparams.split_dataset_seed)

    def getitem(self, ind):

        data = self.data[ind]
        x = data.data
        labels = data.label

        x = x.half() / 255

        if self.training:
            x = self.augmentations(x)

        x = (x.float() - self.mu) / self.sigma
        x = x.to(memory_format=torch.channels_last)

        return {'x': x, 'y': labels}


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


class CIFAR10Algorithm(NeuralAlgorithm):

    def __init__(self, hparams):

        # choose your network
        net = Cifar10Network(hparams.channels, dropout=hparams.dropout,
                             activation=hparams.activation, weight=hparams.temperature)

        if 'prototype' in hparams and hparams.prototype:
            optimizer = BeamOptimizer.prototype(dense_args={'lr': hparams.lr_dense,
                                                            'weight_decay': hparams.weight_decay,
                                                           'momentum': hparams.momentum, 'nesterov': True},
                                                clip=hparams.clip_gradient, accumulate=hparams.accumulate,
                                                amp=hparams.amp,
                                                sparse_args=None, dense_optimizer='SGD')
        else:
            optimizer = BeamOptimizer(net, dense_args={'lr': hparams.lr_dense,
                                                            'weight_decay': hparams.weight_decay,
                                                           'momentum': hparams.momentum, 'nesterov': True},
                                                clip=hparams.clip_gradient, accumulate=hparams.accumulate,
                                                amp=hparams.amp,
                                                sparse_args=None, dense_optimizer='SGD')

        super().__init__(hparams, networks=net, optimizers=optimizer)
        # self.scheduler = self.optimizers['net'].set_scheduler(torch.optim.lr_scheduler.LambdaLR, last_epoch=- 1,
        #                                                       lr_lambda=LRPolicy(gain=hparams.gain,
        #                                                                          turn_point=hparams.turn_point,
        #                                                                          final_point=hparams.final_point,
        #                                                                          minimal_gain=hparams.minimal_gain))

    def postprocess_epoch(self, sample=None, epoch=None, subset=None, training=True, **kwargs):
        x, y = sample['x'], sample['y']
        self.report_images('sample', x[:16].view(16, 3, 32, 32))

    def train_iteration(self, sample=None, counter=None, subset=None, training=True, **kwargs):

        x, y = sample['x'], sample['y']

        net = self.networks['net']
        opt = self.optimizers['net']

        y_hat = net(x)
        loss = F.cross_entropy(y_hat, y, reduction='sum', label_smoothing=self.hparams.label_smoothing)

        self.apply(loss)

        # add scalar measurements
        self.report_scalar('acc', (y_hat.argmax(1) == y).float().mean())

    def inference_iteration(self, sample=None, subset=None, predicting=True, **kwargs):

        if predicting:
            x = sample
        else:
            x, y = sample['x'], sample['y']

        net = self.networks['net']
        y_hat = net(x)

        # add scalar metrics
        self.report_scalar('y_pred', y_hat)

        if not predicting:
            self.report_scalar('acc', (y_hat.argmax(1) == y).float().mean())
            self.report_scalar('target', y)
            return {'y': y, 'y_hat': y_hat}

        return y_hat

    def postprocess_inference(self, sample=None, subset=None, predicting=True, **kwargs):

        if not predicting:

            y_pred = as_numpy(torch.argmax(self.get_scalar('y_pred'), dim=1))
            y_true = as_numpy(self.get_scalar('target'))
            precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)

            self.report_data('metrics/precision', precision)
            self.report_data('metrics/recall', recall)
            self.report_data('metrics/fscore', fscore)
            self.report_data('metrics/support', support)

            self.report_scalar('objective', self.get_scalar('acc', aggregate=True))


# ## Training

if __name__ == '__main__':

    # here you put all actions which are performed only once before initializing the workers
    # for example, setting running arguments and experiment:


    args = beam_arguments(
        f"--project-name=cifar10 --algorithm=CIFAR10Algorithm --device=1 --half --lr-d=1e-4 --batch-size=512",
        "--n-epochs=50 --epoch-length-train=50000 --epoch-length-eval=10000 --clip=0 --n-gpus=1 --accumulate=1 --no-deterministic",
        "--weight-decay=.00256 --momentum=0.9 --beta2=0.999 --temperature=1 --objective=acc --scheduler=one_cycle",
        dropout=.0, activation='gelu', channels=512, label_smoothing=.2, padding=4, scale_down=.7,
        scale_up=1.4, ratio_down=.7, ratio_up=1.4)

    experiment = Experiment(args)
    alg = experiment.fit(CIFAR10Algorithm, CIFAR10Dataset, tensorboard_arguments={'images': {'sample': {'dataformats': 'NCHW'}}})

    # ## Inference
    inference = alg('test')

    logger.info('Test inference results:')
    for n, v in inference.statistics['metrics'].items():
        logger.info(f'{n}:')
        logger.info(v)

