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
from sklearn.datasets import fetch_covtype


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


class CovtypeDataset(UniversalDataset):

    def __init__(self, path,
                 train_batch_size, eval_batch_size, device='cuda', weight_factor=.5, seed=5782):

        dataset = fetch_covtype(data_home=path)
        data = dataset['data']
        columns = dataset['feature_names']
        y = dataset['target']
        df = pd.DataFrame(data=data, columns=columns, index=np.arange(len(data)))

        soils_columns = [c for c in df.columns if 'Soil' in c]
        soil = np.where(df[soils_columns])[1]

        wilderness_columns = [c for c in df.columns if 'Wilderness' in c]
        wilderness = np.where(df[wilderness_columns])[1]

        df_cat = pd.DataFrame({'Soil': soil, 'Wilderness': wilderness})
        df_num = df.drop(columns=(soils_columns+wilderness_columns))

        covtype = pd.concat([df_num, df_cat], axis=1)
        super().__init__(x=covtype.values, y=y)

        self.split(validation=.2, test=.2, seed=seed, stratify=True, labels=self.data['y'])
        self.build_samplers(train_batch_size, eval_batch_size, oversample=True, weight_factor=weight_factor)


class CovtypeAlgorithm(Algorithm):

    def __init__(self, *args, **argv):
        super().__init__(*args, **argv)

        networks, dataloaders, experiment = args

        self.scheduler = self.optimizers['net'].set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlatue, last_epoch=- 1,
                                                              lr_lambda=LRPolicy(gain=experiment.gain,
                                                                                 turn_point=experiment.turn_point,
                                                                                 final_point=experiment.final_point,
                                                                                 minimal_gain=experiment.minimal_gain))

    def postprocess_epoch(self, sample=None, aux=None, results=None, epoch=None, subset=None, training=True):

        x, y = sample['x'], sample['y']

        results['images']['sample'] = x[:16].view(16, 3, 32, 32).data.cpu()

        if training:
            results['scalar'][f'lr'] = self.optimizers['net'].dense.param_groups[0]['lr']

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

        if training:
            self.scheduler.step()

        return aux, results

    def report(self, results, i):

        acc = np.mean(results['validation']['scalar']['acc'])

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


def covtype_algorithm_generator(experiment, **kwargs):

    dataset = CovtypeDataset(experiment.path_to_data,
                             experiment.batch_size_train, experiment.batch_size_eval,
                             padding=experiment.padding, device=experiment.device)

    pin_memory = 'cpu' not in str(experiment.device)
    dataloader = dataset.build_dataloaders(num_workers=experiment.cpu_workers, pin_memory=pin_memory)

    # choose your network
    net = Cifar10Network(experiment.channels, dropout=experiment.dropout, activation=experiment.activation)

    optimizer = None
    optimizer = partial(BeamOptimizer, dense_args={'lr': experiment.lr_dense, 'weight_decay': experiment.weight_decay,
                                                   'momentum': experiment.beta1, 'nesterov': True}, clip=experiment.clip, accumulate=experiment.accumulate,
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

    path_to_data = '/home/shared/data/dataset/covtype'
    root_dir = '/home/shared/data/results/covtype'

    args = beam_arguments(
        f"--project-name=covtype --root-dir={root_dir} --algorithm=CovtypeAlgorithm --device=1 --half --lr-d=1e-4 --batch-size=512",
        "--n-epochs=2 --epoch-length-train=50000 --epoch-length-eval=10000 --clip=0 --parallel=1 --accumulate=1 --cudnn-benchmark",
        "--weight-decay=.00256 --beta1=0.9 --beta2=0.9",
        path_to_data=path_to_data, gamma=1., dropout=.0, activation='celu', channels=512,
        scale_down=.7, scale_up=1.4, ratio_down=.7, ratio_up=1.4)


    experiment = Experiment(args)
    alg = experiment(covtype_algorithm_generator, tensorboard_arguments={'images': {'sample': {'dataformats': 'NCHW'}}})

    # ## Inference
    inference = alg('test')

    print('Test inference results:')
    for n, v in inference['metrics'].items():
        print(f'{n}:')
        print(v)

