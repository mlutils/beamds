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
from src.beam import Algorithm
from src.beam import LinearNet
from src.beam import DataTensor, BeamOptimizer
from src.beam.utils import is_notebook
from torchvision import transforms
import torchvision



class ResBlock(nn.Module):
    """
    Iniialize a residual block with two convolutions followed by batchnorm layers
    """

    def __init__(self, in_size: int, out_size: int, stride=1, bias=False):
        super().__init__()
        self.res = nn.Sequential(nn.BatchNorm2d(in_size), nn.CELU(alpha=0.3),
                                 nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1, bias=bias))
        self.stride = stride
        self.repeat = out_size // in_size


    def forward(self, x):

        r = self.res(x)
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]

        if self.repeat > 1:
            x = torch.repeat_interleave(x, self.repeat, dim=1)

        return x + r  # skip connection


class Cifar10Network(nn.Module):
    """Simple Convolutional and Fully Connect network."""
    def __init__(self, channels=128):
        super().__init__()

        # activation = nn.CELU(alpha=0.3)
        activation = nn.GELU()

        self.conv = nn.Sequential(nn.Conv2d(3, channels // 4, kernel_size=3, stride=1, padding=1, bias=True),
                                  nn.BatchNorm2d(channels // 4), activation,
                                  nn.Conv2d(channels // 4, channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.MaxPool2d(2),
                                  nn.BatchNorm2d(channels // 2), activation,
                                  nn.Conv2d(channels // 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.MaxPool2d(2),
                                  nn.BatchNorm2d(channels), activation,
                                  nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.MaxPool2d(4),
                                  nn.Flatten(),
                                  nn.BatchNorm1d(channels * 4),
                                  activation,
                                  nn.Dropout(.5),
                                  nn.Linear(channels * 4, 32, bias=False),
                                  nn.BatchNorm1d(32),
                                  activation,
                                  nn.Linear(32, 10, bias=True),
                                  )

        # self.conv = nn.Sequential(nn.Conv2d(3, channels // 4, kernel_size=3, stride=1, padding=1, bias=True),
        #                           nn.BatchNorm2d(channels // 4), nn.CELU(alpha=0.3),
        #                           nn.Conv2d(channels // 4, channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
        #                           nn.BatchNorm2d(channels // 2), nn.CELU(alpha=0.3),
        #                           nn.Conv2d(channels // 2, channels, kernel_size=3, stride=2, padding=1, bias=False),
        #                           nn.BatchNorm2d(channels), nn.CELU(alpha=0.3),
        #                           nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
        #                            nn.AdaptiveMaxPool2d((2, 2)),
        #                            nn.Flatten(),
        #                           nn.BatchNorm1d(channels * 4),
        #                            nn.CELU(alpha=0.3),
        #                            nn.Dropout(.25),
        #                            nn.Linear(channels * 4, 10, bias=True)
        #                            )

        # self.conv = nn.Sequential(nn.Conv2d(3, channels // 4, kernel_size=3, stride=1, padding=1, bias=True),
        #                           nn.BatchNorm2d(channels // 4), nn.CELU(alpha=0.3),
        #                           nn.Conv2d(channels // 4, channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
        #                           nn.MaxPool2d(2),
        #                           nn.BatchNorm2d(channels // 2), nn.CELU(alpha=0.3),
        #                           nn.Conv2d(channels // 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
        #                           nn.MaxPool2d(2),
        #                           nn.BatchNorm2d(channels), nn.CELU(alpha=0.3),
        #                           nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
        #                           nn.MaxPool2d(2),
        #                           nn.AdaptiveMaxPool2d((2, 2)),
        #                           nn.Flatten(),
        #                           nn.CELU(alpha=0.3),
        #                           nn.Dropout(.25),
        #                           nn.Linear(channels * 4, 10, bias=True)
        #                           )

        # self.conv = nn.Sequential(nn.Conv2d(3, channels // 4, kernel_size=3, stride=1, padding=1, bias=True),
        #                           ResBlock(channels // 4, channels // 2, stride=2, bias=False),
        #                           ResBlock(channels // 2, channels, stride=2, bias=False),
        #                           ResBlock(channels, channels, stride=2, bias=False),
        #                           nn.AdaptiveMaxPool2d((2, 2)),
        #                           nn.Flatten(),
        #                           nn.CELU(alpha=0.3),
        #                           nn.Dropout(.25),
        #                           nn.Linear(channels * 4, 10, bias=True)
        #                           )

        # self.conv = nn.Sequential(nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=True),
        #                            ResBlock(channels, channels, stride=2),
        #                            ResBlock(channels, channels, stride=2),
        #                            ResBlock(channels, channels, stride=2),
        #                            nn.AdaptiveMaxPool2d((2, 2)),
        #                            nn.Flatten(),
        #                            nn.ReLU(),
        #                            nn.Linear(channels * 4, 10, bias=True)
        #                            )

    def forward(self, x):

        x = self.conv(x)
        return x



# In[2]:

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# def initialize_weights(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.orthogonal_(m.weight.data)
#         if m.bias is not None:
#             nn.init.constant_(m.bias.data, 0)
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight.data, 1)
#         nn.init.constant_(m.bias.data, 0)
#     elif isinstance(m, nn.Linear):
#         nn.init.orthogonal_(m.weight.data)
#         if m.bias is not None:
#             nn.init.constant_(m.bias.data, 0)

class CIFAR10Dataset(UniversalDataset):

    def __init__(self, path, train_batch_size, eval_batch_size, device='cuda'):
        super().__init__()

        file = os.path.join(path, 'dataset.pt')
        if os.path.exists(file):
            x_train, x_test, y_train, y_test = torch.load(file)

        else:
            dataset_train = torchvision.datasets.CIFAR10(root=path, train=True, transform=torchvision.transforms.ToTensor(), download=True)
            dataset_test = torchvision.datasets.CIFAR10(root=path, train=False, transform=torchvision.transforms.ToTensor(), download=True)

            basic_transform = transforms.Compose(
                                [transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

            x_train = torch.stack([basic_transform(di) for di in dataset_train.data]).to(device)
            x_test = torch.stack([basic_transform(di) for di in dataset_test.data]).to(device)

            y_train = torch.LongTensor(dataset_train.targets)
            y_test = torch.LongTensor(dataset_test.targets)

            torch.save((x_train, x_test, y_train, y_test), file)


        self.augmentations = transforms.Compose(
                            [transforms.RandomResizedCrop(32, scale=(0.6, 1.1),
                                                   ratio=(.95, 1.05)), transforms.RandomHorizontalFlip()])

        self.data = torch.cat([x_train, x_test])
        self.labels = torch.cat([y_train, y_test])

        test_indices = len(x_train) + torch.arange(len(x_test))

        self.split(validation=None, test=test_indices)
        self.build_samplers(train_batch_size, eval_batch_size)

    def __getitem__(self, index):

        x = self.data[index]
        if self.train:
            x = self.augmentations(x)

        return {'x': x, 'y': self.labels[index]}


class CIFAR10Algorithm(Algorithm):

    def __init__(self, *args, **argv):
        super().__init__(*args, **argv)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizers['net'].dense, 1, gamma=1/np.sqrt(10))

    def postprocess_epoch(self, sample=None, aux=None, results=None, epoch=None, subset=None, training=True):

        x, y = sample['x'], sample['y']

        # results['images']['sample'] = x[:16].view(16, 1, 28, 28).data.cpu()

        if training:
            results['scalar'][f'lr'] = self.optimizers['net'].dense.param_groups[0]['lr']
            self.scheduler.step()

        aux = {}
        return aux, results

    def iteration(self, sample=None, aux=None, results=None, subset=None, training=True):

        x, y = sample['x'], sample['y']

        net = self.networks['net']
        opt = self.optimizers['net']

        with torch.cuda.amp.autocast(enabled=self.amp):
            y_hat = net(x)
            loss = F.cross_entropy(y_hat, y, reduction='sum')

        opt.apply(loss, training=training)

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['acc'].append(float((y_hat.argmax(1) == y).float().mean()))

        return aux, results

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
                           experiment.batch_size_train, experiment.batch_size_eval, device=experiment.device)

    dataloader = dataset.build_dataloaders(num_workers=experiment.cpu_workers)

    # choose your network
    net = Cifar10Network()

    net.apply(initialize_weights)
    # optimizer = BeamOptimizer(net, dense_args={'lr': experiment.lr_dense,
    #                                            'momentum': .9}, clip=0, accumulate=1, amp=experiment.amp,
    #                           sparse_args=None, dense_optimizer='SGD', sparse_optimizer='SparseAdam')

    # we recommend using the algorithm argument to determine the type of algorithm to be used
    Alg = globals()[experiment.algorithm]
    alg = Alg(net, dataloader, experiment, optimizers=None)

    return alg


def run_cifar10(rank, world_size, experiment):

    dataset = CIFAR10Dataset(experiment.path_to_data,
                           experiment.batch_size_train, experiment.batch_size_eval)

    dataloader = dataset.build_dataloaders(num_workers=experiment.cpu_workers)

    # choose your network
    net = LinearNet(784, 256, 10, 4)

    # we recommend using the algorithm argument to determine the type of algorithm to be used
    Alg = globals()[experiment.algorithm]
    alg = Alg(net, dataloader, experiment)

    # simulate input to the network
    x = next(alg.data_generator('validation'))[1]['x']
    x = x.view(len(x), -1)

    experiment.writer_control(enable=not (bool(rank)), networks=alg.get_networks(), inputs={'net': x})

    for results in iter(alg):
        experiment.save_model_results(results, alg,
                                      print_results=True, visualize_results='yes',
                                      store_results='logscale', store_networks='logscale',
                                      visualize_weights=True,
                                      argv={'images': {'sample': {'dataformats': 'NCHW'}}})

    if world_size > 1:
        return results
    else:
        return alg, results


# ## Training

if __name__ == '__main__':

    # here you put all actions which are performed only once before initializing the workers
    # for example, setting running arguments and experiment:

    args = beam_arguments("--project-name=CIFAR10 --root-dir=/home/shared/data/results --algorithm=CIFAR10Algorithm",
                          "--epoch-length=100000 --n-epochs=2 --clip=1 --parallel=1 --parallel=2",
                          path_to_data='/home/elad/projects/CIFAR10')

    experiment = Experiment(args)

    alg = experiment(CIFAR10_algorithm_generator, experiment)

    # here we initialize the workers (can be single or multiple workers, depending on the configuration)
    # alg = experiment.run(run_CIFAR10)

    # ## Inference

    inference = alg('test')

    print('Test inference results:')
    for n, v in inference['metrics'].items():
        print(f'{n}:')
        print(v)

