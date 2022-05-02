#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from src.beam import parser, Experiment
from src.beam import UniversalDataset, UniversalBatchSampler
from src.beam import Algorithm
from src.beam import LinearNet
from src.beam import DataTensor
from src.beam.utils import is_notebook

import sys

if is_notebook():
    sys.argv = sys.argv[:1]
    # get_ipython().run_line_magic('load_ext', 'autoreload')
    # get_ipython().run_line_magic('autoreload', '2')


# In[2]:


class MNISTDataset(UniversalDataset):

    def __init__(self, path, train_batch_size, eval_batch_size):
        super().__init__()
        dataset_train = torchvision.datasets.MNIST(root=path, train=True, transform=torchvision.transforms.ToTensor())
        dataset_test = torchvision.datasets.MNIST(root=path, train=False, transform=torchvision.transforms.ToTensor())

        self.data = torch.cat([dataset_train.data, dataset_test.data])
        self.labels = torch.cat([dataset_train.targets, dataset_test.targets])

        test_indices = len(dataset_train) + torch.arange(len(dataset_test))

        self.split(validation=.2, test=test_indices)
        self.build_samplers(train_batch_size, eval_batch_size)

    def __getitem__(self, index):
        return {'x': self.data[index].float() / 255, 'y': self.labels[index]}


class MNISTAlgorithm(Algorithm):

    def __init__(self, *args, **argv):
        super().__init__(*args, **argv)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizers['net'].dense, gamma=0.99)

    def postprocess_epoch(self, sample=None, aux=None, results=None, epoch=None, subset=None, training=True):
        x, y = sample['x'], sample['y']

        results['images']['sample'] = x[:16].view(16, 1, 28, 28).data.cpu()

        if training:
            self.scheduler.step()
            results['scalar'][f'lr'] = self.optimizers['net'].dense.param_groups[0]['lr']

        aux = {}
        return aux, results

    def iteration(self, sample=None, aux=None, results=None, subset=None, training=True):

        x, y = sample['x'], sample['y']

        x = x.view(len(x), -1)
        net = self.networks['net']
        opt = self.optimizers['net']

        y_hat = net(x)
        loss = F.cross_entropy(y_hat, y, reduction='mean')
        opt.apply(loss, training=training)

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['acc'].append(float((y_hat.argmax(1) == y).float().mean()))

        return aux, results

    def inference(self, sample=None, aux=None, results=None, subset=None):
        x, y = sample['x'], sample['y']

        x = x.view(len(x), -1)
        net = self.networks['net']

        y_hat = net(x)

        # add scalar measurements
        results['predictions']['y_pred'].append(y_hat.detach())
        results['scalar']['acc'].append(float((y_hat.argmax(1) == y).float().mean()))

        aux['predictions']['target'].append(y)

        return aux, results

    def postprocess_inference(self, sample=None, aux=None, results=None, subset=None):
        y_pred = torch.cat(results['predictions']['y_pred'])

        y_pred = torch.argmax(y_pred, dim=1).data.cpu().numpy()
        y_true = torch.cat(aux['predictions']['target']).data.cpu().numpy()

        precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)
        results['metrics']['precision'] = precision
        results['metrics']['recall'] = recall
        results['metrics']['fscore'] = fscore
        results['metrics']['support'] = support

        return aux, results


def run_mnist(rank, world_size, experiment):

    dataset = MNISTDataset(experiment.path_to_data,
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

    return alg


# ## Training

if __name__ == '__main__':

    # here you put all actions which are performed only once before initializing the workers
    # for example, setting running arguments and experiment:

    args = parser.parse_args()

    # we can set here arguments that are considered as constant for this file (mnist_example.py)
    args.project_name = 'mnist'
    args.root_dir = '/home/shared/data/results'
    args.algorithm = 'MNISTAlgorithm'
    args.path_to_data = '/home/elad/projects/mnist'
    args.epoch_length = 100000
    # args.total_steps = 4000
    args.n_epochs = 10

    experiment = Experiment(args)

    # here we initialize the workers (can be single or multiple workers, depending on the configuration)
    alg = experiment.run(run_mnist)

    # ## Inference

    inference = alg('test')

    print('Test inference results:')
    for n, v in inference['metrics'].items():
        print(f'{n}:')
        print(v)

