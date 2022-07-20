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

from src.beam import beam_arguments, Experiment, beam_algorithm_generator
from src.beam import UniversalDataset, UniversalBatchSampler
from src.beam import Algorithm
from src.beam import LinearNet
from src.beam import DataTensor, PackedFolds


# In[2]:


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


class MNISTAlgorithm(Algorithm):

    def __init__(self, hparams):

        # choose your network
        net = LinearNet(784, 256, 10, 4)
        super().__init__(hparams, networks=net)

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizers['net'].dense, gamma=0.99)
        self.stop_at = hparams.stop_at

    def early_stopping(self, results=None, epoch=None, **kwargs):
        acc = np.mean(results['validation']['scalar']['acc'])
        return acc > self.stop_at

    def postprocess_epoch(self, sample=None, results=None, epoch=None, subset=None, training=True, **kwargs):
        x, y = sample['x'], sample['y']

        results['images']['sample'] = x[:16].view(16, 1, 28, 28).data.cpu()

        if training:
            self.scheduler.step()
            results['scalar'][f'lr'] = self.optimizers['net'].dense.param_groups[0]['lr']

        return results

    def iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x, y = sample['x'], sample['y']

        x = x.view(len(x), -1)
        net = self.networks['net']
        opt = self.optimizers['net']

        y_hat = net(x)
        loss = F.cross_entropy(y_hat, y, reduction='mean')

        opt.apply(loss, training=training)

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['ones'].append(x.sum(dim=-1).detach().cpu().numpy())
        results['scalar']['acc'].append(float((y_hat.argmax(1) == y).float().mean()))

        return results

    def inference(self, sample=None, results=None, subset=None, predicting=True, **kwargs):

        if predicting:
            x = sample
        else:
            x, y = sample['x'], sample['y']

        x = x.view(len(x), -1)
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


def mnist_algorithm_generator(experiment):

    dataset = MNISTDataset(experiment.hparams)
    alg = MNISTAlgorithm(experiment.hparams)

    return beam_algorithm_generator(experiment, alg, dataset)


def run_mnist(rank, world_size, experiment):

    dataset = MNISTDataset(experiment.hparams)

    alg = MNISTAlgorithm(experiment.hparams)
    alg.load_dataset(dataset)
    alg.experiment = experiment

    # simulate input to the network
    x = next(alg.data_generator('validation'))[1]['x']
    x = x.view(len(x), -1)

    experiment.writer_control(enable=not (bool(rank)), networks=alg.get_networks(), inputs={'net': x})

    for i, results in enumerate(iter(alg)):
        experiment.save_model_results(results, alg, i,
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

    path_to_data = '/home/shared/data//dataset/mnist'
    root_dir = '/home/shared/data/results'

    args = beam_arguments(
        f"--project-name=mnist --root-dir={root_dir} --algorithm=MNISTAlgorithm --amp  --device=cpu   ",
        "--epoch-length=20000 --n-epochs=10 --clip=1 --parallel=1", path_to_data=path_to_data, stop_at=.97)

    experiment = Experiment(args)

    # train
    alg = experiment(mnist_algorithm_generator)

    # ## Inference
    inference = alg('test')

    print('Test inference results:')
    for n, v in inference['metrics'].items():
        print(f'{n}:')
        print(v)

