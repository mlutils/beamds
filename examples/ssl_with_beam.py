import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd
import faiss                   # make faiss available
import umap
import seaborn as sns
from byol_pytorch import BYOL

import os
import sys
import matplotlib.pyplot as plt
from sklearn import svm

import torchvision.models as models
from torchvision import transforms
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pl_bolts.models.self_supervised import SimCLR
import lightgbm as lgb
import kornia
from kornia.augmentation.container import AugmentationSequential

from utils import add_beam_to_path
add_beam_to_path()

from src.beam import UniversalDataset, Experiment, Algorithm, beam_arguments, PackedFolds, batch_augmentation
from src.beam import tqdm


class STL10Dataset(UniversalDataset):

    def __init__(self, hparams, subset='unlabeled'):

        path = hparams.path_to_data
        seed = hparams.split_dataset_seed
        device = hparams.device

        super().__init__()

        self.normalize = True

        if subset == 'unlabeled':

            x = torch.tensor(np.fromfile(os.path.join(path, 'unlabeled_X.bin'), dtype=np.uint8))
            self.data = torch.reshape(x, (-1, 3, 96, 96)).permute(0, 1, 3, 2)
            self.labels = torch.LongTensor(len(self.data)).zero_()
            self.split(test=.2, seed=seed)

        else:

            x_train = torch.tensor(np.fromfile(os.path.join(path, 'train_X.bin'), dtype=np.uint8)).reshape((-1, 3, 96, 96)).permute(0, 1, 3, 2)
            x_test = torch.tensor(np.fromfile(os.path.join(path, 'test_X.bin'), dtype=np.uint8)).reshape((-1, 3, 96, 96)).permute(0, 1, 3, 2)
            y_train = torch.tensor(np.fromfile(os.path.join(path, 'train_y.bin'), dtype=np.uint8))
            y_test = torch.tensor(np.fromfile(os.path.join(path, 'test_y.bin'), dtype=np.uint8))

            self.data = PackedFolds({'train': x_train, 'test': x_test})
            self.labels = PackedFolds({'train': y_train, 'test': y_test})
            self.split(test=self.labels['test'].index, seed=seed)

        self.target_device = device

        size = 224
        s = 1
        self.n_augmentations = 2

        self.transform = transforms.Resize((size, size))

        self.gaussian_blur = batch_augmentation(transforms.GaussianBlur(kernel_size=(int(0.1 * size) // 2) * 2 + 1))
        self.augmentations = AugmentationSequential(kornia.augmentation.RandomResizedCrop(size=(size,size)),
                                                    kornia.augmentation.RandomHorizontalFlip(),
                                                    kornia.augmentation.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s, p=.8),
                                                    kornia.augmentation.RandomGrayscale(p=0.2))

    def getitem(self, index):

        x = self.data[index]

        if self.normalize:
            x = x.to(self.target_device).half() / 255

        x = self.transform(x)

        augmentations = [self.gaussian_blur(self.augmentations(x)) for _ in range(self.n_augmentations)]

        return {'x': x, 'y': self.labels[index], 'augmentations': augmentations}


class BeamCLR(Algorithm):

    def __init__(self, hparams):

        # choose your network
        encoder = models.resnet50(pretrained=False)
        layer = 'avgpool'
        return_nodes = {layer: 'features'}
        encoder = create_feature_extractor(encoder, return_nodes=return_nodes)

        d = 2048
        projection = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Linear(d, d), nn.BatchNorm1d(d))

        super().__init__(hparams, networks={'encoder': encoder, 'projection': projection})
        self.features = lambda x: self.networks['encoder'](x)['features'].view(len(x), -1)

    def iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x, y = sample['x'], sample['y']
        x_aug1, x_aug2 = sample['augmentations']

        encoder = self.networks['encoder']
        opt = self.optimizers['net']

        y_hat = net(x)
        loss = F.cross_entropy(y_hat, y, reduction='mean')

        opt.apply(loss, training=training)

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['ones'].append(x.sum(dim=-1).detach().cpu().numpy())
        results['scalar']['acc'].append(float((y_hat.argmax(1) == y).float().mean()))

        return results

    def preprocess_inference(self, results=None, **kwargs):
            self.dataset.normalize = True
            return results

    def inference(self, sample=None, results=None, subset=None, predicting=True, **kwargs):

        if predicting:
            x = sample
        else:
            x, y = sample['x'], sample['y']

        net = self.networks['net']
        # z = net(x)[0]
        z = net(x)

        if not predicting:
            return {'z': z, 'y': y}, results

        return z, results


def show_image(i, aug=False):

    dataset_labeled.normalize = True
    key = 'x_aug' if aug else 'x'
    im = np.array(dataset_labeled[i][1][key].permute(1, 2, 0))
    plt.imshow(im)
    plt.show()
    dataset_labeled.normalize = True