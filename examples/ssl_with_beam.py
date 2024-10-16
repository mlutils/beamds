import torch
from torch import nn
import numpy as np

import os
import torchvision.models as models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

import kornia
from kornia.augmentation.container import AugmentationSequential

from beam import UniversalDataset, Experiment, beam_arguments, PackedFolds, batch_augmentation
from beam import beam_boolean_feature
from beam.ssl import get_ssl_parser
from beam import beam_logger as logger

import requests
from collections import namedtuple

Similarities = namedtuple("Similarities", "index distance")

simclr_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'

pretrained_weights = {'convnext_base': {'weights': models.convnext.ConvNeXt_Base_Weights.DEFAULT},
                          'resnet18': {'weights': models.resnet.ResNet18_Weights.DEFAULT},
                      'convnext_tiny': {'weights': models.convnext.ConvNeXt_Tiny_Weights.DEFAULT},
                      'swin_s': {'weights': 'DEFAULT'},}

hidden_sizes = {'resnet18': 512, 'resnet50': 2048, 'convnext_base': 1024, 'simclr': 2048,
                'convnext_tiny': 768, 'swin_s': 768}


untrained_weights = {'weights': None}

if '1.13' not in torch.__version__:
    pretrained_weights = {k: {'pretrained': True} for k in pretrained_weights.keys()}
    untrained_weights = {'pretrained': None}


def my_default_configuration_by_cluster():

    ip = requests.get(r'http://jsonip.com').json()['ip']

    if '132.70.60' not in ip:
        data_path = '/home/shared/data/dataset/stl10/stl10_binary'
        logs_path = '/home/shared/data/results/'
    else:
        # data_path = '/home/dsi/elads/external/data/datasets/stl10/stl10_binary'
        # logs_path = '/home/dsi/elads/external/data/resutls'
        data_path = '/external/data/datasets/stl10/stl10_binary'
        logs_path = '/external/data/resutls'
        # data_path = '/localdata/elads/data/datasets/stl10/stl10_binary'
        # logs_path = '/localdata/elads/data/resutls'

    return data_path, logs_path


class ImageNetAugmented(UniversalDataset):

    def __init__(self, hparams=None, data=None, normalize=True):

        device = None
        if hparams is not None and 'device' in hparams:
            device = hparams.device

        super().__init__(hparams, data, target_device=device)

        self.normalize = normalize

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

        x = x.to(self.target_device)

        if self.normalize:
            x = x / 255

        x = self.transform(x)

        augmentations = [self.gaussian_blur(self.augmentations(x)) for _ in range(self.n_augmentations)]

        data = {'x': x, 'augmentations': augmentations}

        if hasattr(self, 'labels'):
            data['y'] = self.labels[index]

        return data


class STL10Dataset(ImageNetAugmented):

    def __init__(self, hparams, subset='unlabeled'):

        path = hparams.data_path
        seed = hparams.split_dataset_seed

        super().__init__(hparams)

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


class FeatureEncoder(nn.Module):

    def __init__(self, net, layer='avgpool'):
        super().__init__()

        return_nodes = {layer: 'features'}
        encoder = create_feature_extractor(net, return_nodes=return_nodes)

        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)['features'].view(len(x), -1)


def my_ssl_algorithm(algorithm):

    BaseClass = globals()[algorithm]

    class ImageNetSSL(BaseClass):

        def __init__(self, hparams):

            self.hparams = hparams
            self.model = hparams.net
            self.pretrained = hparams.pretrained
            self.layer = hparams.layer

            super().__init__(hparams)

        @property
        def p_dim(self):
            if self.hparams.p_dim is None:
                return hidden_sizes[self.hparams.net]
            return self.hparams.p_dim

        @property
        def h_dim(self):
            if self.hparams.layer == 'last':
                return 1000
            return hidden_sizes[self.hparams.net]

        def generate_labeled_set(self, *args, pretrained=None, **kwargs):
            """
            This function should be overridden by the child class. Its purpose is to generate a labeled test-set for the
            evaluation of the downstream task.
            @return: UniversalDataset
            """
            labeled_dataset = STL10Dataset(self.hparams, subset='labeled')
            labeled_dataset.normalize = True
            return labeled_dataset

        def generate_encoder(self, *args, pretrained=None, **kwargs):
            """
            This function should be overridden by the child class. Its purpose is to generate a fresh
            (untrained or pretrained) encoder.
            @param pretrained:
            @return: nn.Module
            """
            if pretrained is None:
                pretrained = self.pretrained

            if pretrained:
                argv = pretrained_weights[self.model]
            else:
                argv = untrained_weights

            if self.model == 'simclr':
                from pl_bolts.models.self_supervised import SimCLR as SimCLR_pretrained
                encoder = SimCLR_pretrained.load_from_checkpoint(simclr_path, strict=False)
            else:
                encoder = getattr(models, self.model)(**argv)
                if self.layer != 'last':
                    encoder = FeatureEncoder(encoder, layer=self.layer)

            return encoder

    return ImageNetSSL


def imagenet_ssl_parser():

    parser = get_ssl_parser()

    data_path, logs_path = my_default_configuration_by_cluster()
    beam_boolean_feature(parser, "pretrained", False, "Whether to load pretrained weights", metavar='hparam')

    parser.add_argument('--data-path', type=str, default=data_path, help='Path to the STL10 binaries')
    parser.add_argument('--logs-path', type=str, default=logs_path, help='Root directory for Logs and results')
    parser.add_argument('--n-ensembles', type=int, default=1, help='Size of the ensemble model')
    parser.add_argument('--temperature', type=float, default=1.0, metavar='hparam', help='Softmax temperature')
    parser.add_argument('--model', type=str, default='resnet18', metavar='hparam',
                        help='The encoder model (selected out of torchvision.models)')
    parser.add_argument('--layer', type=str, default='avgpool', metavar='hparam',
                        help='Name of the model layer which'
                             ' extracts the features')

    return parser


if __name__ == '__main__':

    hparams = beam_arguments(imagenet_ssl_parser(),
                         f"--project-name=beam_ssl --epoch-length=200 --reduction=mean --no-scale-epoch-by-batch-size",
                         f" --identifier=parallel_alternatives",
                         "--algorithm=BeamVICReg --n-gpus=4 --amp --model=swin_s --no-pretrained --layer=avgpool ",
                         f"--batch-size=96 --n-epochs=20000 --no-broadcast-buffers --lr-d=1e-3 --weight-decay=1e-5")

    experiment = Experiment(hparams)
    Alg = my_ssl_algorithm(hparams.algorithm)

    logger.info(f"BeamSSL model: {hparams.algorithm}")
    alg = experiment.fit(Alg, STL10Dataset)
