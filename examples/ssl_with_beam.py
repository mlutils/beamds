import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

import os
import torchvision.models as models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

import kornia
from kornia.augmentation.container import AugmentationSequential

from utils import add_beam_to_path
add_beam_to_path()

from src.beam import UniversalDataset, Experiment, Algorithm, beam_arguments, PackedFolds, batch_augmentation
from src.beam import tqdm, beam_logger, get_beam_parser
import lightgbm as lgb
import socket


def my_default_configuration_by_cluster():

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]

    if '172.17' in ip:
        path_to_data = '/home/shared/data/dataset/stl10/stl10_binary'
        root_dir = '/home/shared/data/results/'
    else:
        path_to_data = '/localdata/elads/data/datasets/stl10/stl10_binary'
        root_dir = '/localdata/elads/data/resutls'

    return path_to_data, root_dir


class STL10Dataset(UniversalDataset):

    def __init__(self, hparams, subset='unlabeled'):

        path = hparams.path_to_data
        seed = hparams.split_dataset_seed
        device = hparams.device
        self.half = hparams.half

        super().__init__(target_device=device)

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
            x = x.to(self.target_device) / 255

        x = self.transform(x)

        augmentations = [self.gaussian_blur(self.augmentations(x)) for _ in range(self.n_augmentations)]

        return {'x': x, 'y': self.labels[index], 'augmentations': augmentations}


class FeatureEncoder(nn.Module):

    def __init__(self, net, layer='avgpool'):
        super().__init__()

        return_nodes = {layer: 'features'}
        encoder = create_feature_extractor(net, return_nodes=return_nodes)

        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)['features'].view(len(x), -1)


class BeamCLR(Algorithm):

    def __init__(self, hparams):

        # choose your network
        encoder = FeatureEncoder(models.resnet18(pretrained=False), layer='avgpool')

        self.h = 2048
        self.temperature = 1
        projection = nn.Sequential(nn.LazyLinear(self.h), nn.BatchNorm1d(self.h), nn.ReLU(),
                                   nn.Linear(self.h, self.h), nn.BatchNorm1d(self.h))

        self.labeled_dataset = STL10Dataset(hparams, subset='labeled')
        self.index_train_labeled = np.array(self.labeled_dataset.indices['train'])
        self.index_test_labeled = np.array(self.labeled_dataset.indices['test'])

        super().__init__(hparams, networks={'encoder': encoder, 'projection': projection})

    def iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        # x, y = sample['x'], sample['y']
        x_aug1, x_aug2 = sample['augmentations']

        encoder = self.networks['encoder']
        opt_e = self.optimizers['encoder']

        projection = self.networks['projection']
        opt_p = self.optimizers['projection']

        h1 = encoder(x_aug1)
        h2 = encoder(x_aug2)

        z1 = projection(h1)
        z2 = projection(h2)

        b, h = z1.shape
        z = torch.cat([z1, z2], dim=1).view(-1, h)

        z_norm = torch.norm(z, dim=1, keepdim=True)

        s = (z @ z.T) / (z_norm @ z_norm.T)
        s = s * (1 - torch.eye(2 * b, 2 * b, device=s.device)) / self.temperature

        logsumexp = torch.logsumexp(s[::2], dim=1)
        s_couple = torch.diag(s, diagonal=1)[::2]

        loss = - s_couple + logsumexp

        loss = loss.mean()

        if training:
            self.apply(loss, optimizers=[opt_e, opt_p])

        # add scalar measurements

        results['scalar']['loss'].append(float(loss))
        results['scalar']['acc'].append(float((s_couple >= s[::2].max(dim=1).values).float().mean()))

        return results

    def preprocess_inference(self, results=None, **kwargs):
            self.dataset.normalize = True
            return results

    def postprocess_epoch(self, results=None, training=None, epoch=None, **kwargs):

            if not training and not epoch % 1:
                self.labeled_dataset.normalize = True

                logger.info("Evaluating the downstream task")
                features = self.evaluate(self.labeled_dataset)
                z = features.values['z'].detach().cpu().numpy()
                y = features.values['y'].detach().cpu().numpy()

                train_data = lgb.Dataset(z[self.index_train_labeled], label=y[self.index_train_labeled] - 1)
                validation_data = lgb.Dataset(z[self.index_test_labeled], label=y[self.index_test_labeled] - 1)

                num_round = 30
                param = {'objective': 'multiclass',
                         'num_leaves': 31,
                         'max_depth': 4,
                         'gpu_device_id': 1,
                         'verbosity': -1,
                         'metric': ['multi_error', 'multiclass'],
                         'num_class': 10}
                bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])

                results['scalar']['classification_acc'] = 1 - bst.best_score['valid_0']['multi_error']
                results['scalar']['classification_loss'] = bst.best_score['valid_0']['multi_logloss']

            return results

    def inference(self, sample=None, results=None, subset=None, predicting=True, **kwargs):

        if predicting:
            x = sample
        else:
            x, y = sample['x'], sample['y']

        encoder = self.networks['encoder']
        z = encoder(x)

        if not predicting:
            return {'z': z, 'y': y}, results

        return z, results


def get_ssl_parser():

    parser = get_beam_parser()

    path_to_data, root_dir = my_default_configuration_by_cluster()
    parser.add_argument('--path-to-data', type=str, default=path_to_data, help='Path to the STL10 binaries')
    parser.add_argument('--root-dir', type=str, default=root_dir, help='Root directory for Logs and results')

    return parser


if __name__ == '__main__':

    args = beam_arguments(
        f"--project-name=beam_ssl --algorithm=BeamCLR --device=0 --amp "
        f"--batch-size=256 --reload",
        "--n-epochs=100")

    logger = beam_logger()

    experiment = Experiment(args)
    Alg = globals()[experiment.hparams.algorithm]

    alg = experiment.fit(Alg, STL10Dataset)

    # ## Inference
    inference = alg('test')

    logger.info('Test inference results:')
    for n, v in inference.statistics['metrics'].items():
        logger.info(f'{n}:')
        logger.info(v)