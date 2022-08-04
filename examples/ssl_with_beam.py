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
from src.beam import tqdm, beam_logger, get_beam_parser, beam_boolean_feature
from src.beam.model import soft_target_update, target_copy
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


class BeamSSL(Algorithm):

    def __init__(self, hparams, algorithm=None):

        logger.info(f"BeamSSL model: {algorithm}")
        networks = {}
        # choose your network

        encoder = getattr(models, hparams.model)(pretrained=hparams.pretrained)
        encoder = FeatureEncoder(encoder, layer=hparams.layer)
        networks['encoder'] = encoder

        hidden_sizes = {'resnet18': 512, 'resnet50': 2048}
        h = hidden_sizes[hparams.model]

        self.temperature = hparams.temperature

        if algorithm == 'byol':

            encoder = getattr(models, hparams.model)(pretrained=False)
            encoder = FeatureEncoder(encoder, layer=hparams.layer)
            networks['target_encoder'] = encoder
            networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, h))
            networks['target_projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
                                                          nn.Linear(h, h))
            networks['prediction'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, h))

        elif algorithm == 'simclr':
            networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h))

        elif algorithm == 'simsiam':
            networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h))
            networks['prediction'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, h))

        elif algorithm == 'barlowtwins':
            networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                                   nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                                   nn.ReLU(), nn.Linear(h, 2048))

        elif algorithm == 'moco':
            encoder = getattr(models, hparams.model)(pretrained=False)
            encoder = FeatureEncoder(encoder, layer=hparams.layer)
            networks['target_encoder'] = encoder
            networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, h))
            networks['target_projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
                                                          nn.Linear(h, h))
            networks['prediction'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, h))
        else:
            raise NotImplementedError


        self.labeled_dataset = STL10Dataset(hparams, subset='labeled')
        self.index_train_labeled = np.array(self.labeled_dataset.indices['train'])
        self.index_test_labeled = np.array(self.labeled_dataset.indices['test'])

        super().__init__(hparams, networks=networks)

    @staticmethod
    def get_parser():

        parser = get_beam_parser()

        path_to_data, root_dir = my_default_configuration_by_cluster()
        beam_boolean_feature(parser, "pretrained", False, "Whether to load pretrained weights", metavar='hparam')

        parser.add_argument('--path-to-data', type=str, default=path_to_data, help='Path to the STL10 binaries')
        parser.add_argument('--root-dir', type=str, default=root_dir, help='Root directory for Logs and results')
        parser.add_argument('--n-rules', type=int, default=128, help='Number of rules')
        parser.add_argument('--temperature', type=float, default=1.0, metavar='hparam', help='Softmax temperature')
        parser.add_argument('--tau', type=float, default=.99, metavar='hparam', help='Target update factor')
        parser.add_argument('--lambda-twins', type=float, default=0.005, metavar='hparam',
                            help='Off diagonal weight factor for Barlow Twins loss')
        parser.add_argument('--model', type=str, default='resnet18', metavar='hparam', help='The encoder model '
                                                                                            '(selected out of torchvision.models)')
        parser.add_argument('--layer', type=str, default='avgpool', metavar='hparam',
                            help='Name of the model layer which'
                                 ' extracts the features')

        return parser

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


class BarlowTwins(BeamSSL):

    def __init__(self, hparams):

        super().__init__(hparams, algorithm='barlowtwins')

    def iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x_aug1, x_aug2 = sample['augmentations']

        encoder = self.networks['encoder']
        opt_e = self.optimizers['encoder']

        projection = self.networks['projection']
        opt_p = self.optimizers['projection']

        z1 = projection(encoder(x_aug1))
        z2 = projection(encoder(x_aug2))

        z1 = (z1 - z1.mean(dim=0, keepdim=True)) / (z1.std(dim=0, keepdim=True) + 1e-6)
        z2 = (z2 - z2.mean(dim=0, keepdim=True)) / (z2.std(dim=0, keepdim=True) + 1e-6)

        b, d = z1.shape
        corr = (z1.T @ z2) / b

        I = torch.eye(d, device=corr.device)
        corr_diff = (corr - I) ** 2

        invariance = torch.diag(corr_diff).sum()
        redundancy = (corr_diff * (1 - I)).sum()

        loss = invariance + self.hparams.lambda_twins * redundancy

        if training:
            self.apply(loss, optimizers=[opt_e, opt_p])

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['invariance'].append(float(invariance))
        results['scalar']['redundancy'].append(float(redundancy))

        return results


class SimCLR(BeamSSL):

    def __init__(self, hparams):

        super().__init__(hparams, algorithm='simclr')

    def iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x_aug1, x_aug2 = sample['augmentations']

        encoder = self.networks['encoder']
        opt_e = self.optimizers['encoder']

        projection = self.networks['projection']
        opt_p = self.optimizers['projection']

        z1 = projection(encoder(x_aug1))
        z2 = projection(encoder(x_aug2))

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


class SimSiam(BeamSSL):

    def __init__(self, hparams):

        super().__init__(hparams, algorithm='simsiam')

    @staticmethod
    def simsiam_loss(p, z):

        z = z.detach()
        z = z / torch.norm(z, dim=1, keepdim=True)
        p = p / torch.norm(p, dim=1, keepdim=True)
        return 2 - (z * p).sum(dim=1)

    def iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x_aug1, x_aug2 = sample['augmentations']

        encoder = self.networks['encoder']
        opt_e = self.optimizers['encoder']

        projection = self.networks['projection']
        opt_proj = self.optimizers['projection']

        prediction = self.networks['prediction']
        opt_pred = self.optimizers['prediction']

        z1 = projection(encoder(x_aug1))
        z2 = projection(encoder(x_aug2))

        p1 = prediction(z1)
        p2 = prediction(z2)

        d1 = SimSiam.simsiam_loss(p1, z2)
        d2 = SimSiam.simsiam_loss(p2, z1)

        loss = d1.mean() / 2 + d2.mean() / 2

        if training:
            self.apply(loss, optimizers=[opt_e, opt_proj, opt_pred])

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))

        return results


class BYOL(BeamSSL):

    def __init__(self, hparams):

        super().__init__(hparams, algorithm='byol')

    def iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x_aug1, x_aug2 = sample['augmentations']

        encoder = self.networks['encoder']
        projection = self.networks['projection']
        prediction = self.networks['prediction']

        opt_e = self.optimizers['encoder']
        opt_proj = self.optimizers['projection']
        opt_pred = self.optimizers['prediction']

        z1 = projection(encoder(x_aug1))
        p1 = prediction(z1)

        target_encoder = self.networks['target_encoder']
        target_projection = self.networks['target_projection']

        with torch.no_grad():
            z2 = target_projection(target_encoder(x_aug2))

        z2 = z2 / torch.norm(z2, dim=1, keepdim=True)
        p1 = p1 / torch.norm(p1, dim=1, keepdim=True)

        loss = torch.pow(p1 - z2, 2).sum(dim=1).mean()

        if training:

            soft_target_update(encoder, target_encoder, self.hparams.tau)
            soft_target_update(projection, target_projection, self.hparams.tau)

            self.apply(loss, optimizers=[opt_e, opt_proj, opt_pred])

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))

        return results


if __name__ == '__main__':

    args = beam_arguments(BeamSSL.get_parser(),
        f"--project-name=beam_ssl --algorithm=SimCLR --device=0 --amp "
        f"--batch-size=256 --n-epochs=100")

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