import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math

import os
import torchvision.models as models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

import kornia
from kornia.augmentation.container import AugmentationSequential
from pl_bolts.models.self_supervised import SimCLR as SimCLR_pretrained
import torch_tensorrt

from examples.example_utils import add_beam_to_path
add_beam_to_path()

from src.beam import UniversalDataset, Experiment, Algorithm, beam_arguments, PackedFolds, batch_augmentation
from src.beam import tqdm, beam_logger, get_beam_parser, beam_boolean_feature, BeamOptimizer
from src.beam.model import soft_target_update, target_copy, reset_network, copy_network, BeamEnsemble, beam_weights_initializer
import lightgbm as lgb
import requests
from torch.nn.utils import spectral_norm

simclr_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'

if '1.13'  in torch.__version__:
    pretrained_weights = {'convnext_base': {'weights': models.convnext.ConvNeXt_Base_Weights.DEFAULT},
                          'resnet18': {'weights': models.resnet.ResNet18_Weights.DEFAULT}}
else:
    pretrained_weights = {'convnext_base': {'pretrained': True},
                          'resnet18': {'pretrained': True}}


def my_default_configuration_by_cluster():

    ip = requests.get(r'http://jsonip.com').json()['ip']

    if '132.70.60' not in ip:
        path_to_data = '/home/shared/data/dataset/stl10/stl10_binary'
        root_dir = '/home/shared/data/results/'
    else:
        path_to_data = '/localdata/elads/data/datasets/stl10/stl10_binary'
        root_dir = '/localdata/elads/data/resutls'

    return path_to_data, root_dir


class ImageNetAugmented(UniversalDataset):

    def __init__(self, hparams=None, data=None):

        device = None
        if hparams is not None and 'device' in hparams:
            device = hparams.device

        super().__init__(hparams, data, target_device=device)

        self.normalize = True

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

        path = hparams.path_to_data
        seed = hparams.split_dataset_seed

        super().__init__(hparams)

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


class FeatureEncoder(nn.Module):

    def __init__(self, net, layer='avgpool'):
        super().__init__()

        return_nodes = {layer: 'features'}
        encoder = create_feature_extractor(net, return_nodes=return_nodes)

        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)['features'].view(len(x), -1)


class BeamSSL(Algorithm):

    def __init__(self, hparams):

        networks = {}
        # choose your network

        hidden_sizes = {'resnet18': 512, 'resnet50': 2048, 'convnext_base': 1024, 'simclr': 2048}

        if hparams.layer == 'last':
            self.h_dim = 1000
        else:
            self.h_dim = hidden_sizes[hparams.model]

        self.p_dim = hparams.p_dim

        self.temperature = hparams.temperature
        self.model = hparams.model
        self.pretrained = hparams.pretrained
        self.layer = hparams.layer

        networks['encoder'] = self.generate_encoder()

        self.labeled_dataset = STL10Dataset(hparams, subset='labeled')
        self.index_train_labeled = np.array(self.labeled_dataset.indices['train'])
        self.index_test_labeled = np.array(self.labeled_dataset.indices['test'])

        self.logger = beam_logger()

        super().__init__(hparams, networks=networks)

    def generate_encoder(self, pretrained=None):

        if pretrained is None:
            pretrained = self.pretrained

        if pretrained:
            weights = pretrained_weights[self.model]
        else:
            weights = {'weights': None}

        if self.model == 'simclr':
            encoder = SimCLR_pretrained.load_from_checkpoint(simclr_path, strict=False)
        else:
            encoder = getattr(models, self.model)(**weights)
            if self.layer != 'last':
                encoder = FeatureEncoder(encoder, layer=self.layer)

        return encoder

    @staticmethod
    def get_parser():

        parser = get_beam_parser()

        path_to_data, root_dir = my_default_configuration_by_cluster()
        beam_boolean_feature(parser, "pretrained", False, "Whether to load pretrained weights", metavar='hparam')

        parser.add_argument('--path-to-data', type=str, default=path_to_data, help='Path to the STL10 binaries')
        parser.add_argument('--similarity', type=str, metavar='hparam', default='cosine', help='Similarity distance in UniversalSSL')
        parser.add_argument('--root-dir', type=str, default=root_dir, help='Root directory for Logs and results')
        parser.add_argument('--n-rules', type=int, default=128, help='Number of rules')
        parser.add_argument('--n-ensembles', type=int, default=16, help='Size of the ensemble model')
        parser.add_argument('--p-dim', type=int, default=2048, help='Prediction/Projection output dimension')
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

    def preprocess_inference(self, results=None, augmentations=0, dataset=None, **kwargs):
            self.dataset.normalize = True

            # self.traced_encoder = torch.jit.trace(self.networks['encoder'],
            #                                torch.randn((1, 3, 224, 224)).to(self.hparams.device))
            # torch.jit.save(self.traced_encoder, "/tmp/model.jit.pt")

            # self.traced_encoder = torch_tensorrt.compile(self.networks['encoder'],
            #                                    inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
            #                                    enabled_precisions={torch_tensorrt.dtype.half}  # Run with FP16
            #                                    )

            # self.traced_encoder = torch.jit.script(self.networks['encoder'])

            if augmentations > 0 and dataset is not None:
                results['aux']['org_n_augmentations'] = dataset.n_augmentations
                dataset.n_augmentations = augmentations

            return results

    def postprocess_inference(self, sample=None, results=None, subset=None, dataset=None, **kwargs):

        if 'aux' in results and 'org_n_augmentations' in results['aux'] and dataset is not None:
            dataset.n_augmentations = results['aux']['org_n_augmentations']

        return results

    def evaluate_downstream_task(self, z, y):

        train_data = lgb.Dataset(z[self.index_train_labeled], label=y[self.index_train_labeled] - 1)
        validation_data = lgb.Dataset(z[self.index_test_labeled], label=y[self.index_test_labeled] - 1)

        num_round = 40
        param = {'objective': 'multiclass',
                 'num_leaves': 31,
                 'max_depth': 4,
                 'gpu_device_id': 1,
                 'verbosity': -1,
                 'metric': ['multi_error', 'multiclass'],
                 'num_class': 10}

        return lgb.train(param, train_data, num_round, valid_sets=[validation_data])

    def postprocess_epoch(self, results=None, training=None, epoch=None, **kwargs):

            if not training and not epoch % 1:
                self.labeled_dataset.normalize = True

                self.logger.info("Evaluating the downstream task")
                features = self.evaluate(self.labeled_dataset)
                z = features.values['h'].detach().cpu().numpy()
                y = features.values['y'].detach().cpu().numpy()

                bst = self.evaluate_downstream_task(z, y)

                results['scalar']['encoder_acc'] = 1 - bst.best_score['valid_0']['multi_error']
                results['scalar']['encoder_loss'] = bst.best_score['valid_0']['multi_logloss']

                if 'z' in features.values:

                    z = features.values['z'].detach().cpu().numpy()
                    bst = self.evaluate_downstream_task(z, y)

                    results['scalar']['projection_acc'] = 1 - bst.best_score['valid_0']['multi_error']
                    results['scalar']['projection_loss'] = bst.best_score['valid_0']['multi_logloss']

                    if 'p' in features.values:

                        z = features.values['p'].detach().cpu().numpy()
                        bst = self.evaluate_downstream_task(z, y)

                        results['scalar']['prediction_acc'] = 1 - bst.best_score['valid_0']['multi_error']
                        results['scalar']['prediction_loss'] = bst.best_score['valid_0']['multi_logloss']

            return results

    def inference(self, sample=None, results=None, subset=None, predicting=True,
                  projection=True, prediction=True, augmentations=0, **kwargs):

        data = {}
        if predicting:
            x = sample
        else:
            x = sample['x']

            if 'y' in sample:
                data['y'] = sample['y']

        h = self.networks['encoder'](x)
        data['h'] = h

        if 'projection' in self.networks and projection:
            z = self.networks['projection'](h)
            data['z'] = z

        if 'prediction' in self.networks and prediction:
            p = self.networks['prediction'](z)
            data['p'] = p

        if type(sample) is dict and 'augmentations' in sample and augmentations:
            representations = []
            for a in sample['augmentations']:
                representations.append(self.networks['encoder'](a))

            representations = torch.stack(representations)

            mu = representations.mean(dim=0)
            std = representations.std(dim=0)

            data['mu'] = mu
            data['std'] = std

        return data, results


class BeamBarlowTwins(BeamSSL):

    def __init__(self, hparams):
        super().__init__(hparams)

        h = self.h_dim
        p = self.p_dim

        projection = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, p))

        discriminator = nn.Sequential(spectral_norm(nn.Linear(h, h)),
                                   nn.ReLU(), spectral_norm(nn.Linear(h, h)), nn.ReLU(), nn.Linear(h, 1))

        self.add_networks_and_optmizers(networks={'projection': projection, 'discriminator': discriminator})

        ensemble = BeamEnsemble(self.generate_encoder, n_ensembles=hparams.n_ensembles)
        ensemble.set_optimizers(BeamOptimizer.prototype(dense_args={'lr': self.hparams.lr_dense,
                                                                    'weight_decay': self.hparams.weight_decay,
                                                                    'betas': (self.hparams.beta1, self.hparams.beta2),
                                                                    'eps': self.hparams.eps}))

        # self.add_networks_and_optmizers(networks=ensemble, name='encoder', build_optimizers=False)
        self.add_networks_and_optmizers(networks=ensemble, name='encoder')

        # beam_weights_initializer(self.networks['projection'])
        # beam_weights_initializer(self.networks['discriminator'], method='orthogonal')

    def iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x_aug1, x_aug2 = sample['augmentations']

        encoder = self.networks['encoder']
        opt_e = self.optimizers['encoder']

        projection = self.networks['projection']
        opt_p = self.optimizers['projection']

        index = torch.randperm(len(encoder))
        z1 = projection(encoder(x_aug1, index=index[0]))
        z2 = projection(encoder(x_aug2, index=index[0]))

        z1 = (z1 - z1.mean(dim=0, keepdim=True)) / (z1.std(dim=0, keepdim=True) + 1e-6)
        z2 = (z2 - z2.mean(dim=0, keepdim=True)) / (z2.std(dim=0, keepdim=True) + 1e-6)

        b, d = z1.shape
        corr = (z1.T @ z2) / b

        I = torch.eye(d, device=corr.device)
        corr_diff = (corr - I) ** 2

        invariance = torch.diag(corr_diff)
        redundancy = (corr_diff * (1 - I)).sum(dim=-1)

        loss = invariance + self.hparams.lambda_twins * redundancy

        # opt_1 = encoder.optimizers[index[0]]
        # opt_2 = encoder.optimizers[index[1]]

        # loss = self.apply(loss, training=training, optimizers=[opt_1, opt_2, opt_p])
        loss = self.apply(loss, training=training, optimizers=[opt_e, opt_p])

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['invariance'].append(float(invariance.mean()))
        results['scalar']['redundancy'].append(float(redundancy.mean()))

        return results


class UniversalSSL(BeamSSL):

    def __init__(self, hparams):

        super().__init__(hparams)

        networks = {}
        h = self.h_dim
        p = self.p_dim

        networks['target_encoder'] = self.generate_encoder(pretrained=False)
        reset_network(networks['target_encoder'])

        networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, p))

        networks['target_projection'] = copy_network(networks['projection'])
        reset_network(networks['target_projection'])

        networks['prediction'] = nn.Sequential(nn.Linear(p, p), nn.BatchNorm1d(p), nn.ReLU(), nn.Linear(p, p))
        self.add_networks_and_optmizers(networks=networks)

    def iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x_aug1, x_aug2 = sample['augmentations']

        encoder = self.networks['encoder']
        projection = self.networks['projection']
        prediction = self.networks['prediction']

        opt_e = self.optimizers['encoder']
        opt_proj = self.optimizers['projection']
        opt_pred = self.optimizers['prediction']

        target_encoder = self.networks['target_encoder']
        target_projection = self.networks['target_projection']

        z1 = prediction(projection(encoder(x_aug1)))
        z2 = target_projection(target_encoder(x_aug2))

        b, d = z1.shape
        z = torch.cat([z1, z2], dim=1).view(-1, d)

        z_norm = torch.norm(z, dim=1, keepdim=True)

        if self.hparams.similarity == 'cosine':
            s = (z @ z.T) / (z_norm @ z_norm.T)
        elif self.hparams.similarity == 'l2':
            z_l2_squared = torch.pow(z, 2).sum(dim=1, keepdim=True)
            s = - (z_l2_squared - 2 * (z @ z.T) + z_l2_squared.T)
        else:
            raise NotImplementedError

        s = s * (1 - torch.eye(2 * b, 2 * b, device=s.device)) / (self.temperature * d)

        logsumexp = torch.logsumexp(s[::2], dim=1)
        s_couple = torch.diag(s, diagonal=1)[::2]

        loss = - s_couple + logsumexp
        loss = self.apply(loss, training=training, optimizers=[opt_e, opt_proj, opt_pred])

        if training:

            soft_target_update(encoder, target_encoder, self.hparams.tau)
            soft_target_update(projection, target_projection, self.hparams.tau)

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['acc'].append(float((s_couple >= s[::2].max(dim=1).values).float().mean()))

        return results


class BarlowTwins(BeamSSL):

    def __init__(self, hparams):

        super().__init__(hparams)

        networks = {}
        h = self.h_dim
        p = self.p_dim

        networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, p))

        self.add_networks_and_optmizers(networks=networks)

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

        invariance = torch.diag(corr_diff)
        redundancy = (corr_diff * (1 - I)).sum(dim=-1)

        loss = invariance + self.hparams.lambda_twins * redundancy
        loss = self.apply(loss, training=training, optimizers=[opt_e, opt_p])

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['invariance'].append(float(invariance.mean()))
        results['scalar']['redundancy'].append(float(redundancy.mean()))

        return results


class SimCLR(BeamSSL):

    def __init__(self, hparams):
        super().__init__(hparams)

        networks = {}
        h = self.h_dim
        p = self.p_dim

        networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                                   nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                                   nn.ReLU(), nn.Linear(h, p))

        self.add_networks_and_optmizers(networks=networks)

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
        loss = self.apply(loss, training=training, optimizers=[opt_e, opt_p])

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['acc'].append(float((s_couple >= s[::2].max(dim=1).values).float().mean()))

        return results


class SimSiam(BeamSSL):

    def __init__(self, hparams):
        super().__init__(hparams)

        networks = {}
        h = self.h_dim
        p = self.p_dim

        networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, p))

        networks['prediction'] = nn.Sequential(nn.Linear(p, p), nn.BatchNorm1d(p), nn.ReLU(), nn.Linear(p, p))
        self.add_networks_and_optmizers(networks=networks)

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

        loss = (d1 + d2) / 2
        loss = self.apply(loss, training=training, optimizers=[opt_e, opt_proj, opt_pred])

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))

        return results


class BYOL(BeamSSL):

    def __init__(self, hparams):

        super().__init__(hparams)

        networks = {}
        h = self.h_dim
        p = self.p_dim

        networks['target_encoder'] = self.generate_encoder(pretrained=False)
        reset_network(networks['target_encoder'])

        networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, p))

        networks['target_projection'] = copy_network(networks['projection'])
        reset_network(networks['target_projection'])

        networks['prediction'] = nn.Sequential(nn.Linear(p, p), nn.BatchNorm1d(p), nn.ReLU(), nn.Linear(p, p))
        self.add_networks_and_optmizers(networks=networks)

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

        loss = torch.pow(p1 - z2, 2).sum(dim=1)
        loss = self.apply(loss, training=training, optimizers=[opt_e, opt_proj, opt_pred])

        if training:

            soft_target_update(encoder, target_encoder, self.hparams.tau)
            soft_target_update(projection, target_projection, self.hparams.tau)

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))

        return results


if __name__ == '__main__':

    args = beam_arguments(BeamSSL.get_parser(),
        f"--project-name=beam_ssl --algorithm=SimCLR --device=0 --amp "
        f"--batch-size=256 --n-epochs=200")

    logger = beam_logger()

    experiment = Experiment(args)
    Alg = globals()[experiment.hparams.algorithm]

    logger.info(f"BeamSSL model: {experiment.hparams.algorithm}")
    alg = experiment.fit(Alg, STL10Dataset)

    # ## Inference
    inference = alg('test')

    logger.info('Test inference results:')
    for n, v in inference.statistics['metrics'].items():
        logger.info(f'{n}:')
        logger.info(v)