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
from src.beam.model import soft_target_update, target_copy, reset_network, copy_network, BeamEnsemble
from src.beam.model import beam_weights_initializer, freeze_network_params, free_network_params
from src.beam.utils import to_numpy, pretty_format_number

import lightgbm as lgb
import requests
from torch.nn.utils import spectral_norm
import faiss
# working with faiss and torch
import faiss.contrib.torch_utils
from sklearn.manifold import TSNE
import umap

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
        path_to_data = '/home/shared/data/dataset/stl10/stl10_binary'
        root_dir = '/home/shared/data/results/'
    else:
        # path_to_data = '/home/dsi/elads/external/data/datasets/stl10/stl10_binary'
        # root_dir = '/home/dsi/elads/external/data/resutls'
        path_to_data = '/external/data/datasets/stl10/stl10_binary'
        root_dir = '/external/data/resutls'
        # path_to_data = '/localdata/elads/data/datasets/stl10/stl10_binary'
        # root_dir = '/localdata/elads/data/resutls'

    return path_to_data, root_dir


class Similarity(object):

    def __init__(self, index=None, d=None, expected_population=int(1e6),
                 metric='l2', training_device='cpu', inference_device='cpu', ram_footprint=2**8*int(1e9),
                 gpu_footprint=24*int(1e9), exact=False, nlists=None, M=None,
                 reducer='umap'):

        '''
        To Choose an index, follow https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        @param d:
        @param expected_population:
        @param metric:
        @param ram_size:
        @param gpu_size:
        @param exact_results:
        @param reducer:
        '''

        metrics = {'l2': faiss.METRIC_L2, 'l1': faiss.METRIC_L1, 'linf': faiss.METRIC_Linf,
                   'cosine': faiss.METRIC_INNER_PRODUCT, 'ip': faiss.METRIC_INNER_PRODUCT,
                   'js': faiss.METRIC_JensenShannon}
        metric = metrics[metric]
        self.normalize = False
        if metric == 'cosine':
            self.normalize = True

        # choosing nlists: https://github.com/facebookresearch/faiss/issues/112,
        #  https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        if nlists is None:
            if expected_population <= int(1e6):
                # You will need between 30*K and 256*K vectors for training (the more the better)
                nlists = int(8 * math.sqrt(expected_population))
            elif expected_population > int(1e6) and expected_population <= int(1e7):
                nlists = 2 ** 16
            elif expected_population > int(1e7) and expected_population <= int(1e8):
                nlists = 2 ** 18
            else:
                nlists = 2 ** 20

        if index is not None:
            if inference_device == 'cpu':

                if exact:
                    logger.info(f"Using Flat Index. Expected RAM footprint is "
                                f"{pretty_format_number(4 * d * expected_population / int(1e6))} MB")
                    index = faiss.IndexFlat(d, metric)
                else:
                    if M is None:
                        M = 2 ** np.arange(2, 7)[::-1]
                        footprints = (d * 4 + M * 8) * expected_population
                        M_ind = np.where(footprints < ram_footprint)[0]
                        if len(M_ind):
                            M = M[M_ind[0]]
                    if M is not None:
                        logger.info(f"Using HNSW{M}. Expected RAM footprint is "
                                    f"{pretty_format_number(footprints[M_ind[0]] / int(1e6))} MB")
                        index = faiss.IndexHNSWFlat(d, M)
                    else:
                        logger.info(f"Using OPQ16_64,IVF{nlists},PQ8 Index")
                        index = faiss.index_factory(d, f'OPQ16_64,IVF{nlists},PQ8')

            else:

                res = faiss.StandardGpuResources()
                if exact:
                    config = faiss.GpuIndexFlatConfig()
                    config.device = inference_device
                    logger.info(f"Using GPUFlat Index. Expected GPU-RAM footprint is "
                                f"{pretty_format_number(4 * d * expected_population / int(1e6))} MB")

                    index = faiss.GpuIndexFlat(res, d, metric, config)
                else:

                    if (4 * d + 8) * expected_population <= gpu_footprint:
                        logger.info(f"Using GPUIndexIVFFlat Index. Expected GPU-RAM footprint is "
                                    f"{pretty_format_number((4 * d + 8) * expected_population / int(1e6))} MB")
                        config = faiss.GpuIndexIVFFlatConfig()
                        config.device = inference_device
                        index = faiss.GpuIndexIVFFlat(res, d,  nlists, M, 8, faiss.METRIC_L2, config)
                    else:

                        if M is None:
                            M = 2 ** np.arange(2, 7)[::-1]
                            footprints = (M + 8) * expected_population
                            M_ind = np.where(footprints < gpu_footprint)[0]
                            if len(M_ind):
                                M = M[M_ind[0]]
                        if M is not None:
                            logger.info(f"Using GPUIndexIVFFlat Index. Expected GPU-RAM footprint is "
                                        f"{pretty_format_number((M + 8) * expected_population / int(1e6))} MB")

                            config = faiss.GpuIndexIVFPQConfig()
                            config.device = inference_device
                            index = faiss.GpuIndexIVFPQ(res, d,  nlists, M, 8, faiss.METRIC_L2, config)
                        else:
                            logger.info(f"Using OPQ16_64,IVF{nlists},PQ8 Index")
                            index = faiss.index_factory(d, f'OPQ16_64,IVF{nlists},PQ8')
                            index = faiss.index_cpu_to_gpu(res, inference_device, index)

        if index is None:
            logger.error("Cannot find suitable index type")
            raise Exception

        self.index = index

        self.training_index = None
        res = faiss.StandardGpuResources()
        if training_device != 'cpu' and inference_device == 'cpu':
            self.training_index = faiss.index_cpu_to_gpu(res, training_device, index)

        if reducer == 'umap':
            self.reducer = umap.UMAP()
        elif reducer == 'tsne':
            self.reducer = TSNE()
        else:
            raise NotImplementedError

    def add(self, z, train=False):

        if train or not self.index.is_trained:
            self.index.train(z)
        self.index.add(z)

    def most_similar(self, zi, n=1):
        D, I = self.index.search(zi, n)
        return D, I

    def __len__(self):
        return self.index.ntotal

    def reduce(self, z):
        return self.reducer.fit_transform(z)


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

        if hparams.layer == 'last':
            self.h_dim = 1000
        else:
            self.h_dim = hidden_sizes[hparams.model]

        if hparams.p_dim is None:
            self.p_dim = hidden_sizes[hparams.model]
        else:
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
            argv = pretrained_weights[self.model]
        else:
            argv = untrained_weights

        if self.model == 'simclr':
            encoder = SimCLR_pretrained.load_from_checkpoint(simclr_path, strict=False)
        else:
            encoder = getattr(models, self.model)(**argv)
            if self.layer != 'last':
                encoder = FeatureEncoder(encoder, layer=self.layer)

        return encoder

    @staticmethod
    def get_parser():

        parser = get_beam_parser()

        path_to_data, root_dir = my_default_configuration_by_cluster()
        beam_boolean_feature(parser, "pretrained", False, "Whether to load pretrained weights", metavar='hparam')

        parser.add_argument('--path-to-data', type=str, default=path_to_data, help='Path to the STL10 binaries')
        parser.add_argument('--similarity', type=str, metavar='hparam', default='cosine',
                            help='Similarity distance in UniversalSSL')
        parser.add_argument('--root-dir', type=str, default=root_dir, help='Root directory for Logs and results')
        parser.add_argument('--n-discriminator-steps', type=int, default=1, help='Number of discriminator steps')
        parser.add_argument('--n-ensembles', type=int, default=1, help='Size of the ensemble model')
        parser.add_argument('--p-dim', type=int, default=None, help='Prediction/Projection output dimension')
        parser.add_argument('--temperature', type=float, default=1.0, metavar='hparam', help='Softmax temperature')
        parser.add_argument('--var-eps', type=float, default=0.0001, metavar='hparam', help='Std epsilon in VICReg')
        parser.add_argument('--lambda-vicreg', type=float, default=25., metavar='hparam',
                            help='Lambda weight in VICReg')
        parser.add_argument('--mu-vicreg', type=float, default=25., metavar='hparam', help='Mu weight in VICReg')
        parser.add_argument('--nu-vicreg', type=float, default=1., metavar='hparam', help='Nu weight in VICReg')
        parser.add_argument('--lambda-mean-vicreg', type=float, default=20., metavar='hparam',
                            help='lambda-mean weight in BeamVICReg')
        parser.add_argument('--tau', type=float, default=.99, metavar='hparam', help='Target update factor')
        parser.add_argument('--lambda-twins', type=float, default=0.005, metavar='hparam',
                            help='Off diagonal weight factor for Barlow Twins loss')
        parser.add_argument('--lambda-disc', type=float, default=1., metavar='hparam',
                            help='Discriminator loss for the encoder training')
        parser.add_argument('--model', type=str, default='resnet18', metavar='hparam',
                            help='The encoder model (selected out of torchvision.models)')
        parser.add_argument('--layer', type=str, default='avgpool', metavar='hparam',
                            help='Name of the model layer which'
                                 ' extracts the features')

        return parser

    def preprocess_inference(self, results=None, augmentations=0, dataset=None, **kwargs):

            self.dataset.normalize = True

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

        return lgb.train(param, train_data, num_round, valid_sets=[validation_data], verbose_eval=False)

    def postprocess_epoch(self, results=None, training=None, epoch=None, **kwargs):

            if not training and not epoch % 1:
                self.labeled_dataset.normalize = True

                self.logger.info("Evaluating the downstream task")
                features = self.evaluate(self.labeled_dataset, projection=False, prediction=False, augmentations=0)
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
                  projection=True, prediction=True, augmentations=0, inference_networks=True,
                  **kwargs):

        data = {}
        if predicting:
            x = sample
        else:
            x = sample['x']

            if 'y' in sample:
                data['y'] = sample['y']

        networks = self.inference_networks if inference_networks else self.networks

        b = len(x)
        if b < self.batch_size_eval:
            x = torch.cat([x, torch.zeros((self.batch_size_eval-b, *x.shape[1:]), device=x.device, dtype=x.dtype)])

        h = networks['encoder'](x)

        if b < self.batch_size_eval:
            h = h[:b]

        # h = self.optimized_encoder(x)
        data['h'] = h

        if 'projection' in networks and projection:
            z = networks['projection'](h)
            data['z'] = z

        if 'prediction' in networks and prediction:
            p = networks['prediction'](z)
            data['p'] = p

        if type(sample) is dict and 'augmentations' in sample and augmentations:
            representations = []
            for a in sample['augmentations']:
                representations.append(networks['encoder'](a))

            representations = torch.stack(representations)

            mu = representations.mean(dim=0)
            std = representations.std(dim=0)

            data['mu'] = mu
            data['std'] = std

        return data, results

    def evaluate(self, *args, **kwargs):
        '''
        For validation and test purposes (when labels are known)
        '''
        return self(*args, predicting=False, **kwargs)

    def predict(self, dataset, *args, lazy=False, add_to_population=False, anomaly_detction=False, **kwargs):
        '''
        Build faiss populations and calculate anomalies
        '''

        # if add_to_population:
        #
        #
        # features = super().predict(dataset, *args, lazy=False, **kwargs)

        return self(dataset, *args, predicting=True, **kwargs)


class BeamBarlowTwins(BeamSSL):

    def __init__(self, hparams):
        super().__init__(hparams)

        h = self.h_dim
        p = self.p_dim
        self.n_ensembles = hparams.n_ensembles

        projection = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, p))

        discriminator = nn.Sequential(spectral_norm(nn.Linear(h, h)),
                                   nn.ReLU(), spectral_norm(nn.Linear(h, h)), nn.ReLU(), nn.Linear(h, 1))

        self.add_networks_and_optmizers(networks={'projection': projection, 'discriminator': discriminator})

        # if self.n_ensembles > 1:
        ensemble = BeamEnsemble(self.generate_encoder, n_ensembles=self.n_ensembles)
        ensemble.set_optimizers(BeamOptimizer.prototype(dense_args={'lr': self.hparams.lr_dense,
                                                                    'weight_decay': self.hparams.weight_decay,
                                                                    'betas': (self.hparams.momentum, self.hparams.beta2),
                                                                    'eps': self.hparams.eps}))

        self.add_networks_and_optmizers(networks=ensemble, name='encoder', build_optimizers=False)

        # beam_weights_initializer(self.networks['projection'])
        beam_weights_initializer(self.networks['discriminator'], method='orthogonal')

    def iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x_aug1, x_aug2 = sample['augmentations']

        encoder = self.networks['encoder']

        projection = self.networks['projection']
        opt_p = self.optimizers['projection']

        discriminator = self.networks['discriminator']
        opt_d = self.optimizers['discriminator']

        freeze_network_params(encoder, projection)
        free_network_params(discriminator)

        index = torch.randperm(encoder.n_ensembles)
        h = encoder(x_aug1, index=index[0])
        r = torch.randn_like(h)

        d_h = discriminator(h)
        d_r = discriminator(r)

        loss_d = F.softplus(-d_h) + F.softplus(d_r)
        # loss_d = -d_h + d_r
        loss_d = self.apply(loss_d, training=training, optimizers=[opt_d], name='discriminator')
        results['scalar']['loss_d'].append(float(loss_d))
        results['scalar']['stats_mu'].append(float(h.mean()))
        results['scalar']['stats_std'].append(float(h.std()))

        if not counter % self.hparams.n_discriminator_steps:
            free_network_params(encoder, projection)
            freeze_network_params(discriminator)

            index = torch.randperm(encoder.n_ensembles)

            ind1 = index[0]
            ind2 = index[min(len(index)-1, 1)]
            opt_e1 = encoder.optimizers[ind1]
            opt_e2 = encoder.optimizers[ind2]

            h1 = encoder(x_aug1, index=index[0])
            h2 = encoder(x_aug2, index=index[min(len(index)-1, 1)])

            d1 = discriminator(h1)
            d2 = discriminator(h2)

            z1 = projection(h1)
            z2 = projection(h2)

            z1 = (z1 - z1.mean(dim=0, keepdim=True)) / (z1.std(dim=0, keepdim=True) + 1e-6)
            z2 = (z2 - z2.mean(dim=0, keepdim=True)) / (z2.std(dim=0, keepdim=True) + 1e-6)

            b, d = z1.shape
            corr = (z1.T @ z2) / b

            I = torch.eye(d, device=corr.device)
            corr_diff = (corr - I) ** 2

            invariance = torch.diag(corr_diff)
            redundancy = (corr_diff * (1 - I)).sum(dim=-1)
            # discrimination = -F.softplus(-d1) - F.softplus(-d2)
            discrimination = F.softplus(d1) + F.softplus(d2)
            # discrimination = d1 + d2

            opts = [opt_e1, opt_p] if ind1 == ind2 else [opt_e1, opt_e2, opt_p]
            loss = self.apply(invariance, self.hparams.lambda_twins * redundancy,
                              self.hparams.lambda_disc * discrimination, training=training,
                              optimizers=opts, name='encoder')

            # add scalar measurements
            results['scalar']['loss'].append(float(loss))
            results['scalar']['invariance'].append(float(invariance.mean()))
            results['scalar']['redundancy'].append(float(redundancy.mean()))
            results['scalar']['discrimination'].append(float(discrimination.mean()))

        return results


# class BeamBarlowTwins2(BeamSSL):
#
#     def __init__(self, hparams):
#         super().__init__(hparams)
#
#         h = self.h_dim
#         p = self.p_dim
#         self.n_ensembles = hparams.n_ensembles
#
#         projection = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
#                                    nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
#                                    nn.ReLU(), nn.Linear(h, p))
#
#         self.add_networks_and_optmizers(networks={'projection': projection})
#
#         ensemble = BeamEnsemble(self.generate_encoder, n_ensembles=self.n_ensembles)
#         ensemble.set_optimizers(BeamOptimizer.prototype(dense_args={'lr': self.hparams.lr_dense,
#                                                                     'weight_decay': self.hparams.weight_decay,
#                                                                     'betas': (self.hparams.momentum, self.hparams.beta2),
#                                                                     'eps': self.hparams.eps}))
#
#         self.add_networks_and_optmizers(networks=ensemble, name='encoder', build_optimizers=False)
#
#     def iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):
#
#         x_aug1, x_aug2 = sample['augmentations']
#
#         encoder = self.networks['encoder']
#
#         projection = self.networks['projection']
#         opt_p = self.optimizers['projection']
#
#         index = torch.randperm(encoder.n_ensembles)
#
#         ind1 = index[0]
#         ind2 = index[min(len(index)-1, 1)]
#         opt_e1 = encoder.optimizers[ind1]
#         opt_e2 = encoder.optimizers[ind2]
#
#         h1 = encoder(x_aug1, index=index[0])
#         h2 = encoder(x_aug2, index=index[min(len(index)-1, 1)])
#
#         z1 = projection(h1)
#         z2 = projection(h2)
#
#         mu1 = z1.mean(dim=0, keepdim=True)
#         mu2 = z2.mean(dim=0, keepdim=True)
#
#         std1 = z1.std(dim=0, keepdim=True)
#         std2 = z2.std(dim=0, keepdim=True)
#
#         z1 = (z1 - mu1) / (std1 + 1e-6)
#         z2 = (z2 - mu2) / (std2 + 1e-6)
#
#         b, d = z1.shape
#         corr = (z1.T @ z2) / b
#
#         I = torch.eye(d, device=corr.device)
#         corr_diff = (corr - I) ** 2
#
#         invariance = torch.diag(corr_diff)
#         redundancy = (corr_diff * (1 - I)).sum(dim=-1)
#         variance =
#
#         opts = [opt_e1, opt_p] if ind1 == ind2 else [opt_e1, opt_e2, opt_p]
#         loss = self.apply(invariance, self.hparams.lambda_twins * redundancy,
#                           self.hparams.lambda_disc * variance, training=training,
#                           optimizers=opts, name='encoder')
#
#         # add scalar measurements
#         results['scalar']['loss'].append(float(loss))
#         results['scalar']['invariance'].append(float(invariance.mean()))
#         results['scalar']['redundancy'].append(float(redundancy.mean()))
#         results['scalar']['variance'].append(float(variance.mean()))
#
#         return results


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


class BeamVICReg(BeamSSL):

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

        h1 = encoder(x_aug1)
        h2 = encoder(x_aug2)

        z1 = projection(h1)
        z2 = projection(h2)

        sim_loss = F.mse_loss(z1, z2, reduction='none').mean(dim=0)

        # mu1_h = h1.mean(dim=0, keepdim=True)
        # mu2_h = h2.mean(dim=0, keepdim=True)

        mu1 = z1.mean(dim=0, keepdim=True)
        mu2 = z2.mean(dim=0, keepdim=True)

        mean_loss = mu1.pow(2) + mu2.pow(2)
        # meanh_loss = mu1_h.pow(2) + mu2_h.pow(2)

        std1 = torch.sqrt(z1.var(dim=0) + self.hparams.var_eps)
        std2 = torch.sqrt(z2.var(dim=0) + self.hparams.var_eps)

        # std1_h = torch.sqrt(h1.var(dim=0) + self.hparams.var_eps)
        # std2_h = torch.sqrt(h2.var(dim=0) + self.hparams.var_eps)

        std_loss = F.relu(1 - std1) + F.relu(1 - std2)
        # stdh_loss = std1_h.pow(2) + std2_h.pow(2) - torch.log(std1_h) - torch.log(std2_h)

        z1 = (z1 - mu1)
        z2 = (z2 - mu2)

        b, d = z1.shape

        corr1 = (z1.T @ z1) / (b - 1)
        corr2 = (z2.T @ z2) / (b - 1)

        I = torch.eye(d, device=corr1.device)
        cov_loss = (corr1 * (1 - I)).pow(2).sum(dim=0) + (corr2 * (1 - I)).pow(2).sum(dim=0)

        # self.apply({'sim_loss': sim_loss, 'std_loss': std_loss, 'stdh_loss': stdh_loss,
        #                    'cov_loss': cov_loss, 'mean_loss': mean_loss, 'meanh_loss': meanh_loss},
        #                   weights={'sim_loss': self.hparams.lambda_vicreg,
        #                            'std_loss': self.hparams.mu_vicreg,
        #                            'stdh_loss': self.hparams.mu_vicreg,
        #                            'cov_loss': self.hparams.nu_vicreg,
        #                            'mean_loss': self.hparams.lambda_mean_vicreg,
        #                            'meanh_loss': self.hparams.lambda_mean_vicreg}, results=results,
        #                   training=training, optimizers=[opt_e, opt_p])

        self.apply({'sim_loss': sim_loss, 'std_loss': std_loss,
                           'cov_loss': cov_loss, 'mean_loss': mean_loss, },
                          weights={'sim_loss': self.hparams.lambda_vicreg,
                                   'std_loss': self.hparams.mu_vicreg,
                                   'cov_loss': self.hparams.nu_vicreg,
                                   'mean_loss': self.hparams.lambda_mean_vicreg,}, results=results,
                          training=training, optimizers=[opt_e, opt_p])

        # add scalar measurements
        results['scalar']['h_mean'].append(to_numpy(h1.mean(dim=0).flatten()))
        results['scalar']['h_std'].append(to_numpy(h1.std(dim=0).flatten()))
        results['scalar']['z_mean'].append(to_numpy(mu1.flatten()))
        results['scalar']['z_std'].append(to_numpy(z1.std(dim=0).flatten()))

        return results


class VICReg(BeamSSL):

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

        h1 = encoder(x_aug1)
        h2 = encoder(x_aug2)

        z1 = projection(h1)
        z2 = projection(h2)

        sim_loss = F.mse_loss(z1, z2, reduction='mean')

        mu1 = z1.mean(dim=0, keepdim=True)
        mu2 = z2.mean(dim=0, keepdim=True)

        std1 = torch.sqrt(z1.var(dim=0) + self.hparams.var_eps)
        std2 = torch.sqrt(z2.var(dim=0) + self.hparams.var_eps)
        std_loss = torch.mean(F.relu(1 - std1)) + torch.mean(F.relu(1 - std2))

        z1 = (z1 - mu1)
        z2 = (z2 - mu2)

        b, d = z1.shape
        corr1 = (z1.T @ z1) / (b - 1)
        corr2 = (z2.T @ z2) / (b - 1)

        I = torch.eye(d, device=corr1.device)
        cov_loss = (corr1 * (1 - I)).pow(2).sum() / d + (corr2 * (1 - I)).pow(2).sum() / d

        loss = self.hparams.lambda_vicreg * sim_loss + self.hparams.mu_vicreg * std_loss + self.hparams.nu_vicreg * cov_loss
        loss = self.apply(loss, training=training, optimizers=[opt_e, opt_p])

        # add scalar measurements
        results['scalar']['loss'].append(to_numpy(loss))
        results['scalar']['sim_loss'].append(to_numpy(sim_loss))
        results['scalar']['std_loss'].append(to_numpy(std_loss))
        results['scalar']['cov_loss'].append(to_numpy(cov_loss))
        results['scalar']['stats_mu'].append(h1.mean(dim=0).detach().cpu().numpy())
        results['scalar']['stats_std'].append(h1.std(dim=0).detach().cpu().numpy())

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

    hparams = beam_arguments(BeamSSL.get_parser(),
                         f"--project-name=beam_ssl --epoch-length=200 --reduction=mean --no-scale-epoch-by-batch-size",
                         f" --identifier=parallel_alternatives",
                         "--algorithm=BeamVICReg --parallel=4 --amp --model=swin_s --no-pretrained --layer=avgpool ",
                         f"--batch-size=96 --n-epochs=20000 --no-broadcast-buffers --lr-d=1e-3 --weight-decay=1e-5")

    logger = beam_logger()

    experiment = Experiment(hparams)
    Alg = globals()[experiment.hparams.algorithm]

    logger.info(f"BeamSSL model: {experiment.hparams.algorithm}")
    alg = experiment.fit(Alg, STL10Dataset)