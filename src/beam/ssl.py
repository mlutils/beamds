import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math

from .model import soft_target_update, target_copy, reset_network, copy_network, BeamEnsemble
from .model import beam_weights_initializer, freeze_network_params, free_network_params
from .utils import as_numpy, pretty_format_number, logger, beam_logger
from .algorithm import Algorithm
from .optim import BeamOptimizer
from .config import boolean_feature, get_beam_parser

import lightgbm as lgb
from torch.nn.utils import spectral_norm
import faiss
# working with faiss and torch
import faiss.contrib.torch_utils
from sklearn.manifold import TSNE
import umap
from collections import namedtuple

Similarities = namedtuple("Similarities", "index distance")


def get_ssl_parser():

    parser = get_beam_parser()

    boolean_feature(parser, "verbose-lgb", False, "Print progress in lgb training")
    parser.add_argument('--similarity', type=str, metavar='hparam', default='cosine',
                        help='Similarity distance in UniversalSSL')
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

    parser.add_argument('--lgb-rounds', type=int, default=40, help='LGB argument: num_round')
    parser.add_argument('--lgb-num-leaves', type=int, default=31, help='LGB argument: num_leaves')
    parser.add_argument('--lgb-max-depth', type=int, default=4, help='LGB argument: max_depth')
    parser.add_argument('--lgb-device', type=int, default=None, help='LGB argument: device')
    return parser


class BeamSimilarity(object):

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

        if index is None:
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
                            M = int(M[M_ind[0]])
                    if M is not None:
                        logger.info(f"Using HNSW{M}. Expected RAM footprint is "
                                    f"{pretty_format_number(footprints[M_ind[0]] / int(1e6))} MB")
                        index = faiss.IndexHNSWFlat(d, M, metric)
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
                        index = faiss.GpuIndexIVFFlat(res, d,  nlists, faiss.METRIC_L2, config)
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
        self.inference_device = inference_device

        self.training_index = None
        res = faiss.StandardGpuResources()
        if training_device != 'cpu' and inference_device == 'cpu':
            self.training_index = faiss.index_cpu_to_gpu(res, training_device, index)

        self.training_device = training_device

        if reducer == 'umap':
            self.reducer = umap.UMAP()
        elif reducer == 'tsne':
            self.reducer = TSNE()
        else:
            raise NotImplementedError

    def train(self, x):

        x = x.to(self.training_device)
        self.index.train(x)

    def add(self, x, train=False):

        x = x.to(self.inference_device)
        self.index.add(x)

        if (train is None and not self.index.is_trained) or train:
            self.train(x)

    def most_similar(self, x, n=1):

        x = x.to(self.inference_device)
        D, I = self.index.search(x, n)
        return Similarities(index=I, distance=D)

    def __len__(self):
        return self.index.ntotal

    def reduce(self, z):
        return self.reducer.fit_transform(z)


class BeamSSL(Algorithm):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, dataset=None, labeled_dataset=None):

        if networks is None:
            networks = {}

        encoder = self.generate_encoder()
        if encoder is not None:
            networks['encoder'] = encoder

        self.logger = beam_logger()

        super().__init__(hparams, networks=networks, optimizers=optimizers, schedulers=schedulers)

        if labeled_dataset is None:
            labeled_dataset = self.generate_labeled_set()
        self.labeled_dataset = labeled_dataset

        self.index_train_labeled = np.array(self.labeled_dataset.indices['train'])
        self.index_test_labeled = np.array(self.labeled_dataset.indices['test'])
        self.sim = None

    def generate_labeled_set(self, *args, pretrained=None, **kwargs):
        """
        This function should be overridden by the child class. Its purpose is to generate a labeled test-set for the
        evaluation of the downstream task.
        @return: UniversalDataset
        """
        return None

    def generate_encoder(self, *args, pretrained=None, **kwargs):
        """
        This function should be overridden by the child class. Its purpose is to generate a fresh
        (untrained or pretrained) encoder.
        @param pretrained:
        @return: nn.Module
        """
        return None

    @property
    def p_dim(self):
        raise NotImplementedError

    @property
    def h_dim(self):
        raise NotImplementedError

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

        train_data = lgb.Dataset(z[self.index_train_labeled], label=y[self.index_train_labeled])
        validation_data = lgb.Dataset(z[self.index_test_labeled], label=y[self.index_test_labeled])

        if self.hparams.lgb_device is None:
            device = None if 'cpu' == self.device.type else self.device.index
        else:
            device = self.hparams.lgb_device

        num_round = self.hparams.lgb_rounds
        param = {'objective': 'multiclass',
                 'num_leaves': self.hparams.lgb_num_leaves,
                 'max_depth': self.hparams.lgb_max_depth,
                 'gpu_device_id': device,
                 'verbosity': -1,
                 'metric': ['multi_error', 'multiclass'],
                 'num_class': np.max(y) + 1}

        return lgb.train(param, train_data, num_round, valid_sets=[validation_data], verbose_eval=self.hparams.verbose_lgb)

    def postprocess_epoch(self, results=None, training=None, epoch=None, **kwargs):

        if not training and not epoch % 1:

            self.logger.info("Evaluating the downstream task")
            features = self.evaluate(self.labeled_dataset, projection=False, prediction=False, augmentations=0)
            z = as_numpy(features.values['h'])
            y = as_numpy(features.values['y'])

            bst = self.evaluate_downstream_task(z, y)

            results['scalar']['encoder_acc'] = 1 - bst.best_score['valid_0']['multi_error']
            results['scalar']['encoder_loss'] = bst.best_score['valid_0']['multi_logloss']

            if 'z' in features.values:

                z = as_numpy(features.values['z'])
                bst = self.evaluate_downstream_task(z, y)

                results['scalar']['projection_acc'] = 1 - bst.best_score['valid_0']['multi_error']
                results['scalar']['projection_loss'] = bst.best_score['valid_0']['multi_logloss']

                if 'p' in features.values:

                    z = as_numpy(features.values['p'])
                    bst = self.evaluate_downstream_task(z, y)

                    results['scalar']['prediction_acc'] = 1 - bst.best_score['valid_0']['multi_error']
                    results['scalar']['prediction_loss'] = bst.best_score['valid_0']['multi_logloss']

        return results

    def build_similarity(self, add_sets=None, train_sets=None, metric='l2', training_device=None, inference_device=None,
                         ram_footprint=2 ** 8 * int(1e9), gpu_footprint=24 * int(1e9), exact=False, nlists=None,
                         M=None, latent_variable='h', projection=False, prediction=False):

        device = self.device
        device = device.type if 'cpu' == device.type else device.index

        if training_device is None:
            training_device = device
        if inference_device is None:
            inference_device = device

        if add_sets is None:
            add_sets = ['train', self.eval_subset, self.labeled_dataset]
        if train_sets is None:
            train_sets = add_sets

        d = self.h_dim

        expected_population = 0
        add_dataloaders = {}
        for subset in add_sets:
            dataloader = self.build_dataloader(subset)
            expected_population += len(dataloader.dataset)
            add_dataloaders[id(subset)] = dataloader

        train_population = 0
        train_dataloaders = {}
        for subset in train_sets:
            dataloader = self.build_dataloader(subset)
            train_population += len(dataloader.dataset)
            train_dataloaders[id(subset)] = dataloader

        self.sim = BeamSimilarity(d=d, expected_population=expected_population,
                                  metric=metric, training_device=training_device, inference_device=inference_device,
                                  ram_footprint=ram_footprint, gpu_footprint=gpu_footprint, exact=exact,
                                  nlists=nlists, M=M, reducer='umap')

        h = []
        for i, dataloader in add_dataloaders.items():
            predictions = self.predict(dataloader, prediction=prediction, projection=projection,
                                       add_to_sim=True, latent_variable=latent_variable)
            if i in train_dataloaders:
                h.append(predictions.data[latent_variable])

        for i, dataloader in train_dataloaders.items():
            if i not in add_dataloaders:
                predictions = self.predict(dataloader, prediction=prediction, projection=projection,
                                       add_to_sim=False, latent_variable=latent_variable)

                h.append(predictions.data[latent_variable])

        h = torch.cat(h)
        self.sim.train(h)

        return self.sim

    def inference(self, sample=None, results=None, subset=None, predicting=True, similarity=0,
                  projection=True, prediction=True, augmentations=0, inference_networks=True,
                  add_to_sim=False, latent_variable='h', **kwargs):

        data = {}
        if isinstance(sample, dict):
            x = sample['x']
            if 'y' in sample:
                data['y'] = sample['y']
        else:
            x = sample

        networks = self.inference_networks if inference_networks else self.networks

        # b = len(x)
        # if b < self.batch_size_eval:
        #     x = torch.cat([x, torch.zeros((self.batch_size_eval-b, *x.shape[1:]), device=x.device, dtype=x.dtype)])

        h = networks['encoder'](x)

        # if b < self.batch_size_eval:
        #     h = h[:b]

        data['h'] = h

        if 'projection' in networks and projection:
            z = networks['projection'](h)
            data['z'] = z

        if 'prediction' in networks and prediction:
            p = networks['prediction'](z)
            data['p'] = p

        if isinstance(sample, dict) and 'augmentations' in sample and augmentations:
            representations = []
            for a in sample['augmentations']:
                representations.append(networks['encoder'](a))

            representations = torch.stack(representations)

            mu = representations.mean(dim=0)
            std = representations.std(dim=0)

            results['scalar']['mu'] = mu
            results['scalar']['std'] = std

        if add_to_sim:
            if self.sim is not None:
                self.sim.add(data[latent_variable], train=False)
            else:
                logger.error("Please build similarity object first before adding indices. Use alg.build_similarity()")

        if similarity > 0:
            if self.sim is not None:
                similarities = self.sim.most_similar(data[latent_variable], n=similarity)

                data['similarities_index'] = similarities.index
                data['similarities_distance'] = similarities.distance

            else:
                logger.error("Please build and train similarity object first before calculating similarities. "
                             "Use alg.build_similarity()")

        return data, results


class BeamBarlowTwins(BeamSSL):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, **kwargs):

        if networks is None:
            networks = {}
        h = self.h_dim
        p = self.p_dim

        self.n_ensembles = hparams.n_ensembles

        networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, p))

        networks['discriminator'] = nn.Sequential(spectral_norm(nn.Linear(h, h)),
                                   nn.ReLU(), spectral_norm(nn.Linear(h, h)), nn.ReLU(), nn.Linear(h, 1))

        super().__init__(hparams, networks=networks, optimizers=optimizers, schedulers=schedulers, **kwargs)

        ensemble = BeamEnsemble(self.generate_encoder, n_ensembles=self.n_ensembles)
        ensemble.set_optimizers(BeamOptimizer.prototype(dense_args={'lr': self.hparams.lr_dense,
                                                                    'weight_decay': self.hparams.weight_decay,
                                                                    'betas': (self.hparams.momentum, self.hparams.beta2),
                                                                    'eps': self.hparams.eps}))

        self.add_networks_and_optimizers(networks=ensemble, name='encoder', build_optimizers=False)
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
            discrimination = F.softplus(d1) + F.softplus(d2)

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


class BarlowTwins(BeamSSL):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, **kwargs):

        if networks is None:
            networks = {}
        h = self.h_dim
        p = self.p_dim

        networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, p))
        super().__init__(hparams, networks=networks, optimizers=optimizers, schedulers=schedulers, **kwargs)

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

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, **kwargs):

        if networks is None:
            networks = {}
        h = self.h_dim
        p = self.p_dim

        networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, p))
        super().__init__(hparams, networks=networks, optimizers=optimizers, schedulers=schedulers, **kwargs)

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

        std1 = torch.sqrt(z1.var(dim=0) + self.hparams.var_eps)
        std2 = torch.sqrt(z2.var(dim=0) + self.hparams.var_eps)

        std_loss = F.relu(1 - std1) + F.relu(1 - std2)

        z1 = (z1 - mu1)
        z2 = (z2 - mu2)

        b, d = z1.shape

        corr1 = (z1.T @ z1) / (b - 1)
        corr2 = (z2.T @ z2) / (b - 1)

        I = torch.eye(d, device=corr1.device)
        cov_loss = (corr1 * (1 - I)).pow(2).sum(dim=0) + (corr2 * (1 - I)).pow(2).sum(dim=0)

        self.apply({'sim_loss': sim_loss, 'std_loss': std_loss,
                           'cov_loss': cov_loss, 'mean_loss': mean_loss, },
                          weights={'sim_loss': self.hparams.lambda_vicreg,
                                   'std_loss': self.hparams.mu_vicreg,
                                   'cov_loss': self.hparams.nu_vicreg,
                                   'mean_loss': self.hparams.lambda_mean_vicreg,}, results=results,
                          training=training, optimizers=[opt_e, opt_p])

        # add scalar measurements
        results['scalar']['h_mean'].append(as_numpy(h1.mean(dim=0).flatten()))
        results['scalar']['h_std'].append(as_numpy(h1.std(dim=0).flatten()))
        results['scalar']['z_mean'].append(as_numpy(mu1.flatten()))
        results['scalar']['z_std'].append(as_numpy(z1.std(dim=0).flatten()))

        return results


class VICReg(BeamSSL):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, **kwargs):

        if networks is None:
            networks = {}
        h = self.h_dim
        p = self.p_dim

        networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, p))

        super().__init__(hparams, networks=networks, optimizers=optimizers, schedulers=schedulers, **kwargs)

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
        results['scalar']['loss'].append(as_numpy(loss))
        results['scalar']['sim_loss'].append(as_numpy(sim_loss))
        results['scalar']['std_loss'].append(as_numpy(std_loss))
        results['scalar']['cov_loss'].append(as_numpy(cov_loss))
        results['scalar']['stats_mu'].append(as_numpy(h1.mean(dim=0)))
        results['scalar']['stats_std'].append(as_numpy(h1.std(dim=0)))

        return results


class SimCLR(BeamSSL):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, **kwargs):

        if networks is None:
            networks = {}
        h = self.h_dim
        p = self.p_dim

        networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                                   nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                                   nn.ReLU(), nn.Linear(h, p))

        super().__init__(hparams, networks=networks, optimizers=optimizers, schedulers=schedulers, **kwargs)

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
        s = s * (1 - torch.eye(2 * b, 2 * b, device=s.device)) / self.hparams.temperature

        logsumexp = torch.logsumexp(s[::2], dim=1)
        s_couple = torch.diag(s, diagonal=1)[::2]

        loss = - s_couple + logsumexp
        loss = self.apply(loss, training=training, optimizers=[opt_e, opt_p])

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['acc'].append(float((s_couple >= s[::2].max(dim=1).values).float().mean()))

        return results


class SimSiam(BeamSSL):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, **kwargs):

        if networks is None:
            networks = {}
        h = self.h_dim
        p = self.p_dim

        networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, p))

        networks['prediction'] = nn.Sequential(nn.Linear(p, p), nn.BatchNorm1d(p), nn.ReLU(), nn.Linear(p, p))
        super().__init__(hparams, networks=networks, optimizers=optimizers, schedulers=schedulers, **kwargs)

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

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, **kwargs):

        if networks is None:
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
        super().__init__(hparams, networks=networks, optimizers=optimizers, schedulers=schedulers, **kwargs)

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