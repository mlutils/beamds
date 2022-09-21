import rtdl

try:
    from examples.example_utils import add_beam_to_path
except ModuleNotFoundError:
    from example_utils import add_beam_to_path
add_beam_to_path()

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from src.beam import beam_arguments, Experiment, as_tensor, as_numpy
from src.beam import UniversalDataset, UniversalBatchSampler
from src.beam import Algorithm, PackedFolds, beam_logger, LinearNet
from src.beam.ssl import get_ssl_parser
from src.beam.config import get_beam_parser

from sklearn.datasets import fetch_covtype
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from src.beam import BeamSimilarity, Similarities, BeamSSL, BYOL, BeamVICReg, BarlowTwins, VICReg, SimCLR, SimSiam, tqdm


class EmbeddingCovtypeDataset(UniversalDataset):

    def __init__(self, hparams):

        emb_size = 256
        path = hparams.path_to_data
        device = hparams.device
        seed = hparams.seed
        self.mask = hparams.mask
        self.quantiles = hparams.quantiles
        self.n_augmentations = 1

        dataset = fetch_covtype(data_home=path)
        data = dataset['data']
        columns = dataset['feature_names']
        y = np.array(dataset['target'], dtype=np.int64)
        df = pd.DataFrame(data=data, columns=columns, index=np.arange(len(data)))

        soils_columns = [c for c in df.columns if 'Soil' in c]
        soil = np.where(df[soils_columns])[1]

        wilderness_columns = [c for c in df.columns if 'Wilderness' in c]
        wilderness = np.where(df[wilderness_columns])[1]

        df_cat = pd.DataFrame({'Soil': soil, 'Wilderness': wilderness})
        df_num = df.drop(columns=(soils_columns + wilderness_columns))

        super().__init__(x=df.values, y=y, device=device)

        self.split(validation=92962, test=116203, seed=seed, stratify=False, labels=self.data['y'].cpu())
        self.mask_distribution = torch.distributions.Bernoulli(probs=self.mask)

        self.qt = QuantileTransformer(n_quantiles=self.quantiles,
                                      output_distribution='uniform',
                                      ignore_implicit_zeros=False,
                                      subsample=100000,
                                      random_state=None,
                                      copy=True).fit(df_num.iloc[self.indices['train']].values)

        x_cat = as_tensor(df_cat.values, device=self.device)
        x_num = (self.qt_transform(df_num.values, device=self.device) * self.quantiles).long()

        self.data['x_cat'] = torch.cat([x_num, x_cat], dim=1)

        self.features_names = list(df_num.columns) + list(df_cat.columns)

        self.n_categories = torch.cat([(self.quantiles + 1) * torch.ones(x_num.shape[-1], dtype=torch.int64),
                                       torch.tensor(df_cat.nunique().values)])

        self.offset = torch.cumsum(self.n_categories, dim=0) - self.n_categories
        self.offset = self.offset.to(self.device)

        self.n_classes = int(y.max() + 1)
        self.n_num = 0
        self.n_cat = self.data['x_cat'].shape[-1]
        self.features_index = torch.arange(self.data['x_cat'].shape[-1]).unsqueeze(0)
        self.embedding = torch.randn(self.n_categories.sum(), emb_size, device=self.device)

    def qt_transform(self, df_num, device=None):
        return as_tensor(self.qt.transform(as_numpy(df_num)).astype(np.float32), device=device)

    def getitem(self, ind):

        x = self.data['x'][ind]
        x_cat = self.data['x_cat'][ind]
        y = self.data['y'][ind]

        emb = self.embedding[x_cat + self.offset]

        data = {'x': x, 'x_cat': x_cat, 'emb':emb, 'y': y}

        return data

class CovtypeDatasetOrg(UniversalDataset):

    def __init__(self, hparams):

        path = hparams.path_to_data
        device = hparams.device
        seed = hparams.seed

        dataset = fetch_covtype(data_home=path)
        data = dataset['data']
        columns = dataset['feature_names']
        y = np.array(dataset['target'], dtype=np.int64)
        df = pd.DataFrame(data=data, columns=columns, index=np.arange(len(data)))

        super().__init__(x_num=df.values.astype(np.float32), x_cat=None,
                         y=y, device=device)
        # super().__init__(x_num=df_num.values.astype(np.float32), x_cat=df_cat.values.astype(np.float32),
        #                  y=y, device=device)

        self.split(validation=92962, test=116203, seed=seed, stratify=False, labels=self.data['y'].cpu())

    def qt_transform(self, df_num, device=None):
        return as_tensor(self.qt.transform(as_numpy(df_num)).astype(np.float32), device=device)

    def getitem(self, ind):
        x_num = self.data['x_num'][ind]
        y = self.data['y'][ind]

        x = (x_num, None)

        data = {'x': x, 'y': y}

        return data

class CovtypeDataset(UniversalDataset):

    def __init__(self, hparams):

        path = hparams.path_to_data
        device = hparams.device
        seed = hparams.seed
        self.mask = hparams.mask
        self.n_augmentations = 2

        dataset = fetch_covtype(data_home=path)
        data = dataset['data']
        columns = dataset['feature_names']
        y = np.array(dataset['target'], dtype=np.int64)
        df = pd.DataFrame(data=data, columns=columns, index=np.arange(len(data)))

        soils_columns = [c for c in df.columns if 'Soil' in c]
        soil = np.where(df[soils_columns])[1]

        wilderness_columns = [c for c in df.columns if 'Wilderness' in c]
        wilderness = np.where(df[wilderness_columns])[1]

        df_cat = pd.DataFrame({'Soil': soil, 'Wilderness': wilderness}) + 1
        # df_cat = df[soils_columns + wilderness_columns]
        df_num = df.drop(columns=(soils_columns + wilderness_columns))

        # covtype = pd.concat([df_num, df_cat], axis=1)
        # super().__init__(x=covtype.values.astype(np.float32), y=y, device=device)
        super().__init__(x_num=df_num.values.astype(np.float32), x_cat=df_cat.values.astype(np.int64),
                         y=y, device=device)
        # super().__init__(x_num=df_num.values.astype(np.float32), x_cat=df_cat.values.astype(np.float32),
        #                  y=y, device=device)

        self.split(validation=92962, test=116203, seed=seed, stratify=False, labels=self.data['y'].cpu())
        self.mask_distribution = torch.distributions.Bernoulli(probs=self.mask)

        self.qt = QuantileTransformer(n_quantiles=1000,
                                      output_distribution='uniform',
                                      ignore_implicit_zeros=False,
                                      subsample=100000,
                                      random_state=None,
                                      copy=True).fit(df_num.iloc[self.indices['train']].values)

        self.data['x_num'] = self.qt_transform(df_num, device=self.device)

        self.n_categories = torch.tensor(df_cat.nunique().values) + 1

        self.n_classes = int(y.max() + 1)
        self.n_num = df_num.shape[-1]
        self.n_cat = df_cat.shape[-1]

    def augment(self, x):
        x_num, x_cat = x
        x_cat = x_cat * self.mask_distribution.sample(x_cat.shape).to(self.device).long()

        mask = self.mask_distribution.sample(x_num.shape).to(self.device)
        x_num = .5 * mask + (1 - mask) * x_num
        return x_num, x_cat

    def qt_transform(self, df_num, device=None):
        return as_tensor(self.qt.transform(as_numpy(df_num)).astype(np.float32), device=device)

    def getitem(self, ind):
        x_num = self.data['x_num'][ind]
        x_cat = self.data['x_cat'][ind]
        y = self.data['y'][ind]

        x = (x_num, x_cat)
        augmentations = [self.augment(x) for _ in range(self.n_augmentations)]

        data = {'x': x, 'y': y, 'augmentations': augmentations}

        return data


class CovtypeMaskedDataset(UniversalDataset):

    def __init__(self, hparams):

        path = hparams.path_to_data
        device = hparams.device
        seed = hparams.seed
        self.mask = hparams.mask
        self.quantiles = hparams.quantiles
        self.n_augmentations = 1

        dataset = fetch_covtype(data_home=path)
        data = dataset['data']
        columns = dataset['feature_names']
        y = np.array(dataset['target'], dtype=np.int64)
        df = pd.DataFrame(data=data, columns=columns, index=np.arange(len(data)))

        soils_columns = [c for c in df.columns if 'Soil' in c]
        soil = np.where(df[soils_columns])[1]

        wilderness_columns = [c for c in df.columns if 'Wilderness' in c]
        wilderness = np.where(df[wilderness_columns])[1]

        df_cat = pd.DataFrame({'Soil': soil, 'Wilderness': wilderness})
        df_num = df.drop(columns=(soils_columns + wilderness_columns))

        super().__init__(y=y, device=device)

        self.split(validation=92962, test=116203, seed=seed, stratify=False, labels=self.data['y'].cpu())
        self.mask_distribution = torch.distributions.Bernoulli(probs=self.mask)

        self.qt = QuantileTransformer(n_quantiles=self.quantiles,
                                      output_distribution='uniform',
                                      ignore_implicit_zeros=False,
                                      subsample=100000,
                                      random_state=None,
                                      copy=True).fit(df_num.iloc[self.indices['train']].values)

        x_cat = as_tensor(df_cat.values, device=self.device)
        x_num = (self.qt_transform(df_num.values, device=self.device) * self.quantiles).long()

        self.data['x_cat'] = torch.cat([x_num, x_cat], dim=1)

        self.features_names = list(df_num.columns) + list(df_cat.columns)

        self.n_categories = torch.cat([(self.quantiles + 1) * torch.ones(x_num.shape[-1], dtype=torch.int64),
                                       torch.tensor(df_cat.nunique().values)])

        self.n_classes = int(y.max() + 1)
        self.n_num = 0
        self.n_cat = self.data['x_cat'].shape[-1]
        self.features_index = torch.arange(self.data['x_cat'].shape[-1]).unsqueeze(0)

    def augment(self, x):

        _, x_cat = x

        # we sample the corruption from the train subset
        ind_patched = torch.randint(len(self.indices['train']), size=x_cat.shape)
        x_patched = self.data['x_cat'][self.indices['train'][ind_patched], self.features_index]
        mask = self.mask_distribution.sample(x_cat.shape).to(self.device).long()
        x_cat = x_patched * mask + x_cat * (1 - mask)

        return {'x': (_, x_cat), 'mask': mask}

    def qt_transform(self, df_num, device=None):
        return as_tensor(self.qt.transform(as_numpy(df_num)).astype(np.float32), device=device)

    def getitem(self, ind):

        x_num = None
        x_cat = self.data['x_cat'][ind]
        y = self.data['y'][ind]

        x = (x_num, x_cat)
        augmentations = [self.augment(x) for _ in range(self.n_augmentations)]

        data = {'x': x, 'y': y, 'augmentations': augmentations}

        return data


class CovModuleWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, x):

        x_num, x_cat = x
        y = self.model(x_num, x_cat)

        return y


class CovTypeIdentity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):

        x_num, x_cat = x
        return x_cat


class BeamTabularSSL(BeamSSL):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, dataset=None, **kwargs):

        if networks is None:
            networks = {}
        h = self.h_dim
        p_cat = sum(dataset.n_categories)
        p_num = dataset.n_num
        self.mask_distribution = torch.distributions.Bernoulli(probs=self.hparams.mask)
        self.cat_splits = torch.cumsum(dataset.n_categories, dim=0)[:-1]

        networks['decoder_cat'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, p_cat))
        networks['decoder_num'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                                nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                                nn.ReLU(), nn.Linear(h, p_num))

        super().__init__(hparams, networks=networks, optimizers=optimizers, schedulers=schedulers, **kwargs)

    def iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x_num, x_cat = sample['x']

        mask_cat = self.mask_distribution.sample(x_cat.shape).to(self.device).long()
        x_cat_masked = x_cat * mask_cat

        mask_num = self.mask_distribution.sample(x_num.shape).to(self.device)
        x_num_masked = .5 * mask_num + (1 - mask_num) * x_num

        x = x_num_masked, x_cat_masked

        encoder = self.networks['encoder']
        decoder_cat = self.networks['decoder_cat']
        decoder_num = self.networks['decoder_num']

        h = encoder(x)
        x_cat_hat = decoder_cat(h)
        x_num_hat = decoder_num(h)

        loss_num = F.smooth_l1_loss(x_num_hat, x_num, reduction='none')
        loss_num = loss_num * (1 - mask_num)

        x_cat_hat = torch.tensor_split(x_cat_hat, self.cat_splits, dim=-1)
        loss_cat = []

        for i in range(x_cat.shape[-1]):
            loss_cat_i = F.cross_entropy(x_cat_hat[i], x_cat[:, i], reduction='none')
            loss_cat_i = loss_cat_i * (1 - mask_cat[:, i])
            loss_cat.append(loss_cat_i)

        loss_cat = torch.stack(loss_cat, dim=-1)

        self.apply({'loss_num': loss_num, 'loss_cat': loss_cat}, weights={'loss_num': 1, 'loss_cat': .1},
                   training=training, results=results)

        return results


class Vime(BeamSSL):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, dataset=None, **kwargs):

        if networks is None:
            networks = {}
        h = self.h_dim
        self.cat_splits = torch.cumsum(dataset.n_categories, dim=0)[:-1]

        networks['decoder'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, sum(dataset.n_categories)))

        networks['decoder_masks'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, len(dataset.n_categories)))

        # networks['decoder'] = nn.Sequential(nn.Linear(h, h),
        #                                        nn.ReLU(), nn.Linear(h, sum(dataset.n_categories)))
        #
        # networks['decoder_masks'] = nn.Sequential(nn.Linear(h, h),
        #                                        nn.ReLU(), nn.Linear(h, len(dataset.n_categories)))

        super().__init__(hparams, networks=networks, optimizers=optimizers, schedulers=schedulers, **kwargs)


    def iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        _, x = sample['x']
        _, x_aug = sample['augmentations'][0]['x']
        mask = sample['augmentations'][0]['mask']

        encoder = self.networks['encoder']
        decoder = self.networks['decoder']
        decoder_masks = self.networks['decoder_masks']

        h = encoder((_, x_aug))
        x_hat = decoder(h)
        masks_hat = decoder_masks(h)

        loss_mask = F.binary_cross_entropy_with_logits(masks_hat, mask.float(), reduction='none',
                                 pos_weight=1/self.hparams.mask * torch.ones(masks_hat.shape, device=masks_hat.device))

        results['scalar'][f'acc_mask'].append(as_numpy((masks_hat > 0).long() == mask))

        x_hat = torch.tensor_split(x_hat, self.cat_splits, dim=-1)
        loss_cat = []

        for i in range(x.shape[-1]):
            loss_cat_i = F.cross_entropy(x_hat[i], x[:, i], reduction='none')
            loss_cat.append(loss_cat_i)
            results['scalar'][f'acc_{self.dataset.features_names[i]}'].append(as_numpy(torch.argmax(x_hat[i], dim=1)
                                                                                       == x[:, i]))

        loss_cat = torch.stack(loss_cat, dim=-1)

        self.apply({'loss_cat': loss_cat, 'loss_mask': loss_mask}, weights={'loss_cat': 1, 'loss_mask': 10},
                   training=training, results=results)

        return results


class Dummy(BeamSSL):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, dataset=None, **kwargs):

        if networks is None:
            networks = {}
        networks['encoder'] = CovTypeIdentity()

        super().__init__(hparams, networks=networks, optimizers=optimizers, schedulers=schedulers, **kwargs)

    @property
    def h_dim(self):
        return self.dataset.n_cat

    def iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):
        results['scalar']['const'].append(1)
        return results


def my_ssl_algorithm(algorithm):

    BaseClass = globals()[algorithm]

    class CovtypeSSL(BaseClass):

        def __init__(self, hparams, dataset=None):

            networks = {}
            optimizers = {}

            self.hparams = hparams
            model = rtdl.FTTransformer.make_baseline(n_num_features=dataset.n_num,
                                                      cat_cardinalities=list(as_numpy(dataset.n_categories)),
                                                      d_token=hparams.channels, n_blocks=hparams.n_layers,
                                                      attention_dropout=hparams.dropout,
                                                      ffn_dropout=hparams.dropout, d_out=hparams.h_dim,
                                                      residual_dropout=hparams.dropout, ffn_d_hidden=hparams.channels)

            networks['encoder'] = CovModuleWrapper(model)

            optimizers['encoder'] = torch.optim.AdamW(model.optimization_param_groups(),
                                                      lr=hparams.lr_dense, weight_decay=hparams.weight_decay)

            test_ind = dataset.indices['test']

            if 'x_num' not in dataset.data:
                labeled_dataset = UniversalDataset(x=(None, dataset.data['x_cat'][test_ind]),
                                                   y=dataset.data['y'][test_ind], device=dataset.device)
            elif 'x_cat' not in dataset.data:
                labeled_dataset = UniversalDataset(x=(dataset.data['x_num'][test_ind], None),
                                                   y=dataset.data['y'][test_ind], device=dataset.device)
            else:
                labeled_dataset = UniversalDataset(x=(dataset.data['x_num'][test_ind], dataset.data['x_cat'][test_ind]),
                                                   y=dataset.data['y'][test_ind], device=dataset.device)

            labeled_dataset.split(test=.5, stratify=False, labels=labeled_dataset.data['y'].cpu())

            super().__init__(hparams, networks=networks, optimizers=optimizers, labeled_dataset=labeled_dataset,
                             dataset=dataset)

        @property
        def p_dim(self):
            return self.hparams.p_dim

        @property
        def h_dim(self):
            return self.hparams.h_dim

    return CovtypeSSL


# Add experiment hyperparameter arguments
def get_covtype_parser():

    parser = get_ssl_parser()

    parser.add_argument('--weight-factor', type=float, default=0.,
                        help='Squashing factor for the oversampling probabilities')
    parser.add_argument('--objective', type=str, default='encoder_acc',
                        help='The objective is the accuracy of the downstream task')
    parser.add_argument('--dataset', type=str, default='CovtypeMaskedDataset', help='The dataset class')
    parser.add_argument('--channels', type=int, default=128, help='Size of embedding')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout value for rule layers')
    parser.add_argument('--mask', type=float, default=0.2, help='Masking augmentation parameter')
    parser.add_argument('--n-layers', type=int, default=3, help='Number of Hidden layers')

    parser.add_argument('--h-dim', type=int, default=64, help='Hidden size dimension')
    parser.add_argument('--p-dim', type=int, default=64, help='Projection size dimension')
    parser.add_argument('--quantiles', type=int, default=64, help='Number of quantiles for the numerical features')

    return parser


if __name__ == '__main__':

    # here you put all actions which are performed only once before initializing the workers
    # for example, setting running arguments and experiment:

    path_to_data = '/home/shared/data/dataset/covtype'
    root_dir = '/home/shared/data/results/covtype'

    logger = beam_logger()
    hparams = beam_arguments(get_covtype_parser(),
                             f"--project-name=covtype_ssl --root-dir={root_dir} --algorithm=BeamVICReg --device=2",
                             "--batch-size=512 --n-epochs=100 --parallel=1 --momentum=0.9 --beta2=0.99",
                             weight_factor=.0, weight_decay=1e-5, path_to_data=path_to_data, dropout=.0, channels=256,
                             n_layers=2)

    experiment = Experiment(hparams)
    Alg = my_ssl_algorithm(hparams.algorithm)
    Dataset = globals()[hparams.dataset]

    logger.info(f"BeamSSL algorithm: {hparams.algorithm}")
    logger.info(f"BeamSSL dataset: {hparams.dataset}")

    alg = experiment.fit(Alg=Alg, Dataset=Dataset)
