import numpy as np
import torch
from ..algorithm import NeuralAlgorithm
from ..logging import beam_logger as logger
from ..utils import as_numpy
from .beam_similarity import BeamSimilarity


class BeamSSL(NeuralAlgorithm):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, dataset=None, labeled_dataset=None):

        if networks is None:
            networks = {}

        encoder = self.generate_encoder()
        if encoder is not None:
            networks['encoder'] = encoder

        self.logger = logger

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

            results[Types.scalar]['encoder_acc'] = 1 - bst.best_score['valid_0']['multi_error']
            results[Types.scalar]['encoder_loss'] = bst.best_score['valid_0']['multi_logloss']

            if 'z' in features.values:

                z = as_numpy(features.values['z'])
                bst = self.evaluate_downstream_task(z, y)

                results[Types.scalar]['projection_acc'] = 1 - bst.best_score['valid_0']['multi_error']
                results[Types.scalar]['projection_loss'] = bst.best_score['valid_0']['multi_logloss']

                if 'p' in features.values:

                    z = as_numpy(features.values['p'])
                    bst = self.evaluate_downstream_task(z, y)

                    results[Types.scalar]['prediction_acc'] = 1 - bst.best_score['valid_0']['multi_error']
                    results[Types.scalar]['prediction_loss'] = bst.best_score['valid_0']['multi_logloss']

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

    def inference_iteration(self, sample=None, results=None, subset=None, predicting=True, similarity=0,
                            projection=True, prediction=True, augmentations=0, add_to_sim=False, latent_variable='h',
                            **kwargs):

        data = {}
        if isinstance(sample, dict):
            x = sample['x']
            if 'y' in sample:
                data['y'] = sample['y']
        else:
            x = sample

        networks = self.networks

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

            results[Types.scalar]['mu'] = mu
            results[Types.scalar]['std'] = std

        if add_to_sim:
            if self.sim is not None:
                self.sim.add(data[latent_variable], train=False)
            else:
                logger.error("Please build similarity object first before adding indices. Use alg.build_similarity()")

        if similarity > 0:
            if self.sim is not None:
                similarities = self.sim.search(data[latent_variable], k=similarity)

                data['similarities_index'] = similarities.index
                data['similarities_distance'] = similarities.distance

            else:
                logger.error("Please build and train similarity object first before calculating similarities. "
                             "Use alg.build_similarity()")

        return data, results
