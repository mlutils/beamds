import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from src.beam import beam_arguments, Experiment, beam_algorithm_generator
from src.beam import UniversalDataset, UniversalBatchSampler
from src.beam import NeuralAlgorithm
from src.beam import LinearNet
from src.beam import DataTensor, PackedFolds, as_numpy
from src.beam.data import BeamData
from src.beam.utils import DataBatch


class MNISTDataset(UniversalDataset):

    def __init__(self, hparams):

        path = hparams.data_path
        seed = hparams.split_dataset_seed

        super().__init__()
        dataset_train = torchvision.datasets.MNIST(root=path, train=True, transform=torchvision.transforms.ToTensor(), download=True)
        dataset_test = torchvision.datasets.MNIST(root=path, train=False, transform=torchvision.transforms.ToTensor(), download=True)

        # self.data = PackedFolds({'train': dataset_train.data, 'test': dataset_test.data})
        # self.labels = PackedFolds({'train': dataset_train.targets, 'test': dataset_test.targets})
        # self.split(test=self.labels['test'].index, seed=seed)

        # self.data = torch.cat([dataset_train.data, dataset_test.data])
        # self.labels = torch.cat([dataset_train.targets, dataset_test.targets])
        # test_indices = len(dataset_train.data) + torch.arange(len(dataset_test.data))
        # self.split(validation=.2, test=test_indices, seed=seed)

        # self.data = BeamData({'train': dataset_train.data, 'test': dataset_test.data},
        #                      label={'train': dataset_train.targets, 'test': dataset_test.targets}, quick_getitem=True)
        # self.labels = self.data.label
        # self.split(validation=.2, test=self.data['test'].index, seed=seed)

        self.data = BeamData.simple({'train': dataset_train.data, 'test': dataset_test.data},
                             label={'train': dataset_train.targets, 'test': dataset_test.targets}, quick_getitem=True)
        self.labels = self.data.label
        self.split(validation=.2, test=self.data['test'].index, seed=seed)

    def getitem(self, index):

        data = self.data[index]

        if isinstance(data, BeamData):
            x = data.stacked_values.float() / 255
            y = data.stacked_labels
        elif isinstance(data, DataBatch):
            x = data.data.float() / 255
            y = data.label

        else:
            x = self.data[index].float() / 255
            y = self.labels[index]

        return {'x': x, 'y': y}


class MNISTAlgorithm(NeuralAlgorithm):

    def __init__(self, hparams):

        # choose your network
        net = LinearNet(784, 256, 10, 4)
        super().__init__(hparams, networks=net)

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizers['net'].dense, gamma=0.99)
        self.stop_at = hparams.stop_at

    def postprocess_epoch(self, sample=None, epoch=None, subset=None, training=True, **kwargs):
        x, y = sample['x'], sample['y']
        self.report_images('sample', x[:16].view(16, 1, 28, 28))

    def train_iteration(self, sample=None, subset=None, counter=None, training=True, **kwargs):

        x, y = sample['x'], sample['y']

        x = x.view(len(x), -1)
        net = self.networks['net']

        y_hat = net(x)
        loss = F.cross_entropy(y_hat, y, reduction='mean')
        self.apply(loss)

        # add scalar measurements
        self.report_scalar('ones', x.sum(dim=-1))
        self.report_scalar('acc', (y_hat.argmax(1) == y).float().mean())

    def inference_iteration(self, sample=None, subset=None, predicting=True, **kwargs):

        if predicting:
            x = sample
        else:
            x, y = sample['x'], sample['y']

        x = x.view(len(x), -1)
        net = self.networks['net']

        y_hat = net(x)

        # add scalar metrics
        self.report_scalar('y_pred', y_hat)

        if not predicting:
            self.report_scalar('acc', (y_hat.argmax(1) == y).float().mean())
            self.report_scalar('target', y)
            return {'y': y, 'y_hat': y_hat}

        return y_hat

    def postprocess_inference(self, sample=None, subset=None, predicting=True, **kwargs):

        if not predicting:

            y_pred = as_numpy(torch.argmax(self.get_scalar('y_pred'), dim=1))
            y_true = as_numpy(self.get_scalar('target'))
            precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)

            self.report_data('metrics/precision', precision)
            self.report_data('metrics/recall', recall)
            self.report_data('metrics/fscore', fscore)
            self.report_data('metrics/support', support)

            self.report_scalar('objective', self.get_scalar('acc', aggregate=True))


# ## Training

if __name__ == '__main__':

     # in this example we do not set the logs-path and the data-path, so the defaults will be used

    args = beam_arguments(
        f"--project-name=mnist --algorithm=MNISTAlgorithm --amp  --device=0   ",
        " --n-epochs=10 --epoch-length=1000 --objective=acc --model-dtype=float16", stop_at=.99,
        scheduler='exponential', gamma=.999, scale_epoch_by_batch_size=False)

    experiment = Experiment(args)

    dataset = MNISTDataset(experiment.hparams)
    alg = MNISTAlgorithm(experiment.hparams)

    # train
    alg = experiment.fit(alg=alg, dataset=dataset)

    examples = alg.dataset[np.random.choice(len(alg.dataset), size=50000, replace=True)]
    res = alg.predict(examples.data['x'])

    # ## Inference
    inference = alg('validation')
    inference = alg({'x': examples.data['x'], 'y': examples.data['y']})

    print('Test inference results:')
    for n, v in inference.statistics['metrics'].items():
        print(f'{n}:')
        print(v)

