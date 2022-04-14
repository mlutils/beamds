import torch
import torchvision
import torch.nn.functional as F

from src.beam import parser, Experiment
from src.beam import UniversalDataset
from src.beam import Algorithm
from src.beam import LinearNet


class MNISTDataset(UniversalDataset):

    def __init__(self, path, train=True):
        super().__init__()
        self.data = torchvision.datasets.MNIST(root=path, train=train, transform=torchvision.transforms.ToTensor())

    def __getitem__(self, index):
        return {'x': self.data.data[index].float() / 255, 'y': self.data.targets[index]}

    def __len__(self):
        return len(self.data)


class MNISTAlgorithm(Algorithm):

    def __init__(self, *args, **argv):
        super().__init__(*args, **argv)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizers['net'].dense, gamma=0.99)

    def postprocess_epoch(self, sample, aux, results, epoch, train=True):

        x, y = sample['x'], sample['y']

        results['images']['sample'] = x[:16].view(16, 1, 28, 28).data.cpu()

        if train:
            self.scheduler.step()
            results['scalar'][f'lr'] = self.optimizers['net'].dense.param_groups[0]['lr']

        aux = {}
        return aux, results

    def iteration(self, sample, aux, results, training=True):

        x, y = sample['x'], sample['y']

        x = x.view(len(x), -1)
        net = self.networks['net']
        opt = self.optimizers['net']

        y_hat = net(x)
        loss = F.cross_entropy(y_hat, y, reduction='mean')

        if training:

            opt.zero_grad()
            loss.backward()
            opt.step()

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['acc'].append(float((y_hat.argmax(1) == y).float().mean()))
        aux = {}

        return aux, results


def run_mnist(rank, world_size, experiment):

    dataloader = {}
    for k, v in {'train': True, 'test': False}.items():

        dataloader[k] = MNISTDataset(experiment.path_to_data, train=v).dataloader(batch_size=experiment.batch_size,
                                                                       num_workers=experiment.cpu_workers,
                                                                       pin_memory=True)

    # choose your network
    net = LinearNet(784, 256, 10, 4)

    # we recommend using the algorithm argument to determine the type of algorithm to be used
    Alg = globals()[experiment.algorithm]
    alg = Alg(net, dataloader, experiment)

    # simulate input to the network
    x = next(alg.data_generator(training=False))[1]['x']
    x = x.view(len(x), -1)

    experiment.writer_control(enable=not(bool(rank)), networks=alg.get_networks(), inputs={'net': x})

    for results in iter(alg):
        experiment.save_model_results(results, alg,
                                      print_results=True, visualize_results='yes',
                                      store_results='logscale', store_networks='logscale',
                                      visualize_weights=True,
                                                    argv={'images': {'sample': {'dataformats': 'NCHW'}}})


if __name__ == '__main__':

    # here you put all actions which are performed only once before initializing the workers
    # for example, setting running arguments and experiment:

    args = parser.parse_args()

    # we can set here arguments that are considered as constant for this file (mnist_example.py)
    args.project_name = 'mnist'
    args.root_dir = '/home/shared/data/results'
    args.algorithm = 'MNISTAlgorithm'
    args.path_to_data = '/home/elad/projects/mnist'

    experiment = Experiment(args)

    # here we initialize the workers (can be single or multiple workers, depending on the configuration)
    experiment.run(run_mnist)
