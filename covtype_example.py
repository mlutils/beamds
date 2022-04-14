import torch
import torchvision
import torch.nn.functional as F

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

from beam import parser, Experiment
from beam import UniversalDataset
from beam import Algorithm
from beam import LinearNet
#from temp import  FT_transformer,Embed_fc
from Models_new import FT_transformer
import numpy as np
import ray
from ray import tune
from fast_transformers.builders import TransformerEncoderBuilder

#roy

def load_split_cov_data(SplitFreq = [0.8, 0.1, 0.1], randomstate = 0):
    # X_all, y_all = fetch_covtype(return_X_y=True)
    # X, y = {}, {}
    # X['train'], X['val'], y['train'], y['val'] = train_test_split(X_all, y_all, train_size=SplitFreq[0],
    #                                                               random_state=randomstate, stratify=y_all)
    # X['val'], X['test'], y['val'], y['test'] = train_test_split(X['val'], y['val'],
    #                                                             train_size=SplitFreq[1] / (1 - SplitFreq[0]),
    #                                                             random_state=randomstate, stratify=y['val'])

    X = {}
    y = {}
    dataset_name = 'covtype'
    for part in ['train', 'test', 'val']:
        X[part] = np.load(f'/home/elad/projects/rulnet/data/data/{dataset_name}/N_{part}.npy')
        y[part] = np.load(f'/home/elad/projects/rulnet/data/data/{dataset_name}/y_{part}.npy')

    # y_all = np.concatenate(list(y.values()))
    # X_all = np.concatenate(list(X.values()))

    return  X, y

def Transfrom_data(X,N_Numric = 10):
    Data_transformer = QuantileTransformer()
    X['train'][:,:N_Numric] = Data_transformer.fit_transform(X['train'][:,:N_Numric])
    X['val'][:,:N_Numric] = Data_transformer.transform(X['val'][:,:N_Numric])
    X['test'][:,:N_Numric] = Data_transformer.transform(X['test'][:,:N_Numric])
    return  X


class covtype_dataset(UniversalDataset):

    def __init__(self,Data,target,Datatype = 'train'):
        super().__init__()

        self.data = {'data':Data[Datatype],'targets':target[Datatype]}

    def __getitem__(self, index):

        return {'x': torch.tensor(self.data['data'][index],dtype=torch.float) , 'y': torch.from_numpy(self.data['targets'][index]).long()}

    def __len__(self):
        s = len(self.data['data'])
        return s


class CovNetAlgorithm(Algorithm):

    def __init__(self, *args, **argv):
        super().__init__(*args, **argv)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizers['net'].dense, gamma=0.99)

    def postprocess_epoch(self, sample, aux, results, epoch, train=True):

        x, y = sample['x'], sample['y']

        #results['images']['sample'] = x[:16].view(16, 1, 28, 28).data.cpu()

        if train:
            self.scheduler.step()
            results['scalar'][f'lr'] = self.optimizers['net'].dense.param_groups[0]['lr']

        aux = {}
        return aux, results

    def iteration(self, sample, aux, results, train=True):

        x, y = sample['x'], sample['y']

        x = x.view(len(x), -1)
        net = self.networks['net']
        opt = self.optimizers['net']

        y_hat = net(x)

        loss = F.cross_entropy(y_hat, y, reduction='mean')

        if train:

            opt.zero_grad()
            loss.backward()
            opt.step()

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['acc'].append(float((y_hat.argmax(1) == y).float().mean()))
        aux = {}

        return aux, results


def run_covtype(rank, world_size, experiment):


    X, y = load_split_cov_data([0.8,0.1,0.1])
    X = Transfrom_data(X)
    dataloader = {}
    for  v in ['train','test']:

        dataloader[v] = covtype_dataset(X,y,Datatype=v).dataloader(batch_size=experiment.batch_size,
                                                                       num_workers=experiment.cpu_workers,
                                                                       pin_memory=True)

    # choose your network
    N_classes = len(set(y['train']))
    N_numeric = 10
    Categoric_numbers = np.max(X['train'][:,N_numeric:],axis=0) + 1  # assume max number is number of categories -1



    #net = LinearNet(54, 256, N_classes, 4)
    net = FT_transformer(N_classes,N_numeric,Categoric_numbers,EmbeddingSize=experiment.emb_size,
                         N_heads=experiment.n_heads,N_transormer_L=experiment.nlayers_transformer,
                         dim_ff=experiment.dim_ff,Fast_Transformer=True)

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
    parser.add_argument('--emb-size', type=int, default=128, help='Size of embedding')
    parser.add_argument('--n-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--nlayers_transformer', type=int, default=4, help='Number of transformer layes')
    parser.add_argument('--dim-ff', type=int, default=2048, help='Transformer feed forward dimention')

    args = parser.parse_args()

    # we can set here arguments that are considered as constant for this file (mnist_example.py)
    args.project_name = 'covtype'
    args.root_dir = '/home/shared/data/results'
    args.algorithm = 'CovNetAlgorithm'
    args.identifier = 'check'
    args.path_to_data = '/home/elad/projects/covtype'

    experiment = Experiment(args)
    # here we initialize the workers (can be single or multiple workers, depending on the configuration)
    experiment.run(run_covtype)
