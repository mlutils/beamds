from torch.utils.data import  DataLoader
import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from src.beam import parser, Experiment
from src.beam import UniversalBatchSampler
from src.beam import Algorithm
from src.beam import LinearNet
import numpy as np
from collections import namedtuple
import os
import pandas as pd
from nat_src.dataset import IP_ID_Dataset
from nat_src.dataset import ClusterModel, Discriminator
import pickle
from nat_src.utils import Tracker, isin, ar_collate
import torch_geometric as tg


class NATGraphAlg(Algorithm):

    def __init__(self, networks, dataloader, experiment, remove_small_clusters=False, rank=0, optimizers=None, feature_map=None):

        self.packed_metadata = namedtuple('packed_metadata', ('batch_sizes', 'sorted_indices', 'unsorted_indices'))
        self.remove_small_clusters = remove_small_clusters
        self.exp = experiment
        self.feature_map = feature_map
        self.pos_track = {}

        self.n_lst = []
        for n in range(120):
            epoch_dir = os.path.join(self.exp.epoch_dir_base, f'epoch_{n:04}/')
            if 'train' in os.listdir(epoch_dir):
                self.n_lst.append(n)

        super().__init__(networks, dataloader, experiment, rank=rank, optimizers=optimizers)

    def preprocess_epoch(self, aux, epoch, train=True):

        train_flag = 'train' if train else 'test'
        if train:
            epoch_dir = os.path.join(self.exp.epoch_dir_base, f'epoch_{self.n_lst[epoch % len(self.n_lst)]:04}/{train_flag}')
        else:
            epoch_dir = self.exp.epoch_dir_test

        if epoch == 0 or train:

            file = open(os.path.join(epoch_dir, 'data.pkl'), 'rb')
            data = pickle.load(file).to(self.device)
            file = open(os.path.join(epoch_dir, 'edge_tags.pkl'), 'rb')
            edge_tags = pickle.load(file).to(self.device)
            file = open(os.path.join(epoch_dir, 'edges.pkl'), 'rb')
            edges = pickle.load(file).to(self.device)
            file = open(os.path.join(epoch_dir, 'edge_attribute.pkl'), 'rb')
            edge_attribute = pickle.load(file).to(self.device)

            if self.remove_small_clusters:
                file = open(os.path.join(epoch_dir, 'df.pkl'), 'rb')
                df = pickle.load(file)

                cluster_size = df.groupby('cluster_id_sorted')['index'].count().reset_index()
                cluster_size.columns = ['cluster_id_sorted', 'count']
                big_clusters = torch.LongTensor(
                    cluster_size[cluster_size['count'] > 4]['cluster_id_sorted'].values)

                data = data[isin(data[:, self.feature_map['cluster_id_sorted']].to('cpu'), big_clusters)]
                edges_org = edges.clone()
                edges, edge_attribute = tg.utils.subgraph(big_clusters, edges_org, edge_attr=edge_attribute.T,
                                                          relabel_nodes=False,
                                                          num_nodes=None)
                edge_attribute = edge_attribute.T
                _, edge_tags = tg.utils.subgraph(big_clusters, edges_org, edge_attr=edge_tags,
                                                 relabel_nodes=False, num_nodes=None)

            datasets = IP_ID_Dataset(data, edge_tags, edges, edge_attribute, self.exp.device, self.feature_map)

            self.dataloader[train_flag] = datasets.dataloader(batch_size=self.exp.batch_size, num_workers=0,
                                                              collate_fn=ar_collate)

        return aux

    def iteration(self, sample, aux, results, train=True):

        net = self.networks['model']
        opt_net = self.optimizers['model']

        discriminator = self.networks['discriminator']
        opt_disc = self.optimizers['discriminator']

        pm = self.packed_metadata(batch_sizes=sample['emb_data'].batch_sizes,
                                  sorted_indices=sample['emb_data'].sorted_indices,
                                  unsorted_indices=sample['emb_data'].unsorted_indices)

        emb = sample['emb_data'].data
        edge_index = sample['edge_index']
        edge_tags = sample['edge_tags']
        edge_attr = sample['edge_attr']

        x_emb = net(pm, edge_index, emb, edge_attr)
        pairs = x_emb[edge_index].permute(1, 0, 2).reshape(-1, 2 * self.exp.l_h)
        out = discriminator(pairs)

        pos_weight = (edge_tags.shape[0] - edge_tags.sum()) / edge_tags.sum()

        if not train in self.pos_track:
            self.pos_track[train] = Tracker(alpha=1 / 50, x=pos_weight)
        else:
            pos_weight = self.pos_track[train].update(pos_weight)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight,
                                         reduction='sum')
        loss = criterion(out, edge_tags.float()) / float(sample['n_seq'])

        if train:
            opt_net.zero_grad()
            opt_disc.zero_grad()
            loss.backward()
            opt_net.step()
            opt_disc.step()

        y_pred = (out > 0).int()
        y_true = edge_tags
        accu = float((y_pred == y_true).sum()) / y_true.shape[0]
        tp = float((y_true * y_pred).sum())
        tn = float(((1 - y_true) * (1 - y_pred)).sum())
        fp = float(((1 - y_true) * y_pred).sum())
        fn = float((y_true * (1 - y_pred)).sum())
        epsilon = 1e-7
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)

        results_dic = {'accuracy': accu, 'precision': precision, 'recall': recall, 'f1_score': f1}
        results['scalar']['loss'].append(float(loss.detach()))
        pos_edge = float(edge_tags.sum() / edge_tags.numel())
        results['scalar']['pos_edges'].append(pos_edge)
        results['scalar']['pos_weight'].append(float(pos_weight))
        results['scalar']['batch_len'].append(sample['len'])
        results['scalar']['n_seq'].append(sample['n_seq'])

        for k, v in results_dic.items():
            results['scalar'][k].append(v)

        return aux, results


def run_nat(rank, world_size, experiment):

    time_sec_diff_bins = pd.read_pickle(os.path.join(experiment.bins_folder, 'time_sec_diff_bins.pkl'))
    ip_diff_bins = pd.read_pickle(os.path.join(experiment.bins_folder, 'ip_diff_bins.pkl'))
    slope_bins = pd.read_pickle(os.path.join(experiment.bins_folder, 'slope_bins.pkl'))
    cosine_similaity_bins = pd.read_pickle(os.path.join(experiment.bins_folder, 'cosine_similaity_bins.pkl'))
    intersections_bins = pd.read_pickle(os.path.join(experiment.bins_folder, 'intersections_bins.pkl'))
    unions_bins = pd.read_pickle(os.path.join(experiment.bins_folder, 'unions_bins.pkl'))
    clusters_ip_id_diff_bins = pd.read_pickle(
        os.path.join(experiment.bins_folder, 'clusters_ip_id_diff_bins.pkl'))
    clusters_time_diff_bins = pd.read_pickle(
        os.path.join(experiment.bins_folder, 'clusters_time_diff_bins.pkl'))

    num_embeddings = len(time_sec_diff_bins) + len(ip_diff_bins) + len(slope_bins)

    features = ['frame/time_sec', 'ip/id_int', 'cluster_id_sorted', 'frame/time_sec_min', 'frame/time_sec_max',
                'frame/time_sec_min_label', 'frame/time_sec_max_label', 'cluster_center', 'index', 'ip/id_int_diff',
                'frame/time_sec_diff', 'slope']

    emb_features = ['frame/time_sec_bin', 'ip/id_int_bin', 'slope_bin']
    quantile_emb_dict = {1: 16, 3: 8}

    all_features = features + emb_features
    feature_map = {k: i for i, k in enumerate(all_features)}

    model = ClusterModel(experiment.l_in_factor * experiment.n_fourier,
                         experiment.l_h, experiment.l_out, experiment.n_fourier,
                         edge_attr_num_emb=len(cosine_similaity_bins),
                         edge_union_num_emb=len(unions_bins),
                         edge_intersection_num_emb=len(intersections_bins),
                         edge_ip_id_diff_num_emb=len(clusters_ip_id_diff_bins),
                         edge_time_diff_num_emb=len(clusters_time_diff_bins),
                         emb_dict=quantile_emb_dict,
                         add_self_loops=True,
                         num_embeddings=num_embeddings,
                         dropout=experiment.dropout)

    discriminator = Discriminator(2 * experiment.l_out, experiment.l_out)

    # we recommend to use the algorithm argument to determine the type of algorithm to be used
    Alg = globals()[experiment.algorithm]
    alg = Alg({'model': model, 'discriminator': discriminator}, {}, experiment, feature_map=feature_map)

    experiment.writer_control(not bool(rank), networks=None)

    for results in iter(alg):
        experiment.save_model_results(results, alg,
                                      print_results=True, visualize_results='yes',
                                      store_results='logscale', store_networks='logscale',
                                      visualize_weights=False,
                                                    argv={'images': {'sample': {'dataformats': 'NCHW'}}})


if __name__ == '__main__':

    # here you put all actions which are performed only once before initializing the workers
    # for example, setting running arguments and experiment:

    parser.add_argument('--n-fourier', type=int, default=25,
                        help='number of fourier bins for harmonic expansion')
    parser.add_argument('--l-in-factor', type=int, default=4,
                        help='length of input is l_in_factor * n_fourier')
    parser.add_argument('--l-out', type=int, default=32,
                        help='length of output layer')
    parser.add_argument('--l-h', type=int, default=32,
                        help='length of hidden layer')
    parser.add_argument('--batch-size', type=int, default=60, help='Batch Size')

    parser.add_argument('--total-steps', type=int, default=200000, help='Total number of environment steps')
    parser.add_argument('--epoch-length', type=int, default=100, help='Length of each epoch')

    parser.add_argument('--algorithm', type=str, default='NATGraphAlg', help='algorithm name')

    args = parser.parse_args()

    # we can set here arguments that are considered as constant for this file (mnist_example.py)
    args.project_name = 'nat'

    args.root_dir = '/home/shared/data/results'
    args.store_path = '/home/uri/notebooks/nat/elad_data/dns'
    args.epoch_dir_base = '/home/uri/notebooks/nat/elad_data/cyclic_augmentations'
    args.bins_folder = '/home/uri/notebooks/nat/elad_data/bins_quantiles_20'
    args.epoch_dir_test = '/home/uri/notebooks/nat/elad_data/dns_quantile_20/test'

    args.data_dir = '/home/uri/notebooks/nat/elad_data'

    experiment = Experiment(args)

    # here we initialize the workers (can be single or multiple workers, depending on the configuration)
    experiment.run(run_nat)

