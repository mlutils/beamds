import torch
from src.beam import UniversalDataset
import torch_geometric as tg
import numpy as np
from torch.nn.utils.rnn import PackedSequence, pack_sequence


class PositionalHarmonicExpansion(object):

    def __init__(self, xs, xe, delta=2, rank='cuda', n=100):

        self.xs = xs
        self.xe = xe

        delta = delta / (xe - xs)
        i = torch.arange(n+1, device=rank)[None, :]
        self.sigma = 2 * ((1 / delta) ** (i / n))

    def transform(self, x):

        l = len(x)
        x = x.unsqueeze(-1)

        sin = torch.sin(2 * np.pi / self.sigma * x)
        cos = torch.cos(2 * np.pi / self.sigma * x)

        out = torch.cat([sin, cos], dim=-1)
        return out


class IP_ID_Dataset(UniversalDataset):

    def __init__(self, data_torch, edge_tags, edges, edge_attribute, rank, feature_map, ip_id_window=2**10,
                 time_window=1800, num_hops=1):

        super().__init__()

        self.rank = rank
        self.ip_id_window = ip_id_window
        self.time_window = time_window
        self.data_torch = data_torch
        self.min_time = self.data_torch[:, feature_map['frame/time_sec']].min()
        self.max_time = self.data_torch[:, feature_map['frame/time_sec']].max()
        self.seq_delta = self.data_torch[:, feature_map['frame/time_sec_max_label']] - self.data_torch[:, feature_map[
                                                                                                              'frame/time_sec_min_label']]
        self.edge_tags = edge_tags
        self.edges = edges
        self.edge_attribute = edge_attribute
        self.graph = tg.data.Data(edge_index=self.edges, edge_attr=self.edge_attribute, y=self.edge_tags)

        self.mapping = self.data_torch[:, feature_map['cluster_id_sorted']].long().unique()
        self.clusters_num = len(self.mapping)

        self.num_hops = num_hops
        self.feature_map = feature_map

    def __len__(self):
        return self.clusters_num

    def __getitem__(self, idx):
        cluster_idx = self.mapping[torch.LongTensor(idx)]

        sub_indices_s, _, _, _ = tg.utils.k_hop_subgraph(cluster_idx, self.num_hops, self.graph.edge_index,
                                                         relabel_nodes=False,
                                                         num_nodes=None, flow='source_to_target')
        sub_indices_t, _, _, _ = tg.utils.k_hop_subgraph(cluster_idx, self.num_hops, self.graph.edge_index,
                                                         relabel_nodes=False,
                                                         num_nodes=None, flow='target_to_source')
        uniques = torch.cat([sub_indices_s, sub_indices_t]).unique().sort()[0]

        edge_index_unrelabeled, edge_attr = tg.utils.subgraph(uniques, self.graph.edge_index,
                                                              edge_attr=self.graph.edge_attr[[0, 3, 4]].T,
                                                              relabel_nodes=False, num_nodes=None)
        edge_index, edge_tag = tg.utils.subgraph(uniques, self.graph.edge_index, edge_attr=self.graph.y,
                                                 relabel_nodes=True,
                                                 num_nodes=None)

        clusters_idx = torch.zeros((len(self.data_torch),), device=self.rank)
        for un in uniques:
            clusters_idx += (self.data_torch[:, self.feature_map['cluster_id_sorted']] == un)
        data_chosen_flat = self.data_torch[clusters_idx.bool()]

        uniques, split_indexes = data_chosen_flat[:, self.feature_map['cluster_id_sorted']].unique(return_counts=True)
        uniques = uniques.long()
        split_indexes = list(split_indexes.cpu().numpy())

        sub_batch_elements = list(torch.split(data_chosen_flat, split_size_or_sections=split_indexes, dim=0))
        sub_batch_elements_packed = pack_sequence(sub_batch_elements, enforce_sorted=False)

        data_emb = sub_batch_elements_packed.data[:, [self.feature_map['ip/id_int_bin'],
                                                      self.feature_map['slope_bin']]].long()

        sub_batch_elements_emb_packed = PackedSequence(data_emb, batch_sizes=sub_batch_elements_packed.batch_sizes,
                                                       sorted_indices=sub_batch_elements_packed.sorted_indices,
                                                       unsorted_indices=sub_batch_elements_packed.unsorted_indices)
        results = {

            'emb_data': sub_batch_elements_emb_packed,
            'edge_index': edge_index,
            'edge_index_unrelabeled': edge_index_unrelabeled,
            'edge_tags': edge_tag,
            'edge_attr': edge_attr.long(),
            'uniques': uniques,
            'cluster_idx': cluster_idx,
            'len': len(data_chosen_flat),
            'n_seq': len(sub_batch_elements)
        }

        return results





import torch
from torch import nn
import torch_geometric as tg
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class HalfLinear(nn.Module):

    def __init__(self, l_in, l_out):

        super(HalfLinear, self).__init__()
        self.lin = nn.Linear(l_in, l_out - l_in)

    def forward(self, x):

        x2 = self.lin(x)
        y = torch.cat([x, x2], dim=-1)

        return y


class ResGCN(nn.Module):

    def __init__(self, l_in, l_out, dropout=0., normalize=True, add_self_loops=True, edge_dim=None):
        super(ResGCN, self).__init__()
        #         self.GCN1 = tg.nn.GCNConv(l_in, l_in, add_self_loops=add_self_loops, normalize=normalize)
        #         self.GCN2 = tg.nn.GCNConv(l_in, l_out,add_self_loops=add_self_loops, normalize=normalize)

        self.GCN1 = tg.nn.TransformerConv(l_in, l_in, heads=1, concat=True, beta=False, dropout=dropout,
                                          edge_dim=edge_dim, bias=True)
        self.GCN2 = tg.nn.TransformerConv(l_in, l_out, heads=1, concat=True, beta=False, dropout=dropout,
                                          edge_dim=edge_dim, bias=True)

        if l_in == l_out:
            self.identity = nn.Identity()
        elif l_out > l_in:
            self.identity = HalfLinear(l_in, l_out)
        else:
            self.identity = nn.Linear(l_in, l_out)

    def forward(self, x, edge_index, edge_attr_emb, u=None, batch=None):

        r = F.relu(x)
        r = self.GCN1(r, edge_index, edge_attr_emb)
        r = F.relu(r)
        r = self.GCN2(r, edge_index, edge_attr_emb)

        x = self.identity(x)

        return x + r


class ResLin(nn.Module):

    def __init__(self, l_in, l_out):
        super(ResLin, self).__init__()
        if l_in == l_out:
            self.identity = nn.Identity()
        elif l_out > l_in:
            self.identity = HalfLinear(l_in, l_out)
        else:
            self.identity = nn.Linear(l_in, l_out)
        self.base = nn.Sequential(nn.BatchNorm1d(l_in), nn.ReLU(), nn.Linear(l_in, l_in),
                                  nn.BatchNorm1d(l_in), nn.ReLU(), nn.Linear(l_in, l_out))

    def forward(self, x):
        r = self.base(x)
        x = self.identity(x)

        return x + r


class EdgeModel(nn.Module):

    def __init__(self, n_f, e_a_in, e_a_out):
        super(EdgeModel, self).__init__()

        if e_a_in == e_a_out:
            self.identity = nn.Identity()
        elif e_a_out > e_a_in:
            self.identity = HalfLinear(e_a_in, e_a_out)
        else:
            self.identity = nn.Linear(e_a_in, e_a_out)

        l_in = e_a_in + 2 * n_f
        self.base = nn.Sequential(nn.BatchNorm1d(l_in), nn.ReLU(), nn.Linear(l_in, l_in),
                                  nn.BatchNorm1d(l_in), nn.ReLU(), nn.Linear(l_in, e_a_out))

    def forward(self, source, target, edge_attr, u=None, batch=None):

        ea = torch.cat([edge_attr, source, target], dim=1)

        r = self.base(ea)
        ea = self.identity(edge_attr)

        return ea + r


class ClusterModel(nn.Module):

    def __init__(self, l_in, l_h, l_out, n,
                 edge_attr_num_emb, edge_union_num_emb, edge_intersection_num_emb,
                 edge_ip_id_diff_num_emb, edge_time_diff_num_emb, emb_dict, num_embeddings,
                 edge_drop=0.0, dropout=0., normalize=True, add_self_loops=True,):

        super(ClusterModel, self).__init__()

        self.base = nn.Sequential(nn.Linear(l_in, l_h), nn.ReLU())
        self.emb = nn.Embedding(num_embeddings, 2 * n, sparse=True)
        self.lstm = nn.LSTM(input_size=l_h, hidden_size=l_h, num_layers=1,
                            dropout=dropout, bidirectional=True, batch_first=True)
        self.emb_dict = emb_dict
        self.edge_dim = 3 * np.sum(list(emb_dict.values()))

        self.GCN3 = tg.nn.MetaLayer(
            node_model=ResGCN(l_h, l_out, add_self_loops=add_self_loops, normalize=normalize, edge_dim=self.edge_dim),
            edge_model=EdgeModel(l_h, self.edge_dim, self.edge_dim))

        self.GCN1 = ResGCN(2 * l_h, l_h, add_self_loops=add_self_loops, normalize=normalize, edge_dim=self.edge_dim)

        self.QuantileWeightedEmbedding = QuantileWeightedEmbedding(edge_attr_num_emb, self.emb_dict)
        self.QuantileWeightedEmbeddingIpIdDiff = QuantileWeightedEmbedding(edge_ip_id_diff_num_emb, self.emb_dict)
        self.QuantileWeightedEmbeddingTimeDiff = QuantileWeightedEmbedding(edge_time_diff_num_emb, self.emb_dict)

        self.edge_attr_num_emb = edge_attr_num_emb
        self.edge_ip_id_diff_num_emb = edge_ip_id_diff_num_emb
        self.edge_time_diff_num_emb = edge_time_diff_num_emb

        self.n_embeddings = 2 * n
        self.edge_drop = edge_drop

    def forward(self, pm, edge_index, x_emb, edge_attr):
        edge_attr_emb = self.QuantileWeightedEmbedding(
            torch.clamp(edge_attr[:, 0] // 2, max=self.edge_attr_num_emb - 1))

        edge_ip_id_diff_emb = self.QuantileWeightedEmbeddingIpIdDiff(torch.clamp(edge_attr[:, 1] // 2,
                                                                                 max=self.edge_ip_id_diff_num_emb - 1))
        edge_time_diff_emb = self.QuantileWeightedEmbeddingTimeDiff(torch.clamp(edge_attr[:, 2] // 2,
                                                                                max=self.edge_time_diff_num_emb - 1))

        edge_attr_emb_all = torch.cat([edge_attr_emb, edge_ip_id_diff_emb, edge_time_diff_emb], dim=-1)
        b, n_features = x_emb.shape
        x_emb = self.emb(x_emb).view(-1, n_features * self.n_embeddings)

        x = x_emb
        x_new = self.base(x)
        x_new = PackedSequence(data=x_new, batch_sizes=pm.batch_sizes.to('cpu'), sorted_indices=pm.sorted_indices,
                               unsorted_indices=pm.unsorted_indices)
        y, _ = self.lstm(x_new)
        y_unpacked, lens_unpacked = pad_packed_sequence(y, batch_first=True)
        y_unpacked_sum = y_unpacked.sum(dim=1) / lens_unpacked.to(y_unpacked.device).unsqueeze(1)

        x_emb = self.GCN1(y_unpacked_sum, edge_index, edge_attr_emb_all)

        x_emb, _, _ = self.GCN3(x_emb, edge_index, edge_attr_emb_all)

        return x_emb


class Discriminator(nn.Module):

    def __init__(self, l_in, l_h, drop=0.0):
        super(Discriminator, self).__init__()

        self.base = nn.Sequential(ResLin(l_in, l_h), ResLin(l_h, l_h), nn.ReLU(), nn.Linear(l_h, 1))
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.drop(x)
        y = self.base(x).squeeze(1)
        return y


class QuantileWeightedEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dict, reduction='none'):
        super(QuantileWeightedEmbedding, self).__init__()

        '''
        Embedding dict should be <window size>: <embedding dim>
        if reduction is not none, all embedding dim must be equal
        '''

        self.embeddings = nn.ParameterDict(
            {str(k): nn.Parameter(torch.FloatTensor(num_embeddings, i).normal_()) for k, i in embedding_dict.items()})
        self.flag = True
        self.weighted_embeddings = {k: None for k in self.embeddings.keys()}
        self.reduction = reduction

        def hook(self, grad_input, grad_output):
            self.flag = True

        self.register_backward_hook(hook)

    def update_weights(self):

        if self.flag:

            for k, w in self.embeddings.items():
                kernel = int(k)
                w = F.pad(w, ((kernel - 1) // 2, (kernel - 1) // 2), "constant", 0).unfold(1, kernel, 1).mean(dim=-1)
                self.weighted_embeddings[k] = w

            self.flag = False

    def forward(self, x):

        self.update_weights()
        y = []
        for w in self.weighted_embeddings.values():
            y.append(w[x])

        n = len(x.shape) + 1
        if self.reduction == 'none':
            y = torch.cat(y, dim=-1)
        elif self.reduction == 'sum':
            y = torch.stack(y, dim=n).sum(dim=n)
        elif self.reduction == 'mean':
            y = torch.stack(y, dim=n).mean(dim=n)
        else:
            return NotImplementedError

        return y