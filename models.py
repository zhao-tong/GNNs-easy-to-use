import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DataLoader(object):
    def __init__(self, adj_matrix, raw_features, labels, supervised, device):
        super(DataLoader, self).__init__()
        self.adj_matrix = adj_matrix
        # load adjacency list and node features
        self.adj_list = self.get_adj_list(adj_matrix)
        if raw_features is None:
            features = self.get_features()
        else:
            features = raw_features
        assert features.shape[0] == len(self.adj_list) == self.adj_matrix.shape[0]
        self.features = torch.FloatTensor(features).to(device)
        self.nodes_train = list(range(len(self.adj_list)))
        if supervised:
            self.labels = np.asarray(labels)

    def get_adj_list(self, adj_matrix):
        """build adjacency list from adjacency matrix"""
        adj_list = {}
        for i in range(adj_matrix.shape[0]):
            adj_list[i] = set(np.where(adj_matrix[i].toarray() != 0)[1])
        return adj_list

    def get_features(self):
        """
        When raw features are not available,
        build one-hot degree features from the adjacency list.
        """
        max_degree = np.max(np.sum(self.adj_matrix != 0, axis=1))
        features = np.zeros((self.adj_matrix.shape[0], max_degree))
        for node, neighbors in self.adj_list.items():
            features[node, len(neighbors)-1] = 1
        return features


class Classification(nn.Module):
    def __init__(self, emb_size, num_classes):
        super(Classification, self).__init__()
        self.fc1 = nn.Linear(emb_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, embeds):
        x = F.elu(self.fc1(embeds))
        x = F.elu(self.fc2(x))
        logists = torch.log_softmax(x, 1)
        return logists


class Unsup_Loss(object):
    """docstring for UnsupervisedLoss"""
    def __init__(self, dl, device):
        super(Unsup_Loss, self).__init__()
        self.Q = 10
        self.N_WALKS = 4
        self.WALK_LEN = 4
        self.N_WALK_LEN = 5
        self.MARGIN = 3
        self.adj_lists = dl.adj_list
        self.adj_matrix = dl.adj_matrix
        self.train_nodes = dl.nodes_train
        self.device = device

        self.target_nodes = None
        self.positive_pairs = []
        self.negtive_pairs = []
        self.node_positive_pairs = {}
        self.node_negtive_pairs = {}
        self.unique_nodes_batch = []

    def get_loss_sage(self, embeddings, nodes):
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
        node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

        nodes_score = []
        assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negtive_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            # Q * Exception(negative score)
            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score = self.Q*torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)

            # multiple positive score
            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score = torch.log(torch.sigmoid(pos_score))

            nodes_score.append(torch.mean(- pos_score - neg_score).view(1,-1))

        loss = torch.mean(torch.cat(nodes_score, 0))
        return loss

    def get_loss_margin(self, embeddings, nodes):
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
        node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

        nodes_score = []
        assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negtive_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score, _ = torch.min(torch.log(torch.sigmoid(pos_score)), 0)

            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score, _ = torch.max(torch.log(torch.sigmoid(neg_score)), 0)

            nodes_score.append(torch.max(torch.tensor(0.0).to(self.device),
                                         neg_score-pos_score+self.MARGIN).view(1, -1))
        loss = torch.mean(torch.cat(nodes_score, 0), 0)
        return loss

    def extend_nodes(self, nodes, num_neg=6):
        self.positive_pairs = []
        self.node_positive_pairs = {}
        self.negtive_pairs = []
        self.node_negtive_pairs = {}

        self.target_nodes = nodes
        self.get_positive_nodes(nodes)
        self.get_negtive_nodes(nodes, num_neg)
        self.unique_nodes_batch = list(set([i for x in self.positive_pairs for i in x])
                                       | set([i for x in self.negtive_pairs for i in x]))
        assert set(self.target_nodes) < set(self.unique_nodes_batch)
        return self.unique_nodes_batch

    def get_positive_nodes(self, nodes):
        return self._run_random_walks(nodes)

    def get_negtive_nodes(self, nodes, num_neg):
        for node in nodes:
            neighbors = set([node])
            frontier = set([node])
            for _ in range(self.N_WALK_LEN):
                current = set()
                for outer in frontier:
                    current |= self.adj_lists[int(outer)]
                frontier = current - neighbors
                neighbors |= current
            far_nodes = set(self.train_nodes) - neighbors
            neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
            self.negtive_pairs.extend([(node, neg_node) for neg_node in neg_samples])
            self.node_negtive_pairs[node] = [(node, neg_node) for neg_node in neg_samples]
        return self.negtive_pairs

    def _run_random_walks(self, nodes):
        for node in nodes:
            if len(self.adj_lists[int(node)]) == 0:
                continue
            cur_pairs = []
            for _ in range(self.N_WALKS):
                curr_node = node
                for _ in range(self.WALK_LEN):
                    cnts = self.adj_matrix[int(curr_node)].toarray().squeeze()
                    neighs = []
                    for n in np.where(cnts != 0)[0]:
                        neighs.extend([n] * int(cnts[n]))
                    # neighs = self.adj_lists[int(curr_node)]
                    next_node = random.choice(list(neighs))
                    # self co-occurrences are useless
                    if next_node != node and next_node in self.train_nodes:
                        self.positive_pairs.append((node,next_node))
                        cur_pairs.append((node,next_node))
                    curr_node = next_node

            self.node_positive_pairs[node] = cur_pairs
        return self.positive_pairs


class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, input_size, out_size, gat=False):
        super(SageLayer, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.gat = gat
        self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gat else 2 * self.input_size))

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feats, aggregate_feats):
        """
        Generates embeddings for a batch of nodes.
        nodes	 -- list of nodes
        """
        if not self.gat:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)
        else:
            combined = aggregate_feats
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined

class Attention(nn.Module):
    """Computes the self-attention between pair of nodes"""
    def __init__(self, input_size, out_size):
        super(Attention, self).__init__()

        self.input_size = input_size
        self.out_size = out_size
        self.attention_raw = nn.Linear(2*input_size, 1, bias=False)
        self.attention_emb = nn.Linear(2*out_size, 1, bias=False)

    def forward(self, row_embs, col_embs):
        if row_embs.size(1) == self.input_size:
            att = self.attention_raw
        elif row_embs.size(1) == self.out_size:
            att = self.attention_emb
        e = att(torch.cat((row_embs, col_embs), dim=1))
        return F.leaky_relu(e, negative_slope=0.2)

class GraphSage(nn.Module):
    """docstring for GraphSage"""
    def __init__(self, num_layers, out_size, dl, device, gat=False, agg_func='MEAN'):
        super(GraphSage, self).__init__()

        self.input_size = dl.features.size(1)
        self.out_size = out_size
        self.num_layers = num_layers
        self.gat = gat
        self.device = device
        self.agg_func = agg_func

        self.raw_features = dl.features
        self.adj_lists = dl.adj_list
        self.adj_matrix = dl.adj_matrix

        for index in range(1, num_layers+1):
            layer_size = out_size if index != 1 else self.input_size
            setattr(self, 'sage_layer'+str(index), SageLayer(layer_size, out_size, gat=self.gat))
        if self.gat:
            self.attention = Attention(self.input_size, out_size)

    def forward(self, nodes_batch):
        """
        Generates embeddings for a batch of nodes.
        nodes_batch	-- batch of nodes to learn the embeddings
        """
        lower_layer_nodes = list(nodes_batch)
        nodes_batch_layers = [(lower_layer_nodes,)]
        for _ in range(self.num_layers):
            lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict= self._get_unique_neighs_list(lower_layer_nodes)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

        assert len(nodes_batch_layers) == self.num_layers + 1

        pre_hidden_embs = self.raw_features
        for index in range(1, self.num_layers+1):
            nb = nodes_batch_layers[index][0]
            pre_neighs = nodes_batch_layers[index-1]
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
            sage_layer = getattr(self, 'sage_layer'+str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_neighs)
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb], aggregate_feats=aggregate_feats)
            pre_hidden_embs = cur_hidden_embs

        return pre_hidden_embs

    def _nodes_map(self, nodes, neighs):
        _, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]
        return index

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        _set = set
        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
        samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        _unique_nodes_list = list(set.union(*samp_neighs))
        i = list(range(len(_unique_nodes_list)))
        # unique node 2 index
        unique_nodes = dict(list(zip(_unique_nodes_list, i)))
        return _unique_nodes_list, samp_neighs, unique_nodes

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs):
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs

        assert len(nodes) == len(samp_neighs)
        indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
        assert False not in indicator
        if not self.gat:
            samp_neighs = [(samp_neighs[i]-set([nodes[i]])) for i in range(len(samp_neighs))]
        if len(pre_hidden_embs) == len(unique_nodes):
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
        # get row and column nonzero indices for the mask tensor
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        # get the edge counts for each edge
        edge_counts = self.adj_matrix[nodes][:, unique_nodes_list].toarray()
        edge_counts = torch.FloatTensor(edge_counts).to(embed_matrix.device)
        if self.gat:
            indices = (torch.LongTensor(row_indices), torch.LongTensor(column_indices))
            nodes_indices = torch.LongTensor([unique_nodes[nodes[n]] for n in row_indices])
            row_embs = embed_matrix[nodes_indices]
            col_embs = embed_matrix[column_indices]
            atts = self.attention(row_embs, col_embs).squeeze()
            # mask = torch.sparse.FloatTensor(indices, atts, size=(len(samp_neighs),len(unique_nodes))).to_dense()
            mask = torch.zeros(len(samp_neighs), len(unique_nodes)).to(embed_matrix.device)
            mask.index_put_(indices, atts)
            mask = mask * edge_counts
            # softmax
            mask = torch.exp(mask) * (mask != 0).float()
            mask = F.normalize(mask, p=1, dim=1)
        else:
            mask = torch.zeros(len(samp_neighs), len(unique_nodes)).to(embed_matrix.device)
            mask[row_indices, column_indices] = 1
            # multiply edge counts to mask
            mask = mask * edge_counts
            mask = F.normalize(mask, p=1, dim=1)
            mask = mask.to(embed_matrix.device)

        if self.agg_func == 'MEAN':
            aggregate_feats = mask.mm(embed_matrix)
        elif self.agg_func == 'MAX':
            indexs = [x.nonzero() for x in mask != 0]
            aggregate_feats = []
            for feat in [embed_matrix[x.squeeze()] for x in indexs]:
                if len(feat.size()) == 1:
                    aggregate_feats.append(feat.view(1, -1))
                else:
                    aggregate_feats.append(torch.max(feat,0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)

        return aggregate_feats
