import math
import logging
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.nn.pytorch import GATConv, SAGEConv

class GNNs(object):
    def __init__(self, adj_matrix, features, labels, tvt_nids, cuda=-1, hidden_size=64, n_layers=1, epochs=200, seed=-1, lr=1e-2, weight_decay=5e-4, dropout=0.5, log=True, name='debug', gnn='gcn'):
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = epochs
        # create a logger, logs are saved to GNN-[name].log when name is not None
        if log:
            self.logger = self.get_logger(name)
        else:
            # disable logger if wanted
            self.logger = logging.getLogger()
        # config device (force device to cpu when cuda is not available)
        if not torch.cuda.is_available():
            cuda = -1
        self.device = torch.device(f'cuda:{cuda}' if cuda>=0 else 'cpu')
        # log all parameters to keep record
        all_vars = locals()
        self.log_parameters(all_vars)
        # fix random seeds if needed
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # load data
        self.load_data(adj_matrix, features, labels, tvt_nids, gnn)
        # setup the model
        if gnn == 'gcn':
            self.model = GCN_model(self.features.size(1),
                                   hidden_size,
                                   self.n_class,
                                   n_layers,
                                   F.relu,
                                   dropout)
        elif gnn == 'gsage':
            self.model = GraphSAGE_model(self.features.size(1),
                                        hidden_size,
                                        self.n_class,
                                        n_layers,
                                        F.relu,
                                        dropout,
                                        aggregator_type='gcn')
        elif gnn == 'gat':
            heads = ([8] * n_layers) + [1]
            self.model = GAT_model(self.features.size(1),
                                hidden_size,
                                self.n_class,
                                n_layers,
                                F.elu,
                                heads,
                                dropout,
                                attn_drop=0.6,
                                negative_slope=0.2)


    def load_data(self, adj, features, labels, tvt_nids, gnnlayer_type):
        """ preprocess data """
        # features (torch.FloatTensor)
        if isinstance(features, torch.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)
        self.features = F.normalize(self.features, p=1, dim=1)
        # labels (torch.LongTensor) and train/validation/test nids (np.ndarray)
        if isinstance(labels, np.ndarray):
            labels = torch.LongTensor(labels)
        self.labels = labels
        assert len(labels.size()) == 1
        self.train_nid = tvt_nids[0]
        self.val_nid = tvt_nids[1]
        self.test_nid = tvt_nids[2]
        # number of classes
        self.n_class = len(torch.unique(self.labels))
        # adj for training
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        adj = sp.csr_matrix(adj)
        self.adj = adj
        self.G = DGLGraph(self.adj)
        if gnnlayer_type == 'gcn':
            # normalization (D^{-1/2})
            degs = self.G.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(self.device)
            self.G.ndata['norm'] = norm.unsqueeze(1)

    def fit(self):
        """ train the model """
        # move data to device
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        # keep record of the best validation accuracy for early stopping
        best_val_acc = 0.
        # train model
        for epoch in range(self.n_epochs):
            model.train()
            logits = model(self.G, features)
            # losses
            loss = F.nll_loss(logits[self.train_nid], labels[self.train_nid])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # validate with original graph (without dropout)
            self.model.eval()
            with torch.no_grad():
                logits_eval = model(self.G, features)
            val_acc = self.eval_node_cls(logits_eval[self.val_nid], labels[self.val_nid])
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = self.eval_node_cls(logits_eval[self.test_nid], labels[self.test_nid])
                self.logger.info('Epoch [{:3}/{}]: loss {:.4f}, val acc {:.4f}, test acc {:.4f}'
                            .format(epoch+1, self.n_epochs, loss.item(), val_acc, test_acc))
            else:
                self.logger.info('Epoch [{:3}/{}]: loss {:.4f}, val acc {:.4f}'
                            .format(epoch+1, self.n_epochs, loss.item(), val_acc))
        # get final test result without early stop
        with torch.no_grad():
            logits_eval = model(self.G, features)
        test_acc_final = self.eval_node_cls(logits_eval[self.test_nid], labels[self.test_nid])
        # log both results
        self.logger.info('Final test acc with early stop: {:.4f}, without early stop: {:.4f}'
                    .format(test_acc, test_acc_final))
        return test_acc

    def log_parameters(self, all_vars):
        """ log all variables in the input dict excluding the following ones """
        del all_vars['self']
        del all_vars['adj_matrix']
        del all_vars['features']
        del all_vars['labels']
        del all_vars['tvt_nids']
        self.logger.info(f'Parameters: {all_vars}')

    @staticmethod
    def eval_node_cls(nc_logits, labels):
        """ evaluate node classification results """
        preds = torch.argmax(nc_logits, dim=1)
        correct = torch.sum(preds == labels)
        acc = correct.item() / len(labels)
        return acc

    @staticmethod
    def get_logger(name):
        """ create a nice logger """
        logger = logging.getLogger(name)
        # clear handlers if they were created in other runs
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        # create console handler add add to logger
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # create file handler add add to logger when name is not None
        if name is not None:
            fh = logging.FileHandler(f'GAug-{name}.log')
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
        return logger


class GAT_model(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 heads,
                 dropout,
                 attn_drop,
                 negative_slope):
        super(GAT_model, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GATConv(in_feats, n_hidden, heads[0], dropout, attn_drop, negative_slope, False, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GATConv(n_hidden * heads[i-1], n_hidden, heads[i], dropout, attn_drop, negative_slope, False, activation=activation))
        # output layer
        self.layers.append(GATConv(n_hidden * heads[-2], n_classes, heads[-1], dropout, attn_drop, negative_slope, False, activation=None))

    def forward(self, g, features):
        h = features
        for l in range(self.n_layers):
            h = self.layers[l](g, h).flatten(1)
        logits = self.layers[-1](g, h).mean(1)
        return F.log_softmax(logits, dim=1)


class GraphSAGE_model(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE_model, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=0., activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return F.log_softmax(h, dim=1)


class GCN_model(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN_model, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(in_feats, n_hidden, activation, 0.))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(n_hidden, n_hidden, activation, dropout))
        # output layer
        self.layers.append(GCNLayer(n_hidden, n_classes, None, dropout))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return F.log_softmax(h, dim=1)


class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        h = h * g.ndata['norm']
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'),
                     fn.sum(msg='m', out='h'))
        h = g.ndata.pop('h')
        # normalization by square root of dst degree
        h = h * g.ndata['norm']
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h



