import sys
import math
import copy
import random
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix

from models import DataLoader, GraphSage, Unsup_Loss, Classification

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

class GNN(object):
    """Graph Neural Networks that can be easily called and used.

    Authors of this code package:
    Tong Zhao, tzhao2@nd.edu
    Tianwen Jiang, twjiang@ir.hit.edu.cn

    Last updated: 11/25/2019

    Parameters
    ----------
    adj_matrix: scipy.sparse.csr_matrix
        The adjacency matrix of the graph, where nonzero entries indicates edges.
        The number of each nonzero entry indicates the number of edges between these two nodes.

    features: numpy.ndarray, optional
        The 2-dimension np array that stores given raw feature of each node, where the i-th row
        is the raw feature vector of node i.
        When raw features are not given, one-hot degree features will be used.

    labels: list or 1-D numpy.ndarray, optional
        The class label of each node. Used for supervised learning.

    supervised: bool, optional, default False
        Whether to use supervised learning.

    model: {'gat', 'graphsage'}, default 'gat'
        The GNN model to be used.
        - 'graphsage' is GraphSAGE: https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
        - 'gat' is graph attention network: https://arxiv.org/pdf/1710.10903.pdf

    n_layer: int, optional, default 2
        Number of layers in the GNN

    emb_size: int, optional, default 128
        Size of the node embeddings to be learnt

    random_state, int, optional, default 1234
        Random seed

    device: {'cpu', 'cuda', 'auto'}, default 'auto'
        The device to use.

    epochs: int, optional, default 5
        Number of epochs for training

    batch_size: int, optional, default 20
        Number of node per batch for training

    lr: float, optional, default 0.7
        Learning rate

    unsup_loss_type: {'margin', 'normal'}, default 'margin'
        Loss function to be used for unsupervised learning
        - 'margin' is a hinge loss with margin of 3
        - 'normal' is the unsupervised loss function described in the paper of GraphSAGE

    print_progress: bool, optional, default True
        Whether to print the training progress
    """
    def __init__(self, adj_matrix, features=None, labels=None, supervised=False, model='gat', n_layer=2, emb_size=128, random_state=1234, device='auto', epochs=5, batch_size=20, lr=0.7, unsup_loss_type='margin', print_progress=True):
        super(GNN, self).__init__()
        # fix random seeds
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        # set parameters
        self.supervised = supervised
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.unsup_loss_type = unsup_loss_type
        self.print_progress = print_progress
        self.gat = True if model == 'gat' else False
        # set device
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # load data
        self.dl = DataLoader(adj_matrix, features, labels, supervised, self.device)

        self.gnn = GraphSage(n_layer, emb_size, self.dl, self.device, gat=self.gat)
        self.gnn.to(self.device)

        if supervised:
            n_classes = len(set(labels))
            self.classification = Classification(emb_size, n_classes)
            self.classification.to(self.device)

    def fit(self):
        train_nodes = copy.deepcopy(self.dl.nodes_train)

        if self.supervised:
            labels = self.dl.labels
            models = [self.gnn, self.classification]
        else:
            unsup_loss = Unsup_Loss(self.dl, self.device)
            models = [self.gnn]
            if self.unsup_loss_type == 'margin':
                num_neg = 6
            elif self.unsup_loss_type == 'normal':
                num_neg = 100

        for epoch in range(self.epochs):
            np.random.shuffle(train_nodes)

            params = []
            for model in models:
                for param in model.parameters():
                    if param.requires_grad:
                        params.append(param)
            optimizer = torch.optim.SGD(params, lr=self.lr)
            optimizer.zero_grad()
            for model in models:
                model.zero_grad()

            batches = math.ceil(len(train_nodes) / self.batch_size)
            visited_nodes = set()
            for index in range(batches):
                if not self.supervised and len(visited_nodes) == len(train_nodes):
                    # finish this epoch if all nodes are visited
                    break
                nodes_batch = train_nodes[index*self.batch_size:(index+1)*self.batch_size]
                # extend nodes batch for unspervised learning
                if not self.supervised:
                    nodes_batch = np.asarray(list(unsup_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
                visited_nodes |= set(nodes_batch)
                # feed nodes batch to the GNN and returning the nodes embeddings
                embs_batch = self.gnn(nodes_batch)
                # calculate loss
                if self.supervised:
                    # superivsed learning
                    logists = self.classification(embs_batch)
                    labels_batch = labels[nodes_batch]
                    loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
                    loss_sup /= len(nodes_batch)
                    loss = loss_sup
                else:
                    # unsupervised learning
                    if self.unsup_loss_type == 'margin':
                        loss_net = unsup_loss.get_loss_margin(embs_batch, nodes_batch)
                    elif self.unsup_loss_type == 'normal':
                        loss_net = unsup_loss.get_loss_sage(embs_batch, nodes_batch)
                    loss = loss_net

                if self.print_progress:
                    logging.info('Epoch: [{}/{}],Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, self.epochs, index+1, batches, loss.item(), len(visited_nodes), len(train_nodes)))

                loss.backward()
                for model in models:
                    nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad()
                for model in models:
                    model.zero_grad()

    def generate_embeddings(self):
        nodes = self.dl.nodes_train
        b_sz = 500
        batches = math.ceil(len(nodes) / b_sz)
        embs = []
        for index in range(batches):
            nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
            with torch.no_grad():
                embs_batch = self.gnn(nodes_batch)
            assert len(embs_batch) == len(nodes_batch)
            embs.append(embs_batch)
        assert len(embs) == batches
        embs = torch.cat(embs, 0)
        assert len(embs) == len(nodes)
        return embs.cpu().numpy()

    def predict(self):
        if not self.supervised:
            print('GNN.predict() is only supported for supervised learning.')
            sys.exit(0)
        nodes = self.dl.nodes_train
        b_sz = 500
        batches = math.ceil(len(nodes) / b_sz)
        preds = []
        for index in range(batches):
            nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
            with torch.no_grad():
                embs_batch = self.gnn(nodes_batch)
                logists = self.classification(embs_batch)
                _, predicts = torch.max(logists, 1)
                preds.append(predicts)
        assert len(preds) == batches
        preds = torch.cat(preds, 0)
        assert len(preds) == len(nodes)
        return preds.cpu().numpy()
