## A PyTorch GNNs

This package contains a easy-to-use PyTorch implementation of [GCN](https://arxiv.org/pdf/1609.02907.pdf), [GraphSAGE](http://snap.stanford.edu/graphsage/), and [Graph Attention Network](https://arxiv.org/pdf/1710.10903.pdf). It can be easily imported and used like using logistic regression from sklearn. Two versions for supervised GNNs are provided: one implemented with only PyTorch, the other implemented with DGL and PyTorch.

Note: The unsupervised version is built upon our [GraphSAGE-pytorch](https://github.com/twjiang/graphSAGE-pytorch) implementation, and the DGL version is built upon the examples given by [DGL](https://github.com/dmlc/dgl/tree/master/examples/pytorch).

#### Authors of this code package:
[Tong Zhao](https://github.com/zhao-tong) (tzhao2@nd.edu),
[Tianwen Jiang](https://github.com/twjiang) (tjiang2@nd.edu).


## Important dependencies

- python==3.6.8
- pytorch==1.0.1.post2
- dgl==0.4.2


## Usage

**Parameters (GNNs_unsupervised):**
```
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
```
**Example Usage**

A detailed example of usage for unsupervised GNNs under different settings on the Cora dataset can be found in `example_usage.py`

To run the unsupervised GraphSAGE on Cuda:
```python
from GNNs_unsupervised import GNN
gnn = GNN(adj_matrix, features=raw_features, supervised=False, model='graphsage', device='cuda')
# train the model
gnn.fit()
# get the node embeddings with the trained model
embs = gnn.generate_embeddings()
```

**TODO**
Docs and examples for supervised GNNs will be added soon.
