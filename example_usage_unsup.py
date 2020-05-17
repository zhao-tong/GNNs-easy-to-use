import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from GNNs_unsupervised import GNN

def example_with_cora():
    """load cora dataset"""
    cora_content_file = 'cora/cora.content'
    cora_cite_file = 'cora/cora.cites'
    # load features and labels
    feat_data = []
    labels = [] # label sequence of node
    node_map = {} # map node to Node_ID
    label_map = {} # map label to Label_ID
    with open(cora_content_file, 'r') as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data.append([float(x) for x in info[1:-1]])
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels.append(label_map[info[-1]])
    raw_features = np.asarray(feat_data)
    labels = np.asarray(labels, dtype=np.int64)
    # load adjacency matrix
    row = []
    col = []
    with open(cora_cite_file, 'r') as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            assert len(info) == 2
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            row.extend([paper1, paper2])
            col.extend([paper2, paper1])
    row = np.asarray(row)
    col = np.asarray(col)
    adj_matrix = csr_matrix((np.ones(len(row)), (row, col)), shape=(len(node_map), len(node_map)))

    """
    Example of using GraphSAGE for supervised learning.
    using CPU and do not print training progress
    """
    gnn = GNN(adj_matrix, features=raw_features, labels=labels, supervised=True, model='graphsage', device='cpu', print_progress=False)
    # train the model
    gnn.fit()
    # make predictions with the built-in MLP classifier and evaluate
    preds = gnn.predict()
    f1 = f1_score(labels, preds, average='micro')
    print(f'F1 score for supervised learning on Cora dataset: {f1:.4f}')
    embs = gnn.generate_embeddings()

    """
    Example of using Graph Attention Network for unsupervised learning.
    using CUDA and print training progress
    """
    gnn = GNN(adj_matrix, features=raw_features, supervised=False, model='gat', device='cuda')
    # train the model
    gnn.fit()
    # get the node embeddings with the trained GAT
    embs = gnn.generate_embeddings()
    # evaluate the embeddings with logistic regression
    lr = LogisticRegression(penalty='l2', random_state=0, solver='liblinear')
    preds = lr.fit(embs, labels).predict(embs)
    f1 = f1_score(labels, preds, average='micro')
    print(f'F1 score for unsupervised learning on Cora dataset: {f1:.4f}')

if __name__ == "__main__":
    example_with_cora()
