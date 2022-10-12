# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:54:38 2022

@author: zll
"""

from torch_geometric.nn import global_mean_pool
import numpy as np
import networkx as nx
import os
import torchnet as tnt
import torch.nn.functional as F
import torch.nn as nn
import torch
import wget
import zipfile
# %%
wget.download(
    "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip")

file_name = 'MUTAG.zip'
zip_File = zipfile.ZipFile(file_name, 'r')
# 解压
zip_File.extractall()  # 括号内也可赋值解压后指定存储的文件夹名


# %%


def indices_to_one_hot(number, nb_classes, label_dummy=-1):
    """Convert an iterable of indices to one-hot encoded labels."""
    if number == label_dummy:
        return np.zeros(nb_classes)
    else:
        return np.eye(nb_classes)[number]


def get_graph_signal(nx_graph):
    d = dict((k, v) for k, v in nx_graph.nodes.items())
    x = []
    invd = {}
    j = 0
    for k, v in d.items():
        x.append(v['attr_dict'])
        invd[k] = j
        j = j + 1
    return np.array(x)


def load_data(path, ds_name, use_node_labels=True, max_node_label=10):
    node2graph = {}
    Gs = []
    data = []
    dataset_graph_indicator = f"{ds_name}_graph_indicator.txt"
    dataset_adj = f"{ds_name}_A.txt"
    dataset_node_labels = f"{ds_name}_node_labels.txt"
    dataset_graph_labels = f"{ds_name}_graph_labels.txt"

    path_graph_indicator = os.path.join(path, dataset_graph_indicator)
    path_adj = os.path.join(path, dataset_adj)
    path_node_lab = os.path.join(path, dataset_node_labels)
    path_labels = os.path.join(path, dataset_graph_labels)

    with open(path_graph_indicator, "r") as f:
        c = 1
        for line in f:
            node2graph[c] = int(line[:-1])
            if not node2graph[c] == len(Gs):
                Gs.append(nx.Graph())
            Gs[-1].add_node(c)
            c += 1

    with open(path_adj, "r") as f:
        for line in f:
            edge = line[:-1].split(",")
            edge[1] = edge[1].replace(" ", "")
            Gs[node2graph[int(edge[0])] -
               1].add_edge(int(edge[0]), int(edge[1]))

    if use_node_labels:
        with open(path_node_lab, "r") as f:
            c = 1
            for line in f:
                node_label = indices_to_one_hot(int(line[:-1]), max_node_label)
                Gs[node2graph[c] - 1].add_node(c, attr_dict=node_label)
                c += 1

    labels = []
    with open(path_labels, "r") as f:
        for line in f:
            labels.append(int(line[:-1]))

    return list(zip(Gs, labels))


def create_loaders(dataset, batch_size, split_id, offset=-1):
    train_dataset = dataset[:split_id]
    val_dataset = dataset[split_id:]
    return to_pytorch_dataset(train_dataset, offset, batch_size), to_pytorch_dataset(val_dataset, offset, batch_size)


def to_pytorch_dataset(dataset, label_offset=0, batch_size=1):
    list_set = []
    for graph, label in dataset:
        F, G = get_graph_signal(graph), nx.to_numpy_matrix(graph)
        numOfNodes = G.shape[0]
        F_tensor = torch.from_numpy(F).float()
        G_tensor = torch.from_numpy(G).float()

        # fix labels to zero-indexing
        if label == -1:
            label = 0

        label += label_offset

        list_set.append(tuple((F_tensor, G_tensor, label)))

    dataset_tnt = tnt.dataset.ListDataset(list_set)
    data_loader = torch.utils.data.DataLoader(
        dataset_tnt, shuffle=True, batch_size=batch_size)
    return data_loader


dataset = load_data(path='./MUTAG/', ds_name='MUTAG',
                    use_node_labels=True, max_node_label=7)
train_dataset, val_dataset = create_loaders(
    dataset, batch_size=1, split_id=150, offset=0)
print('Data are ready')
# %%


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h.shape: (batch,N, in_features),
        #Wh.shape: (batch,N, out_features)
        #e.shape: (batch,N, N)
        #h_prime:(batch,N, out_features)
        # Wh = torch.mm(h, self.W)
        Wh = torch.matmul(h, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (batch,N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (batch,N, 1)
        # e.shape: (batch,N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(-1, -2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nhid2, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(
            nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(
            nhid * nheads, nhid2, dropout=dropout, alpha=alpha, concat=False)
        self.fc = nn.Linear(nhid2, nclass)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        #x.shape: (batch,N, nhid * nheads)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        # print(x.shape)
        x = x.mean(dim=1)
        # print(x.shape)
        return F.elu(self.fc(x))


# %%


criterion = torch.nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Training on {device}')
model = GAT(nfeat=7, nhid=16, nhid2=4, nclass=2,
            dropout=0.6, alpha=0.2, nheads=8,).to(device)

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


def test(model,loader):
    model.eval()
    correct = 0
    for data in loader:
        X, A, labels = data
        X, A, labels = X.to(device), A.to(device), labels.to(device)
        # Forward pass.
        out = model(X, A)
        # Take the index of the class with the highest probability.
        pred = out.argmax(dim=1)
        # Compare with ground-truth labels.
        correct += int((pred == labels).sum())
    return correct / len(loader.dataset)


# main code :)

best_val = -1
for epoch in range(1, 241):
    # train(train_dataset)
    model.train()
    # optimizer.zero_grad()

    for data in train_dataset:
        with torch.no_grad():
            X, A, labels = data
            X, A, labels = X.to(device), A.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(X, A)
        loss = criterion(out, labels)
        loss.backward()
        # Updates the models parameters
        optimizer.step()

    # train_acc = test(train_dataset)
    val_acc = test(model,val_dataset)

    if val_acc > best_val:
        best_val = val_acc
        epoch_best = epoch

    if epoch % 10 == 0:
        print(
            f'Epoch: {epoch:03d},  Val Acc: {val_acc:.4f} || \
            Best Val Score: {best_val:.4f} (Epoch {epoch_best:03d}) ')
        # print(model.state_dict())
