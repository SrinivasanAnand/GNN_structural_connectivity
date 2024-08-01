# NeuroGraph Residual Network Model
# Author: Anwar Said, https://github.com/Anwar-Said/NeuroGraph
# Adapted by Anand Srinivasan for GNN Structural Connectivity project
# Reddick Lab
# Description:
# Residual neural networks for classification and regression tasks

import torch
from torch.nn import Linear
from torch import nn
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn import aggr
import torch.nn.functional as F
from torch_geometric.nn import APPNP, MLP, GCNConv, GINConv, SAGEConv, GraphConv, TransformerConv, ChebConv, GATConv, SGConv, GeneralConv
from torch.nn import Conv1d, MaxPool1d, ModuleList, Flatten
from torch_geometric.utils import to_dense_batch
import random
import numpy as np
softmax = torch.nn.LogSoftmax(dim=1)

class ResidualGNNs(torch.nn.Module):
    def __init__(self,args, train_dataset, hidden_channels,hidden, num_layers, GNN, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        num_features = train_dataset.num_features
        if args.model=="ChebConv":
            if num_layers>0:
                self.convs.append(GNN(num_features, hidden_channels,K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels,K=5))
        else:
            if num_layers>0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))
        
        input_dim1 = int(((num_features * num_features)/2) + (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features)/2) + (num_features/2))
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]        
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        h = []
        for i, xx in enumerate(xs):
            if i== 0:
                xx = xx.reshape(data.num_graphs, x.shape[1],-1)
                #x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
                #x = torch.stack([t.triu().flatten() for t in xx])
                n = x.shape[1]
                triu_indices = torch.triu_indices(n, n)
                x = torch.stack([t.flatten()[triu_indices[0] * n + triu_indices[1]] for t in xx])
                x = self.bn(x)
            else:
                xx = self.aggr(xx,batch)
                h.append(xx)
        
        h = torch.cat(h,dim=1)
        h = self.bnh(h)
        x = torch.cat((x,h),dim=1)
        x = self.mlp(x)
        return x

class ResidualGNNClfs(torch.nn.Module):
    def __init__(self,args, train_dataset, hidden_channels,hidden, num_layers, GNN, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        num_features = train_dataset.num_features
        if args.model=="ChebConv":
            if num_layers>0:
                self.convs.append(GNN(num_features, hidden_channels,K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels,K=5))
        else:
            if num_layers>0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))
        
        input_dim1 = int(((num_features * num_features)/2) + (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features)/2) + (num_features/2))
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]        
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        h = []
        for i, xx in enumerate(xs):
            if i== 0:
                xx = xx.reshape(data.num_graphs, x.shape[1],-1)
                #x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
                #x = torch.stack([t.triu().flatten() for t in xx])
                n = x.shape[1]
                triu_indices = torch.triu_indices(n, n)
                x = torch.stack([t.flatten()[triu_indices[0] * n + triu_indices[1]] for t in xx])
                x = self.bn(x)
            else:
                xx = self.aggr(xx,batch)
                h.append(xx)
        
        h = torch.cat(h,dim=1)
        h = self.bnh(h)
        x = torch.cat((x,h),dim=1)
        x = self.mlp(x)
        return softmax(x)