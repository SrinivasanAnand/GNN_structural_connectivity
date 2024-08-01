# Simple GCN and MLP models
# Author: Anand Srinivasan
# Reddick Lab
# Description:
# Simple (3 layer) GCN model and (3 layer) MLP for classification and regression tasks

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

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x = data.x.view(data.batch.max().item() + 1, -1)
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class SimpleNNClf(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNNClf, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x = data.x.view(data.batch.max().item() + 1, -1)
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return softmax(x)

class GCN(torch.nn.Module):
    def __init__(self, n_node_features):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(n_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)

        self.lin = Linear(64, 32)
        self.lin2 = Linear(32, 1)

    def forward(self, data):

        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv3(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = x.relu()

        x = global_mean_pool(x, batch)

        x = self.lin(x)
        x = x.relu()
        output_logits = self.lin2(x)

        return output_logits

class GCNClf(torch.nn.Module):
    def __init__(self, n_node_features):
        super(GCNClf, self).__init__()

        self.conv1 = GCNConv(n_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)

        self.lin = Linear(64,32)
        self.lin2 = Linear(32, 2)

    def forward(self, data):

        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv3(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = x.relu()

        x = global_mean_pool(x, batch)

        x = self.lin(x)
        x = x.relu()
        output_logits = self.lin2(x)

        return softmax(output_logits)