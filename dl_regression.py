# Cognitive Score Prediction Regression Tasks
# Author: Anand Srinivasan
# Reddick Lab
# Some code adapted from NeuroGraph -- https://github.com/Anwar-Said/NeuroGraph

import os,random
import os.path as osp
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utils import *
from preprocessing.preprocess import DHCP
from preprocessing_bright.preprocess import Bright

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='combined')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--n_nodes', type-int, default=76)
parser.add_argument('--score_type', type=str, default='CPTOmZs')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--model', type=str, default="GCNConv")
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

# Other (non command line) args
args.batch_size = 16
args.early_stopping = 50
args.weight_decay = 0.0005
args.dropout = 0.5
args.num_features = dataset.num_features
args.num_classes = 1

path = "checkpoints/"
if not os.path.isdir(path):
    os.mkdir(path)

fix_seed(args.seed)

dhcp_parent_dir = "/research_jude/rgs01_jude/dept/DI/DIcode/Anand/Data/DHCP/"
dhcp_raw_dir = os.path.join(dhcp_parent_dir, "raw")
dhcp_labels_filename = "cognitivescores_135subjects.csv"
dhcp_labels_filepath = os.path.join(raw_dir, dhcp_labels_filename)

bright_parent_dir = "/research_jude/rgs01_jude/dept/DI/DIcode/Anand/Data/Bright/"
bright_labels_filename = "bright_cognitive_scores.csv"

if ('DHCP' in args.dataset):
    dataset = DHCP(dhcp_parent_dir, dhcp_raw_dir, dhcp_labels_filepath, args.n_nodes, args.score_type)
elif ('bright' in args.dataset):
    dataset = Bright(bright_parent_dir, bright_labels_filename, args.n_nodes, args.score_type)
else:
    print("Invalid dataset")
    exit(0)

print("Loaded dataset {} of size {}".format(args.dataset, len(dataset)))

# Split the data (70:10:20 -- train:val:test)
labels = [d.y.item() for d in dataset]
train_tmp, test_indices = train_test_split(list(range(len(labels))),
                        test_size=0.2,random_state=args.seed,shuffle=True)
tmp = dataset[train_tmp]
train_labels = [d.y.item() for d in tmp]
train_indices, val_indices = train_test_split(list(range(len(train_labels))),
 test_size=0.125,random_state=args.seed,shuffle = True)
train_dataset = tmp[train_indices]
val_dataset = tmp[val_indices]
test_dataset = dataset[test_indices]
print("dataset {} loaded with train {} val {} test {} splits".format(args.dataset,len(train_dataset), len(val_dataset), len(test_dataset)))

train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)


criterion = torch.nn.L1Loss()
def train(train_loader):
    model.train()
    total_loss = 0
    for data in train_loader:  
        data = data.to(args.device)
        out = model(data)  
        loss = criterion(out.squeeze(), data.y) 
        total_loss +=loss
        loss.backward()
        optimizer.step() 
        optimizer.zero_grad()
    return total_loss / len(train_loader)

@torch.no_grad()
def test(loader):
    model.eval()
    total_absolute_error = 0
    total_samples = 0
    for data in loader:  
        data = data.to(args.device)
        out = model(data)
        labels = data.y
        absolute_errors = torch.abs(out.squeeze() - labels)
        total_absolute_error += absolute_errors.sum().item()
        total_samples += data.num_graphs
    return total_absolute_error / total_samples   

seeds = [123] # keep the model seed constant, vary seed for data split
fix_seed(seeds[0])

# Initialize model
if (args.model == 'NN'):
    model = SimpleNN(args.num_features * args.num_features, args.num_classes)
elif (args.model == 'GCN'):
    model = GCN(args.num_features)
else:
    gnn = eval(args.model)
    model = ResidualGNNs(args, train_dataset, 32, 64, 3, gnn).to(args.device)

print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters is: {total_params}")

optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

val_loss_history, test_loss_history = [],[]
best_val_loss = 1e10 #set arbitrarily high at start

for epoch in range(args.epochs):
    loss = train(train_loader)
    val_loss= test(val_loader)
    test_loss= test(test_loader)

    print("epoch: {} -- loss: {}, val_loss: {}, test_loss:{}".format(
        epoch, np.round(loss.item(),6),np.round(val_loss,3), np.round(test_loss,3)))

    val_loss_history.append(val_loss)
    test_loss_history.append(test_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        if epoch > int(args.epochs/10): #do not save early in training
            torch.save(model.state_dict(), path + args.dataset+args.model+str(args.seed)+'task-checkpoint-best-loss.pkl')
    

# Load best model and report test loss
model.load_state_dict(torch.load(path + args.dataset+args.model+str(args.seed)+'task-checkpoint-best-loss.pkl'))
model.eval()
test_loss = test(test_loader)

print("Loss of best model on test set:", test_loss)
