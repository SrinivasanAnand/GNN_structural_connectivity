# Deep Learning for Sex Classification
# Author: Anand Srinivasan
# Reddick Lab
# Some code adapted from NeuroGraph -- https://github.com/Anwar-Said/NeuroGraph
# Description:
# Deep learning (GNNs & NN) for sex classification tasks
# Models implemented in models/models.py and models/neurograph_residual_network.py

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
from preprocessing_dhcp.preprocess import DHCP
from preprocessing_bright.preprocess import Bright
from combined_preprocessing import Bright_DHCP
from models.models import *
from models.neurograph_residual_network import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='combined')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--n_nodes', type=int, default=379)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--model', type=str, default="GCNConv")
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

# Other (non command line) args
args.score_type = 'sex'
args.batch_size = 16
args.early_stopping = 50
args.weight_decay = 0.0005
args.dropout = 0.5
args.num_features = args.n_nodes
args.num_classes = 2

path = "checkpoints/"
if not os.path.isdir(path):
    os.mkdir(path)

fix_seed(args.seed)

#Load datasets
root_dir = "/research_jude/rgs01_jude/dept/DI/DIcode/Anand/Data/"

dhcp_parent_dir = "/research_jude/rgs01_jude/dept/DI/DIcode/Anand/Data/DHCP/"
dhcp_raw_dir = os.path.join(dhcp_parent_dir, "raw")
dhcp_labels_filename = "cognitivescores_135subjects.csv"
dhcp_labels_filepath = os.path.join(dhcp_raw_dir, dhcp_labels_filename)

bright_parent_dir = "/research_jude/rgs01_jude/dept/DI/DIcode/Anand/Data/Bright/"
bright_labels_filename = "bright_cognitive_scores.csv"


if ('DHCP' in args.dataset):
    dataset = DHCP(dhcp_parent_dir, dhcp_raw_dir, dhcp_labels_filepath, args.n_nodes, args.score_type)
    external_dataset = None
elif ('bright' in args.dataset):
    dataset = Bright(bright_parent_dir, bright_labels_filename,args.n_nodes, args.score_type)
    external_dataset = DHCP(dhcp_parent_dir, dhcp_raw_dir, dhcp_labels_filepath, args.n_nodes, args.score_type)
elif ('combined' in args.dataset):
    dataset = Bright_DHCP(root_dir, bright_labels_filename, dhcp_labels_filename, args.n_nodes, args.score_type)
    external_dataset = None
    tags = [data.tag for data in dataset]
else:
    print("Invalid dataset")
    exit(0)

print("Loaded dataset {} of size {}".format(args.dataset, len(dataset)))

# Split the data (70:10:20 -- train:val:test), statifying if applicable
stratify = tags if args.dataset == 'combined' else None
labels = [d.y.item() for d in dataset]
if (stratify is not None):
    train_tmp, test_indices = train_test_split(list(range(len(labels))),
    test_size=0.2,random_state=args.seed,shuffle=True, stratify=stratify)
else:
    train_tmp, test_indices = train_test_split(list(range(len(labels))),
    test_size=0.2,random_state=args.seed,shuffle=True)

tmp = dataset[train_tmp]
train_labels = [d.y.item() for d in tmp]

if (stratify is not None):
    train_indices, val_indices = train_test_split(list(range(len(train_labels))),
    test_size=0.125,random_state=args.seed,shuffle=True, stratify=[stratify[i] for i in train_tmp])
else:
    train_indices, val_indices = train_test_split(list(range(len(train_labels))),
    test_size=0.125,random_state=args.seed,shuffle=True)

train_dataset = tmp[train_indices]
val_dataset = tmp[val_indices]
test_dataset = dataset[test_indices]
print("dataset {} loaded with train {} val {} test {} splits".format(args.dataset,len(train_dataset), len(val_dataset), len(test_dataset)))

train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
if (external_dataset is not None):
    external_loader = DataLoader(external_dataset, args.batch_size, shuffle=False)

# Define train, test, and relevant accuracy/loss statistic functions
def train(model, train_loader, criterion):
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
def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:  
        data = data.to(args.device)
        out = model(data)  
        pred = out.argmax(dim=1)  
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

#equally weighted average of test accuracy for bright and dhcp data
@torch.no_grad()
def test_weighted(model, loader):
    model.eval()
    bright_correct = 0
    bright_count = 0
    dhcp_correct = 0
    dhcp_count = 0
    for data in loader:  
        data = data.to(args.device)
        out = model(data)  
        pred = out.argmax(dim=1)  
        for p, label, tag in zip(pred, data.y, data.tag):
            if (tag == 'bright'):
                bright_count += 1
                if (p == label):
                    bright_correct += 1
            elif (tag == 'dhcp'):
                dhcp_count += 1
                if (p == label):
                    dhcp_correct += 1

    weighted_acc = (bright_correct / bright_count + dhcp_correct / dhcp_count) / 2
    return weighted_acc

@torch.no_grad()
def test_loss(model, loader, criterion):
    model.eval()
    total_loss = 0
    for data in loader:  
        data = data.to(args.device)
        out = model(data)  
        loss = criterion(out, data.y)
        total_loss += loss
    return total_loss/len(loader)

# equally weighted loss for dhcp and bright sets
@torch.no_grad()
def test_loss_weighted(model, loader, criterion):
    model.eval()
    bright_total_loss = 0
    dhcp_total_loss = 0
    bright_count = 0
    dhcp_count = 0
    for data in loader:  
        data = data.to(args.device)
        out = model(data)
        for o, label, tag in zip(out, data.y, data.tag):
            if (tag == 'bright'):
                bright_count += 1
                bright_total_loss += criterion(o, label)
            elif (tag == 'dhcp'):
                dhcp_count += 1
                dhcp_total_loss += criterion(o, label)
    
    weighted_loss = ((bright_total_loss/ bright_count) + (dhcp_total_loss / dhcp_count) ) / 2
    return weighted_loss

@torch.no_grad()
def test_stratified(model, loader):
    model.eval()
    bright_correct = 0
    bright_count = 0
    dhcp_correct = 0
    dhcp_count = 0
    for data in loader:  
        data = data.to(args.device)
        out = model(data)  
        pred = out.argmax(dim=1)  
        for p, label, tag in zip(pred, data.y, data.tag):
            if (tag == 'bright'):
                bright_count += 1
                if (p == label):
                    bright_correct += 1
            elif (tag == 'dhcp'):
                dhcp_count += 1
                if (p == label):
                    dhcp_correct += 1

    return (bright_correct / bright_count, dhcp_correct / dhcp_count)

seeds = [123] # keep the model seed constant, only vary seed for data split
fix_seed(seeds[0])

# Initialize model
if (args.model == 'NN'):
    model = SimpleNNClf(args.num_features * args.num_features, args.num_classes)
elif (args.model == 'GCN'):
    model = GCNClf(args.num_features)
else:
    gnn = eval(args.model)
    model = ResidualGNNClfs(args, train_dataset, 32, 64, 3, gnn).to(args.device)

print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters is: {total_params}")

# Criterion, optimizer
class_sample_count = np.array([len(np.where(np.array(train_labels) == t)[0]) for t in np.unique(np.array(train_labels))])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in np.array(train_labels)])
class_weights = torch.FloatTensor(weight)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

val_loss_history, test_loss_history = [],[]
val_acc_history, test_acc_history = [],[]
best_val_loss = 1e10 #set arbitrarily high at start

# Train model
for epoch in range(args.epochs):
    loss = train(model, train_loader, criterion)
    val_loss = test_loss(model, val_loader, criterion)
    val_acc = test(model, val_loader)
    test_acc = test(model, test_loader)

    if (args.dataset == 'combined'):
        weighted_val_loss = test_loss_weighted(model, val_loader, criterion)
        weighted_val_acc = test_weighted(model, val_loader)
        bright_test_acc, dhcp_test_acc = test_stratified(model, test_loader)

    if (external_dataset is not None):
        ext_acc = test(model, external_loader)

    if (args.dataset == 'combined'):
        print("epoch: {}, loss: {}, val_loss:{}, val_loss_weighted:{}, val_acc:{}, val_acc_weighted: {}, test_acc:{}, bright test acc {}, dhcp test acc {}".format(
        epoch, np.round(loss.item(),6), np.round(val_loss.item(), 4), np.round(weighted_val_loss.item(), 4),
        np.round(val_acc,4),np.round(weighted_val_acc, 4), np.round(test_acc,4), np.round(bright_test_acc, 4),
        np.round(dhcp_test_acc, 4)))
    elif (args.dataset == 'bright'):
        print("epoch: {}, loss: {}, val_loss:{}, val_acc:{}, test_acc:{}, ext_acc: {}".format(
        epoch, np.round(loss.item(),6), np.round(val_loss.item(), 4),
        np.round(val_acc,4), np.round(test_acc,4), np.round(ext_acc, 4)))
    else:
        print("epoch: {}, loss: {}, val_loss:{}, val_acc:{}, test_acc:{}".format(
        epoch, np.round(loss.item(),6), np.round(val_loss.item(), 4),
        np.round(val_acc,4), np.round(test_acc,4)))
    
    if (args.dataset == 'combined'):
        selective_val_loss = weighted_val_loss
        selective_val_acc = weighted_val_acc
    else:
        selective_val_loss = val_loss
        selective_val_acc = val_acc
    
    val_loss_history.append(selective_val_loss)
    val_acc_history.append(selective_val_acc)
    
    if selective_val_loss < best_val_loss:
        best_val_loss = selective_val_loss
        best_val_acc = selective_val_acc
        if epoch> int(args.epochs/10): ## save the best model
            torch.save(model.state_dict(), path + args.dataset+args.model+str(args.seed)+'task-checkpoint-best-acc.pkl')
    elif selective_val_loss == best_val_loss and selective_val_acc > best_val_acc:
        best_val_acc = selective_val_acc
        if epoch > int(args.epochs/10):
            torch.save(model.state_dict(), path + args.dataset+args.model+str(args.seed)+'task-checkpoint-best-acc.pkl')

    
# Load best model and report test loss
model.load_state_dict(torch.load(path + args.dataset+args.model+str(args.seed)+'task-checkpoint-best-acc.pkl'))
model.eval()

test_acc = test(test_loader)
print("Accuracy of best model on test set: ", test_acc)

if (args.dataset == 'combined'):
    bright_test_acc, dhcp_test_acc = test_stratified(test_loader)
    print("Accuracy of best model on bright statification of test set: ", bright_test_acc)
    print("Accuracy of best model on dhcp stratification of test set: ", dhcp_test_acc)

if (external_dataset is not None):
    ext_acc = test(model, external_loader)
    print("Accuracy of best model on external set: ", ext_acc)

