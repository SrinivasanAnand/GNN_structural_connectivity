# Machine Learning for Cognitive Score Prediction
# Author: Anand Srinivasan
# Reddick Lab
# Description:
# Machine learning (random forest & svm) for sex classification tasks

import argparse
import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from sklearn.model_selection import train_test_split
from preprocessing_bright.preprocess import Bright
from preprocessing_dhcp.preprocess import DHCP
from combined_preprocessing import Bright_DHCP
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import numpy as np
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='combined')
parser.add_argument('--n_nodes', type=int, default=379)
parser.add_argument('--n_components', type=int, default=100)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()

# Other (non command line) args
args.score_type = 'sex'
args.n_estimators = 100
args.svm_kernel = 'rbf'
fix_seed(args.seed)

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
    tags = None
elif ('bright' in args.dataset):
    dataset = Bright(bright_parent_dir, bright_labels_filename,ags.n_nodes, args.score_type)
    external_dataset = DHCP(dhcp_parent_dir, dhcp_raw_dir, dhcp_labels_filepath, args.n_nodes, args.score_type)
    tags = None
elif ('combined' in args.dataset):
    dataset = Bright_DHCP(root_dir, bright_labels_filename, dhcp_labels_filename, args.n_nodes, args.score_type)
    external_dataset = None
    tags = np.array([data.tag for data in dataset])
else:
    print("Invalid dataset")
    exit(0)

def percent_ones(dataset):
    count = 0
    for data in dataset:
        if (data.y.item() == 1):
            count += 1
    return (count / len(dataset))

print("Percent female {}: {:.2f}".format(args.dataset, percent_ones(dataset) * 100))

labels = [d.y.item() for d in dataset]
train_indices, test_indices = train_test_split(list(range(len(labels))),
                        test_size=0.2,random_state=args.seed,shuffle=True, stratify=tags)
train_dataset = dataset[train_indices]
test_dataset = dataset[test_indices]
if (tags is not None):
    tag_test = tags[test_indices]

print("{} dataset loaded with train {} test {} splits".format(args.dataset, len(train_dataset), len(test_dataset)))

def get_x_y_data(dataset):
    tensor_x_data = torch.stack([data.x for data in dataset])
    x_data = tensor_x_data.view(tensor_x_data.size(0), -1).numpy()
    tensor_y_data = torch.tensor([data.y for data in dataset])
    y_data = tensor_y_data.numpy()
    x_data = x_data.astype(np.float64)
    y_data = y_data.astype(np.int32)
    return (x_data, y_data)

x_train, y_train= get_x_y_data(train_dataset)
x_test, y_test = get_x_y_data(test_dataset)
if (external_dataset is not None):
    x_external, y_external = get_x_y_data(external_dataset)

pca = PCA(n_components=args.n_components, random_state=args.seed)
pca_train = pca.fit_transform(x_train)
pca_test = pca.transform(x_test)
if (external_dataset is not None):
    pca_external = pca.transform(x_external)

rf_model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.seed, class_weight='balanced')
svm_model = SVC(kernel=args.svm_kernel, random_state=args.seed, class_weight='balanced')

rf_model.fit(pca_train, y_train)
svm_model.fit(pca_train, y_train)

# Make predictions on the test data
y_pred_rf = rf_model.predict(pca_test)
y_pred_svm = svm_model.predict(pca_test)
if (external_dataset is not None):
    y_pred_external_rf = rf_model.predict(pca_external)
    y_pred_external_svm = svm_model.predict(pca_external)

acc_rf = accuracy_score(y_test, y_pred_rf)
acc_svm = accuracy_score(y_test, y_pred_svm)
print("Accuracy score RF:", acc_rf)
print("Accuracy score SVM:", acc_svm)

if (external_dataset is not None):
    acc_external_rf = accuracy_score(y_external, y_pred_external_rf)
    acc_external_svm = accuracy_score(y_external, y_pred_external_svm)
    print("Accuracy score external RF:", acc_external_rf)
    print("Accuracy score external SVM:", acc_external_svm)

def acc_stratified(preds, labels, tags):
    bright_count = 0
    bright_correct = 0
    dhcp_count = 0
    dhcp_correct = 0
    for p, l, t in zip(preds, labels, tags):
        if (t == 'bright'):
            bright_count += 1
            if (p == l):
                bright_correct += 1
        else:
            dhcp_count += 1
            if (p == l):
                dhcp_correct += 1
    
    return (bright_correct / bright_count, dhcp_correct / dhcp_count)

if (tags is not None):
    acc_rf_bright, acc_rf_dhcp = acc_stratified(y_pred_rf, y_test, tag_test)
    acc_svm_bright, acc_svm_dhcp = acc_stratified(y_pred_svm, y_test, tag_test)
    print("Accuracy score RF: bright: {}, dhcp: {}".format(acc_rf_bright,acc_rf_dhcp))
    print("Accuracy score svm: bright: {}, dhcp: {}".format(acc_svm_bright,acc_svm_dhcp))


