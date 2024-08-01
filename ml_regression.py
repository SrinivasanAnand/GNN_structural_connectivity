# Machine Learning for Cognitive Score Prediction
# Author: Anand Srinivasan
# Reddick Lab
# Description:
# Machine learning (random forest & svm) for cognitive score regression tasks

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
import numpy as np
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='bright')
parser.add_argument('--n_nodes', type=int, default=379)
parser.add_argument('--score_type', type=str, default='CPTOmZs')
parser.add_argument('--n_components', type=int, default=100)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()

# Other (non command line) args
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
    if (args.score_type != 'lswm' and args.score_type != 'dccs' and args.score_type != 'pcps'):
        print("Invalid score type for {} dataset".format(args.dataset))
        exit(0)
    dataset = DHCP(dhcp_parent_dir, dhcp_raw_dir, dhcp_labels_filepath, args.n_nodes, args.score_type)
elif ('bright' in args.dataset):
    if (args.score_type != 'CPTOmZs' and args.score_type != 'CodingZs' and args.score_type != 'DigitBwdSpanZs'):
        print("Invalid score type for {} dataset".format(args.dataset))
        exit(0)
    dataset = Bright(bright_parent_dir, bright_labels_filename,args.n_nodes, args.score_type)
else:
    print("Invalid dataset")
    exit(0)

labels = [d.y.item() for d in dataset]
train_indices, test_indices = train_test_split(list(range(len(labels))),
                        test_size=0.2,random_state=args.seed,shuffle=True)
train_dataset = dataset[train_indices]
test_dataset = dataset[test_indices]

print("{} dataset loaded with train {} test {} splits".format(args.dataset, len(train_dataset), len(test_dataset)))

def get_x_y_data(dataset):
    tensor_x_data = torch.stack([data.x for data in dataset])
    x_data = tensor_x_data.view(tensor_x_data.size(0), -1).numpy()
    tensor_y_data = torch.tensor([data.y for data in dataset])
    y_data = tensor_y_data.numpy()
    x_data = x_data.astype(np.float64)
    y_data = y_data.astype(np.int32)
    return (x_data, y_data)

x_train, y_train = get_x_y_data(train_dataset)
x_test, y_test = get_x_y_data(test_dataset)

pca = PCA(n_components=args.n_components, random_state=args.seed)
pca_train = pca.fit_transform(x_train)
pca_test = pca.transform(x_test)

rf_model = RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.seed)
svm_model = SVR(kernel=args.svm_kernel)

rf_model.fit(pca_train, y_train)
svm_model.fit(pca_train, y_train)

# Make predictions on the test data
y_pred_rf = rf_model.predict(pca_test)
y_pred_svm = svm_model.predict(pca_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mae_svm = mean_absolute_error(y_test, y_pred_svm)
print("Mean Absolute Error RF:", mae_rf)
print("Mean Absolute Error SVM:", mae_svm)

