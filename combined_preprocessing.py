# Combined DHCP, Bright dataset preprocessing
# Author: Anand Srinivasan
# Reddick Lab
# Description:
# PyTorch Geometric Dataset for combined DHCP, Bright dataset

import preprocessing_dhcp.preprocess as dhcp_preprocess
import preprocessing_bright.preprocess as bright_preprocess
import torch
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader

class Bright_DHCP(InMemoryDataset):
    def __init__(self, root, bright_label_filename, dhcp_label_filename, n_nodes, score, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.bright_root = os.path.join(self.root, 'Bright')
        self.DHCP_root = os.path.join(self.root, 'DHCP')
        self.bright_raw_data_dir = os.path.join(os.path.join(self.bright_root, '{}nodes'.format(n_nodes)), 'raw')
        self.bright_labels_path = os.path.join(self.bright_root, bright_label_filename)
        self.DHCP_raw_data_dir = os.path.join(self.DHCP_root, 'raw')
        self.DHCP_labels_path = os.path.join(self.DHCP_raw_data_dir, dhcp_label_filename)
        self.n_nodes = n_nodes
        self.score = score
        self.processed_data_dir = os.path.join(os.path.join(
            os.path.join(self.root, 'combined'), 'processed'), '{}{}'.format(self.n_nodes, self.score))
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    @property
    def processed_dir(self):
        return self.processed_data_dir
    
    def process(self):
        data_list_bright = bright_preprocess.process_raw_data(self.bright_raw_data_dir, self.bright_labels_path, n_nodes=self.n_nodes, score=self.score)
        for data in data_list_bright:
            data.tag = 'bright'
        data_list_dhcp = dhcp_preprocess.process_raw_data(self.DHCP_raw_data_dir, self.DHCP_labels_path, self.n_nodes, self.score)
        for data in data_list_dhcp:
            data.tag = 'dhcp'
        data_list = data_list_bright + data_list_dhcp

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

        print("Downloaded data to {}".format(self.processed_paths[0]))


