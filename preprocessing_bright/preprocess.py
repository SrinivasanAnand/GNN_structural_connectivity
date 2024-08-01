# Bright dataset preprocessing
# Author: Anand Srinivasan
# Reddick Lab
# Description:
# PyTorch Geometric Dataset for combined Bright dataset

import torch
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from .graphPreprocessLib import graph, get_patient_id_from_filename, get_label_from_patient_id, has_label
import math

parent_dir = "/research_jude/rgs01_jude/dept/DI/DIcode/Anand/Data/Bright/"
# data_dir = "/research_jude/rgs01_jude/dept/DI/DIcode/Anand/Data/Bright/76nodes/"
# raw_dir = os.path.join(data_dir, 'raw')
# processed_dir = os.path.join(data_dir, 'processed')
# labels_csv_filename = "bright_list.csv"
# labels_csv_path = os.path.join(raw_dir, labels_csv_filename)
labels_filename = 'BRIGHT_DataRequest_NCOG_data.xlsx'
outfile = 'bright_cognitive_scores_with_smk.csv'

def make_labels_csv(parent_dir, labels_filename, outfile):
    labels_excel_path = os.path.join(parent_dir, labels_filename)
    single_score_csv_path = os.path.join(parent_dir, outfile)
    labels_df = pd.read_excel(labels_excel_path, engine='openpyxl')
    #labels_df = pd.read_csv(labels_excel_path)
    single_score_df = pd.DataFrame(columns=["MRN", 'group', 'sex', 'CPTOmZs', 'CodingZs', 'DigitBwdSpanZs', 'smk_gp'])
    data_dir = os.path.join(parent_dir, '{}nodes'.format(76))
    raw_graph_data_dir = os.path.join(data_dir, 'raw')

    for filename in os.listdir(raw_graph_data_dir):
        full_path = os.path.join(raw_graph_data_dir, filename)

        if (filename == 'bright_list.csv'):
            continue

        if os.path.isfile(full_path):
            patient_id = get_patient_id_from_filename(filename=filename)
            if (has_label(patient_id, labels_df)):
                label_CPTOmZs = get_label_from_patient_id(patient_id, labels_df, score_type='CPTOmZs')
                label_CodingZs = get_label_from_patient_id(patient_id, labels_df, score_type='CodingZs')
                label_DigitBwdSpanZs = get_label_from_patient_id(patient_id, labels_df, score_type='DigitBwdSpanZs')
                label_smoker = get_label_from_patient_id(patient_id, labels_df, score_type='smk_gp')
                #age_years = round(get_label_from_patient_id(patient_id, labels_df, score_type='interview_age') / 12, 1)
                group = get_label_from_patient_id(patient_id, labels_df, score_type='Group')
                sex = get_label_from_patient_id(patient_id, labels_df, score_type='sex1')
                single_score_df.loc[len(single_score_df)] = {"MRN": patient_id, 'group': group, 'sex': sex,
                'CPTOmZs': label_CPTOmZs, 'CodingZs': label_CodingZs, 'DigitBwdSpanZs': label_DigitBwdSpanZs, 'smk_gp': label_smoker}
    
    single_score_df.to_csv(single_score_csv_path, index=False)

# make_labels_csv(parent_dir, labels_filename, outfile)

def process_raw_data(raw_data_dir, labels_full_path, n_nodes=379, score='CPTOmZs'):
    '''Processes all patient csv files in raw_data_dir.
    Creates PyTorch Geormetric Data (Graph) object for each
    patient with edges and edge weights determined by each
    patient csv file and labels determined by labels csv.
    Returns list of processed graphs.

    Arg(s):
        raw_data_dir: string
            directory containing all patient csv files
        processed_data_dir: string
            directory where processed data will be stored
        labels_full_path: string
            path to labels csv file
    '''

    if (n_nodes != 76 and n_nodes != 379):
        print("Invalid Num Nodes", n_nodes)
        exit(0)
    
    base_scores = ['CPTOmZs', 'CodingZs', 'DigitBwdSpanZs']
    clf_scores = ['sex', 'group', 'smk_gp']
    binary_scores = [score + '_binary' for score in base_scores]
    quaternary_scores = [score + '_quaternary' for score in base_scores]
    if (not(score in base_scores) and not(score in clf_scores) and not(score in binary_scores) and not(score in quaternary_scores)):
        print("Invalid Score", score)
        exit(0)
    
    labels_df = pd.read_csv(labels_full_path)
    patient_graphs = []

    for filename in os.listdir(raw_data_dir):
        full_path = os.path.join(raw_data_dir, filename)

        if os.path.isfile(full_path):

            if ('bright' in full_path):
                continue

            patient_id = get_patient_id_from_filename(filename=filename)
            if (has_label(patient_id, labels_df)):
                dtype = torch.long
                if (score == 'sex'):
                    patient_label = 0 if get_label_from_patient_id(patient_id, labels_df, score_type='sex') == 'Male' else 1
                elif (score == 'group'):
                    patient_label = get_label_from_patient_id(patient_id, labels_df, score_type='group') - 1
                elif (score == 'smk_gp'):
                    patient_label = 0 if get_label_from_patient_id(patient_id, labels_df, score_type='smk_gp') == 'Never' else 1
                elif (score in binary_scores or score in quaternary_scores):
                    patient_label_score = get_label_from_patient_id(patient_id, labels_df, score_type=score.split('_')[0])
                    if (math.isnan(patient_label_score)):
                        continue
                    if (score.split('_')[1] == 'binary'):
                        patient_label = 0 if patient_label_score < 0.2 else 1
                    elif (score.split('_')[1] == 'quaternary'):
                        if (patient_label_score < -1):
                            patient_label = 0
                        elif (patient_label_score < 0):
                            patient_label = 1
                        elif (patient_label_score < 1):
                            patient_label = 2
                        else:
                            patient_label = 3
                else:
                    patient_label = get_label_from_patient_id(patient_id, labels_df, score_type=score)
                    patient_label = 100 + 15 * patient_label
                    dtype = torch.float
                if (math.isnan(patient_label)):
                    continue
                patient_group = get_label_from_patient_id(patient_id, labels_df, score_type='group')
                patient_sex = 0 if get_label_from_patient_id(patient_id, labels_df, score_type='sex') == 'Male' else 1
                patient_df = pd.read_csv(full_path)
                patient_graph = graph(patient_df, n_regions=n_nodes, feature_type='adj_matrix', y=torch.tensor([patient_label], dtype=dtype), sex=patient_sex)
                patient_graphs.append(patient_graph)
    
    return patient_graphs

class Bright(InMemoryDataset):
    def __init__(self, root, label_filename, n_nodes, score, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.raw_data_dir = os.path.join(os.path.join(root, '{}nodes'.format(n_nodes)), 'raw')
        self.labels_path = os.path.join(root, label_filename)
        self.n_nodes = n_nodes
        self.score = score
        self.processed_data_dir = os.path.join(os.path.join(
            os.path.join(root, '{}nodes'.format(n_nodes)), 'processed'), score)
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
        data_list = process_raw_data(self.raw_data_dir, self.labels_path, n_nodes=self.n_nodes, score=self.score)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

        print("Downloaded data to {}".format(self.processed_paths[0]))
    

# bright_dataset = Bright(parent_dir, 'bright_cognitive_scores.csv', 379, 'CodingZs')

# print("Num samples: ", len(bright_dataset))
# # Shuffle and split the dataset
# torch.manual_seed(12345)
# dataset = bright_dataset.shuffle()
# train_split = 0.85
# bright_dataset = bright_dataset.shuffle()
# train_split_idx = int(train_split * len(bright_dataset))
# train_dataset = dataset[:train_split_idx]
# test_dataset = dataset[train_split_idx:]

# print(f'Number of training graphs: {len(train_dataset)}')
# print(f'Number of test graphs: {len(test_dataset)}')

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# for batch in train_loader:
#     data_list = batch.to_data_list()
#     for data in data_list:
#         print(" ----------------------- ")
#         print("Graph: ")
#         print(data)
#     break

