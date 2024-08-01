# DHCP dataset preprocessing
# Author: Anand Srinivasan
# Reddick Lab
# Description:
# PyTorch Geometric Dataset for combined DHCP dataset

import torch
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from .graphPreprocessLib import graph, get_patient_id_from_filename, get_label_from_patient_id, has_label, pack_adjaceny_matrix
import math
# import networkx as nx
# import matplotlib
# import matplotlib.pyplot as plt

data_dir = "/research_jude/rgs01_jude/dept/DI/DIcode/Anand/Data/DHCP/"
raw_dir = os.path.join(data_dir, 'raw')
processed_dir = os.path.join(data_dir, 'processed')
labels_csv_filename = "cognitivescores_135subjects.csv"
labels_csv_path = os.path.join(raw_dir, labels_csv_filename)

def combine_csvs(raw_dir, labels1, labels2, outfile):
    labels_path1 = os.path.join(raw_dir, labels1)
    labels_path2 = os.path.join(raw_dir, labels2)
    labels1_df = pd.read_csv(labels_path1)
    labels2_df = pd.read_csv(labels_path2)

    combined = pd.concat([labels1_df, labels2_df])
    combined = combined.drop_duplicates(subset=['src_subject_id'])

    out_path = os.path.join(raw_dir, outfile)
    combined.to_csv(out_path, index=False)

def make_labels_csv(raw_dir, labels_filename, outfile):
    labels_excel_path = os.path.join(raw_dir, labels_filename)
    single_score_csv_path = os.path.join(raw_dir, outfile)
    #labels_df = pd.read_excel(labels_excel_path, engine='openpyxl')
    labels_df = pd.read_csv(labels_excel_path)
    single_score_df = pd.DataFrame(columns=["src_subject_id", 'age_years', 'sex', 'lswm', 'pcps', 'dccs'])
    raw_graph_data_dir = os.path.join(raw_dir, '{}nodes'.format(76))

    for filename in os.listdir(raw_graph_data_dir):
        full_path = os.path.join(raw_graph_data_dir, filename)

        if os.path.isfile(full_path):
            patient_id = get_patient_id_from_filename(filename=filename)
            if (has_label(patient_id, labels_df)):
                label_lswm = get_label_from_patient_id(patient_id, labels_df, score_type='lswm')
                label_pcps = get_label_from_patient_id(patient_id, labels_df, score_type='pcps')
                label_dccs = get_label_from_patient_id(patient_id, labels_df, score_type='dccs')
                #age_years = round(get_label_from_patient_id(patient_id, labels_df, score_type='interview_age') / 12, 1)
                age_years = get_label_from_patient_id(patient_id, labels_df, score_type='Age_years')
                sex = get_label_from_patient_id(patient_id, labels_df, score_type='sex')
                single_score_df = single_score_df.append({"src_subject_id": patient_id, 'age_years': age_years, 'sex': sex,
                                                        'lswm': label_lswm, 'pcps': label_pcps, 'dccs': label_dccs}, ignore_index=True)
    
    single_score_df.to_csv(single_score_csv_path, index=False)

def process_raw_data(raw_data_dir, labels_full_path, n_nodes=76, score='lswm'):
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
    
    base_scores = ['lswm', 'dccs', 'pcps']
    clf_scores = ['sex', 'group']
    binary_scores = [score + '_binary' for score in base_scores]
    if (not(score in base_scores) and not(score in clf_scores) and not(score in binary_scores)):
        print("Invalid Score", score)
        exit(0)

    labels_df = pd.read_csv(labels_full_path)
    patient_graphs = []
    raw_graph_data_dir = os.path.join(raw_data_dir, '{}nodes'.format(n_nodes))

    for filename in os.listdir(raw_graph_data_dir):
        full_path = os.path.join(raw_graph_data_dir, filename)

        if os.path.isfile(full_path):

            patient_id = get_patient_id_from_filename(filename=filename)
            if (has_label(patient_id, labels_df)):
                dtype = torch.float
                if (score == 'sex'):
                    patient_label = 0 if get_label_from_patient_id(patient_id, labels_df, score_type='sex') == 'M' else 1
                    dtype = torch.long
                elif (score == 'group'):
                    patient_label = 1
                    dtype = torch.long
                elif (score in binary_scores):
                    patient_label_score = get_label_from_patient_id(patient_id, labels_df, score_type=score.split('_')[0])
                    if (math.isnan(patient_label_score)):
                        continue
                    patient_label = 0 if patient_label_score < 104 else 1
                    dtype = torch.long
                else:
                    patient_label = get_label_from_patient_id(patient_id, labels_df, score_type=score)
                patient_age = get_label_from_patient_id(patient_id, labels_df, score_type='age_years')
                patient_sex = 0 if get_label_from_patient_id(patient_id, labels_df, score_type='sex') == 'M' else 1
                patient_df = pd.read_csv(full_path)
                patient_graph = graph(patient_df, n_regions=n_nodes, y=torch.tensor([patient_label], dtype=dtype), sex=patient_sex)
                patient_graphs.append(patient_graph)
    
    return patient_graphs

class DHCP(InMemoryDataset):
    def __init__(self, root, raw_data_dir, labels_path, n_nodes, score, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.raw_data_dir = raw_data_dir
        self.labels_path = labels_path
        self.n_nodes = n_nodes
        self.score = score
        self.processed_data_dir = os.path.join(os.path.join(self.root, 'processed'), '{}{}'.format(self.score, self.n_nodes))
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
        data_list = process_raw_data(self.raw_data_dir, self.labels_path, self.n_nodes, self.score)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

        print("Downloaded data to {}".format(self.processed_paths[0]))


# make_labels_csv(raw_dir, raw_csv_scores, "cognitivescores_49subjects.csv")
# process_raw_data(raw_dir, labels_csv_path)    
# dataset = DHCP(data_dir, raw_dir, labels_csv_path, 76, 'lswm')

# print('Number of graphs: {}'.format(len(dataset)))

# example_graph = dataset[10]

# print("Example graph: ")
# print(example_graph)
# print("--------------")
# print("Edge index info:")
# print(example_graph.edge_index)

# print("Age and sex:")
# print(example_graph.age)
# print(example_graph.sex)

# fig = plt.figure()
# g = to_networkx(example_graph, to_undirected=True)
# nx.draw(g, ax=fig.add_subplot())
# matplotlib.use("Agg") 
# fig.savefig("graph.png")