# DHCP dataset graph preprocessing library
# Author: Anand Srinivasan
# Reddick Lab
# Description:
# Useful functions for PyTorch graph/dataset creation
# from raw adj matrix and other patient info data

import pandas as pd
import numpy as np
import os
import torch
from torch_geometric.data import Data


raw_dir = "/research_jude/rgs01_jude/dept/DI/DIcode/Anand/Data/raw"

def get_filename_from_patient_id(patient_id, file_suffix='_76nodes_connectivity'):
    '''Returns csv filename of adjacency matrix
    csv for patient with patient id: patient_id

    Arg(s):
        patient_id: string
            patient id number as string
    '''

    return patient_id + file_suffix + ".csv"

def get_patient_id_from_filename(filename):
    '''Returns patient id corresponding to input filename. Assumes
    filename is of the form: id_file_suffix.csv

    Arg(s):
        patient_id: string
            patient id number as string
    '''
    id = filename.split('_')[0]
    return id

def has_label(patient_id, labels_df):
    return (labels_df['src_subject_id'] == patient_id).any()

def get_label_from_patient_id(patient_id, labels_df, score_type):
    '''Returns label for patient with patient id: patient_id
    as desdcribed by label dataframe.

    Arg(s):
        patient_id: string
            patient id number as string
        labels_df: pandas dataframe
            dataframe listing patient ids and correponding groups
    '''
    label = labels_df.loc[labels_df['src_subject_id'] == patient_id, score_type].values[0]
    return label

def get_node_features(node_idx, feature_type='adj_matrix', n_regions=76):
    '''Returns node features for node with node index: node_idx

    Arg(s):
        node_idx: int
            node index in range 0-75
        feature_type: string
            type of features to be associated with each node / brain region
        n_regions: int
            number of brain regions being analyzed
    '''

    if (feature_type == 'one_hot'):
        return [1 if i == node_idx else 0 for i in range(n_regions)]
    elif (feature_type == 'ones'):
        return [1]
    else:
        raise('Unsupposrted feature type: {}'.format(feature_type))
    
def get_edges(connectivity_matrix, threshold=0.0):
    '''Returns np arrays of edges and corresponding weights
    from connectivity matrix. Uses sparse matrix edge representation.
    
    Arg(s):
        connectivity_matrix: nxn float numpy matrix
            adjacency matrix representing connection strength between n selected brain regions
        threshold: float
            threhold representing what constitutes a valid connection between regions, default = 0.0

    '''

    edge_indices = np.where(connectivity_matrix > threshold)
    edges = np.row_stack(edge_indices)
    weights = connectivity_matrix[edge_indices]

    return edges, weights

def pack_adjaceny_matrix(df, column_labels=True, n_regions=76):
    '''Returns connectivity matrix in the form of an
    nxn numpy matrix from connectivity matrix dataframe.
    
    Arg(s):
        df: pandas dataframe
            pandas dataframe represenattion of matrix read from csv
        column_labels: bool
            true if the dataframe includes column labels (default assumes column labels present)
        n_regions: int
            number of brain regions being analyzed

    '''

    adjacency_matrix = np.zeros((n_regions, n_regions))
    row_start_idx = 1 if column_labels else 0
    for row_idx, row in df.iterrows():
        for col_idx, value in enumerate(row[row_start_idx:]):
            adjacency_matrix[row_idx][col_idx] = value
    
    return adjacency_matrix

def graph(df, column_labels=True, n_regions=76, threshold=0.0, feature_type='adj_matrix', y=None, age=None, sex=None):
    '''Returns PyTorch Geometric graph representation
    from connectivity matrix dataframe.
    
    Arg(s):
        df: pandas dataframe
            pandas dataframe represenattion of matrix read from csv
        column_labels: bool
            true if the dataframe includes column labels (default assumes column labels present)
        n_regions: int
            number of brain regions being analyzed
        threshold: float
            threhold representing what constitutes a valid connection between regions, default = 0.0
        feature_type: string
            type of features to be associated with each node / brain region

    '''

    adjacency_matrix = pack_adjaceny_matrix(df, column_labels=column_labels, n_regions=n_regions)
    x = torch.tensor(adjacency_matrix, dtype=torch.float)
    edges, weights = get_edges(adjacency_matrix, threshold=threshold)
    edge_index = torch.tensor(edges, dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float)
    py_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    if (age != None):
        py_graph.age = torch.tensor([age], dtype=torch.float)
    if (sex != None):
        py_graph.sex = torch.tensor([sex], dtype=torch.float)
    
    return py_graph

def verify_graph(graph, adjacency_matrix_df):
    adjacency_matrix = pack_adjaceny_matrix(adjacency_matrix_df)
    edges = graph.edge_index
    edge_weights = graph.edge_attr
    for i in range(edges.size(1)):
        edge = edges[:, i]
        true_edge_val = adjacency_matrix[edge[0]][edge[1]]
        if (true_edge_val <= 0):
            raise("Found edge with value less than or equal to 0 at index [{}, {}] in processed graph".format(edge[0], edge[1]))
        if (edge_weights[i] != true_edge_val):
            raise("Edge value in processed graph ({}) not equal to edge value in adjacency matrix ({})".format(edge_weights[i], true_edge_val))
    print("Verified!")

