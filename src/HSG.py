# -*- coding: utf-8 -*-
"""
Link prediction with Heterogeneous GraphSAGE (HinSAGE)
# https://stellargraph.readthedocs.io/en/latest/demos/link-prediction/hinsage-link-prediction.html
"""

# import stellargraph as sg

# try:
#     sg.utils.validate_notebook_version("1.2.1")
# except AttributeError:
#     raise ValueError(
#         f"This notebook requires StellarGraph version 1.2.1, but a different version {sg.__version__} is installed.  Please see <https://github.com/stellargraph/stellargraph/issues/1172>."
#     ) from None


# import networkx as nx
# import pandas as pd
# import numpy as np
# import os
# import random
import pandas as pd
import numpy as np

from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error

import stellargraph as sg
from stellargraph.mapper import HinSAGELinkGenerator
from stellargraph.layer import HinSAGE, link_regression
from tensorflow.keras import Model, optimizers, losses, metrics

import multiprocessing
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt

# import stellargraph as sg
# from stellargraph.data import UnsupervisedSampler
# from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
# from stellargraph.layer import Attri2Vec, link_classification

# from tensorflow import keras

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score

# TODO: heterogeneous
# G_all_nx = nx.Graph()
# for ii in range(len(xxx)):
    # G_all_nx.add_edges_from(layer_link_list[ii], label=relation_types[ii])
# nx.set_node_attributes(G_all_nx, node_features)
# G_all = sg.StellarGraph.from_networkx(G_all_nx)

# ====================================================

# data_dir = "../data/DBLP"

# edgelist = pd.read_csv(
#     os.path.join(data_dir, "edgeList.txt"),
#     sep="\t",
#     header=None,
#     names=["source", "target"],
# )

# edgelist["label"] = "cites"  # set the edge type

# feature_names = ["w_{}".format(ii) for ii in range(2476)]
# node_column_names = feature_names + ["subject", "year"]
# node_data = pd.read_csv(
#     os.path.join(data_dir, "content.txt"), sep="\t", header=None, names=node_column_names
# )

# G_all_nx = nx.from_pandas_edgelist(edgelist, edge_attr="label")

# nx.set_node_attributes(G_all_nx, "paper", "label")

# all_node_features = node_data[feature_names]


# # verify that we're using the correct version of StellarGraph for this notebook

# G_all = sg.StellarGraph.from_networkx(G_all_nx, node_features=all_node_features)


# print('done')
def load_data(path):
    link_df = pd.read_csv(path)
    relation_list = link_df['Relation'].unique().tolist()
    layer_link_list = []
    for idx, ele in enumerate(relation_list):
        link_temp = link_df.loc[link_df['Relation']== ele, ['From', 'To']].values.tolist()
        layer_link_list.append(link_temp)    
    return layer_link_list

net_type = 'drug'
n_node_total, n_layer = 2114, 2
# n_node_total, n_layer = 2196, 4
# n_node_total, n_layer = 2139, 3
# load each layer (a nx class object)
# with open('../data/drug_net_layer_list.pkl', 'rb') as f:
#     net_layer_list = load(f)
    
net_name = '{}_net_{}layers_{}nodes'.format(net_type, n_layer, n_node_total)
layer_link_list = load_data(path='../data/{}.csv'.format(net_name))

if net_type == 'drug':
    node_attr_df = pd.read_csv('../data/drug_net_attr_{}layers_{}nodes.csv'. \
                                  format(n_layer, n_node_total))
    node_attr_df = node_attr_df[['Node_ID', 'Gender', 'Drug_Activity', 'Recode_Level', 'Drug_Type',
                                 'Group_Membership_Type', 'Group_Membership_Code']]    
    node_attr_dict = node_attr_df.set_index('Node_ID').to_dict('index')
    node_attr_df.drop(['Node_ID'], axis=1)
else:
    node_attr_df, node_attr_dict = None, None
# layer_real_node, layer_virt_node = get_layer_node_list(layer_link_list, n_layer, n_node_total)


# for i_lyr in range(len(layer_link_list)):
i_lyr = 0
G_all_nx = nx.from_edgelist(layer_link_list[i_lyr])

# TODO: G_all_nx is one layer. The largest node id is greater than the total number of nodes, i.e., the len of embeddings
n_edges = len(G_all_nx.edges)
node_ids = sorted(list(G_all_nx.nodes))
node_features = node_attr_df.reindex(node_ids)
G_all = sg.StellarGraph.from_networkx(G_all_nx, node_features=node_features)

G_sub = nx.from_edgelist(layer_link_list[i_lyr][:int(n_edges*0.8)])
node_ids_sub = sorted(list(G_sub.nodes))
node_features_sub = node_attr_df.reindex(node_ids_sub)
G_sub = sg.StellarGraph.from_networkx(G_sub, node_features=node_features_sub)