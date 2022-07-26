# -*- coding: utf-8 -*-
"""

"""

# import stellargraph as sg

# try:
#     sg.utils.validate_notebook_version("1.2.1")
# except AttributeError:
#     raise ValueError(
#         f"This notebook requires StellarGraph version 1.2.1, but a different version {sg.__version__} is installed.  Please see <https://github.com/stellargraph/stellargraph/issues/1172>."
#     ) from None


import networkx as nx
import pandas as pd
import numpy as np
import os
import random

import stellargraph as sg
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
from stellargraph.layer import Attri2Vec, link_classification

from tensorflow import keras

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import numba
# conda install -c numba numba

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

def SG(self, n_walks=8, length=10, batch_size=50, epoch=10, worker=4, layer_size=[12]):
    
    # TODO: G_all_nx is one layer. The largest node id is greater than the total number of nodes, i.e., the len of embeddings
    G_all_nx = nx.from_edgelist(self.layer_link_list[i_lyr])
    n_nodes = len(G_all_nx.nodes)
    n_edges = len(G_all_nx.edges)
    node_ids_list = sorted(list(G_all_nx.nodes))
    node_features = node_attr_df.reindex(node_ids_list)
    G_all = sg.StellarGraph.from_networkx(G_all_nx, node_features=node_features)
    
    G_sub_nx = nx.from_edgelist(self.layer_link_obs[i_lyr])
    node_ids_sub = sorted(list(G_sub_nx.nodes))
    node_features_sub = self.node_attr_df.reindex(node_ids_sub)
    G_sub = sg.StellarGraph.from_networkx(G_sub_nx, node_features=node_features_sub)
    
    # print('--- Layer: ', i_lyr)
    # print(G_sub.info())
    
    # --- Layer:  0
    # StellarGraph: Undirected multigraph
    #  Nodes: 1375, Edges: 1446
    
    #  Node types:
    #   default: [1375]
    #     Features: float32 vector, length 7
    #     Edge types: default-default->default
    
    #  Edge types:
    #     default-default->default: [1446]
    #         Weights: all 1 (default)
    #         Features: none
    
    # train attri2vec
    nodes = list(G_sub.nodes())
    n_walks = 2 #8
    length = 5 #10
    batch_size = 50
    epochs = 4 #10
    layer_sizes = [4] #[12]
    workers = 1
    
    unsupervised_samples = UnsupervisedSampler(G_sub, nodes=nodes, length=length,
                                               n_walks=n_walks)
    
    # Define an attri2vec training generator, which generates a batch of (feature of target node,
        # index of context node, label of node pair) pairs per iteration.
    generator = Attri2VecLinkGenerator(G_sub, batch_size)
    
    attri2vec = Attri2Vec(layer_sizes=layer_sizes, generator=generator, bias=False, normalize=None)
    
    x_inp, x_out = attri2vec.in_out_tensors()
    
    # print('---x_inp: ', x_out)
    # print('---x_out: ', x_out)
    
    # ---x_inp:  [<KerasTensor: shape=(None, 12) dtype=float32 (created by layer 'lambda')>, 
                # <KerasTensor: shape=(None, 12) dtype=float32 (created by layer 'reshape')>]
    # ---x_out:  [<KerasTensor: shape=(None, 12) dtype=float32 (created by layer 'lambda')>, 
                # <KerasTensor: shape=(None, 12) dtype=float32 (created by layer 'reshape')>]
    
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip")(x_out)
    
    
    # print('---prediction: ', prediction)
    # KerasTensor(type_spec=TensorSpec(shape=(None, 1), dtype=tf.float32, name=None),
    #             name='reshape_1/Reshape:0', description="created by layer 'reshape_1'")
    
    model = keras.Model(inputs=x_inp, outputs=prediction)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-2),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy,
                 keras.metrics.TruePositives(), keras.metrics.TrueNegatives(),
                 keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()],
        )
    
    history = model.fit(
        generator.flow(unsupervised_samples),
        epochs=epochs,
        verbose=2,
        use_multiprocessing=False,
        workers=workers,
        shuffle=True,
    )
    
    
    x_inp_src = x_inp[0]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    
    
    # node_ids_list = node_attr_df.index
    node_gen = Attri2VecNodeGenerator(G_all, batch_size).flow(node_ids_list)
    node_embeddings = embedding_model.predict(node_gen, workers=workers, verbose=1)
    
    print('\n--- Embedding done\n')
    
    
    # print('--- type of node_embeddings: ', type(node_embeddings))
    # <class 'numpy.ndarray'>
    
    # @numba.njit
    # def get_permuts_half_numba(vec: np.ndarray):
    #     k, size = 0, vec.size
    #     output = np.empty((size * (size - 1) // 2, 2))
    #     for i in range(size):
    #         for j in range(i+1, size):
    #             output[k,:] = [vec[i], vec[j]]
    #             k += 1
    #     return output
    
    in_sample_edges = np.array(layer_link_list[i_lyr][:int(n_edges*0.8)])
    out_of_sample_edges = np.array(layer_link_list[i_lyr][int(n_edges*0.8):])
    
    # https://stackoverflow.com/questions/32191029/getting-the-indices-of-several-elements-in-a-numpy-array-at-once
    node_ids_arr = np.array(node_ids_list)
    all_possib_edges_half = get_permuts_half_numba(node_ids_arr)
    
    
    sorter = np.argsort(node_ids_arr)
    
    in_sample_edge_start_indices = sorter[np.searchsorted(node_ids_arr,  in_sample_edges[:, 0], sorter=sorter)]
    in_sample_edge_end_indices = sorter[np.searchsorted(node_ids_arr,  in_sample_edges[:, 1], sorter=sorter)]
    in_sample_edge_labels = np.zeros(len(in_sample_edge_start_indices))
    in_sample_edge_labels[]
    
    out_of_sample_edge_start_indices = sorter[np.searchsorted(node_ids_arr,  out_of_sample_edges[:, 0], sorter=sorter)]
    out_of_sample_edge_end_indices = sorter[np.searchsorted(node_ids_arr,  out_of_sample_edges[:, 1], sorter=sorter)]
    in_sample_edge_labels = np.zeros(len(out_of_sample_edge_start_indices))
    
    # Construct the edge features from the learned node representations with l2 normed difference,
    # where edge features are the element-wise square of the difference between the embeddings of two head nodes. 
    # Other strategy like element-wise product can also be used to construct edge features.
    in_sample_edge_feat_from_emb = (
        node_embeddings[in_sample_edge_start_indices] - node_embeddings[in_sample_edge_end_indices]) ** 2
    
    out_of_sample_edge_feat_from_emb = (node_embeddings[out_of_sample_edge_start_indices]
        - node_embeddings[out_of_sample_edge_end_indices]) ** 2
    
    clf_edge_pred_from_emb = LogisticRegression(verbose=0, solver="lbfgs", n_jobs=4,
                                                multi_class="ovr", max_iter=50 ) #max_iter=500)
    
    clf_edge_pred_from_emb.fit(in_sample_edge_feat_from_emb, in_sample_edge_labels)
    
    edge_pred_from_emb = clf_edge_pred_from_emb.predict_proba(out_of_sample_edge_feat_from_emb)
    
    
    if clf_edge_pred_from_emb.classes_[0] == 1:
        positive_class_index = 0
    else:
        positive_class_index = 1
    
    
    roc_auc_val = roc_auc_score(out_of_sample_edge_labels, edge_pred_from_emb[:, positive_class_index])
    
    print('\n--- roc_auc: ', roc_auc_val, '\n')