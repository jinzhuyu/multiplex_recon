# -*- coding: utf-8 -*-
"""
calculate the katz index for each pair of nodes
"""

# https://stackoverflow.com/questions/62069781/how-to-find-the-similarity-between-pair-of-vertices-using-katz-index-in-python

import networkx as nx
import numpy as np
from numpy.linalg import inv



def _apply_prediction(G_layer_list, adj_pred_arr, pred_meth, n_layer, n_link_left):
    ''' Apply the input function to each layer of the multiplex network G .
    'G_layer_list' is a list of networkx graph for each layer
    '''
    pred_meth = ['resource_allocation_index', 'jaccard_coefficient', 'adamic_adar_index',
                 'preferential_attachment', 'common_neighbor_centrality']

    adj_pred_arr_list = []
    for i in range(n_layer):
        pred_index = exec('list(nx.{}(self.net_layer_list[i], link_unobs_left[i]))'.format(pred_meth))
        pred_index_sort = sorted(pred_index, key = lambda x: x[2], reverse=True)
        link_select = np.array([ (ele[0], ele[1]) for ele in pred_index_sort[:n_link_left[i]] ])
        # mask = (link_select[0], link_select[1])
        adj_pred_arr[i, (link_select[0], link_select[1])] = 1
        adj_pred_arr_list.append(adj_pred_arr)
    return adj_pred_arr_list
        

def kat_score(G):

    #Calculate highest eigenvector
    L = nx.normalized_laplacian_matrix(G)
    e = np.linalg.eigvals(L.A)
    print("Largest eigenvalue:", max(e))
    beta = 1/max(e)
    I = np.identity(len(G.nodes)) #create identity matrix
    
    #Katz score
    return inv(I - nx.to_numpy_array(G)*beta) - I




# https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/homogeneous-comparison-link-prediction.html

from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Node2VecLinkGenerator, Node2VecNodeGenerator
from stellargraph.layer import Node2Vec, link_classification
from tensorflow import keras


def node2vec_embedding(graph, name):
    
    # Set the embedding dimension and walk number:
    dimension = 60
    walk_number = 20

    print(f"Training Node2Vec for '{name}':")


from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
from stellargraph.layer import Attri2Vec


def attri2vec_embedding(graph, name):

    # Set the embedding dimension and walk number:
    dimension = [128]
    walk_number = 4    