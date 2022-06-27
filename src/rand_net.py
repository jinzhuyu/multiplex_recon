# -*- coding: utf-8 -*-
"""

"""
import networkx as nx
from itertools import combinations, groupby
import random
import pandas as pd

# class RandomNet:
#     ''' generate a multiplex network where each layer is a random network
#         and return the list of links for each layer
#     '''

def gen_single_rand_net(n, p):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted
    source: https://stackoverflow.com/questions/61958360/how-to-create-random-graph-where-each-node-has-at-least-1-edge-using-networkx
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    assert p >= 0 and p <= 1, 'link probability p should be in [0,1]'
    if p == 0:
        return G
    if p == 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G

def gen_multiplex(n_layer, n_node, p_list):
    layer_link_list = []
    for i in range(n_layer):
        G = gen_single_rand_net(n_node, p_list[i])
        links = [[ele[0]+1, ele[1]+1, 'L{}'.format(i+1)] for ele in G.edges]
        layer_link_list.append(links)
    layer_link_list = [x for sub in layer_link_list for x in sub]    
    return layer_link_list

n_node = 100
n_layer = 2
layer_link_list = gen_multiplex(n_layer, n_node, [0.2, 0.15, 0.1])
link_df = pd.DataFrame(layer_link_list, columns=['Actor_A', 'Actor_B', 'Type_relation'])

writer = pd.ExcelWriter("../data/{}layers_{}nodes.xlsx".format(n_layer, n_node), engine='xlsxwriter')
link_df.to_excel(writer, sheet_name = 'LINKS IND AND GROUP', index=False)
writer.save() 
