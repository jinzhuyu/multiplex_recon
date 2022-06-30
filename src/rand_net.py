# -*- coding: utf-8 -*-

import networkx as nx
from itertools import combinations, groupby
import random
import pandas as pd

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
        links = [[ele[0], ele[1], 'L{}'.format(i+1)] for ele in G.edges]
        layer_link_list.append(links)
    layer_link_list = [x for sub in layer_link_list for x in sub]    
    return layer_link_list

# TODO: remove nodes and associated links in some layers to model real dark networks

def main():
    n_node_list = [30, 50, 100]
    n_layer_list = [2, 3, 4]        
    link_prob = [0.15, 0.1, 0.08, 0.05, 0.05]
    for n_node in n_node_list:
        for n_layer in n_layer_list:
            layer_link_list = gen_multiplex(n_layer, n_node, link_prob[:n_layer])
            link_df = pd.DataFrame(layer_link_list, columns=['From', 'To', 'Relation'])
            link_df.to_excel("../data/rand_net_{}layers_{}nodes.xlsx".format(n_layer, n_node),
                             index=False) 

if __name__ == '__main__':
    main()