# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations, groupby
import random
import pandas as pd
# import pickle
from networkx.generators.random_graphs import erdos_renyi_graph

import os
os.chdir('c:/code/illicit_net_resil/src')

# configuration_model(deg_sequence[, ...])

# Returns a random graph with the given degree sequence.
# M.E.J. Newman, “The structure and function of complex networks”, SIAM REVIEW 45-2, pp 167-256, 2003.
# n = 100
def gen_power_net(n, p=5e-3):
    sequence = nx.random_powerlaw_tree_sequence(n, tries=50000)
    def sub(sequence):
        G = nx.configuration_model(sequence)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G
    G = sub(sequence)
    if abs(nx.density(G) - p) / p > 0.1:
        factor = sum(sequence)/2 / ( p*(n*(n-1)/2) )
        sequence = [int(np.round(x/factor/2)*2) for x in sequence]  # make the sum a even number
        idx = 0
        while idx < n:
            # if there is an isolated node, connected it randomly to another node
            if sequence[idx] == 0:
                sequence[idx] += 1
                the_other_idx = random.choice([i for i in range(n) if i != idx])
                sequence[the_other_idx] += 1
            idx += 1
        G = sub(sequence)    
    return G

def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    print(degrees)
    plt.hist(degrees)
    plt.show()
    
# G = nx.random_degree_sequence_graph(sequence, seed=42)


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

def gen_multiplex(n_layer, n_node, p_list, net_type='power'):
    layer_link_list = []
    for i in range(n_layer):
        if net_type == 'rand':
            # used to generate connected random net. The link density is higher than input p
            # G = gen_single_rand_net(n_node, p_list[i])  
            G = erdos_renyi_graph(n_node, p_list[i])     # may be unconnected
        if net_type == 'power':
            G = gen_power_net(n_node, p_list[i])
        links = [[ele[0], ele[1], 'L{}'.format(i+1)] for ele in G.edges]
        layer_link_list.append(links)
    layer_link_list = [x for sub in layer_link_list for x in sub]    
    return layer_link_list

def get_layer(n_layer, n_node, layer_link_list):
    G_layer_list = []
    for i in range(n_layer):
        G_sub = nx.Graph()
        lyr_id = 'L{}'.format(i+1)
        selected_edges = [(ele[0], ele[1]) for ele in layer_link_list if ele[2]==lyr_id ]
        G_sub.add_edges_from(selected_edges)
        G_layer_list.append(G_sub)


def gen_overlap_net(n_layer, n_node, p_list, f_dup):
    ''' p_list: the probability of a link in each of the generated random networks
        f_dup: fraction of duplicate edges
    '''    
    if f_dup == 0:  #snp.all(np.array(f_dup) == 0):
        layer_link_list = gen_non_overlap_net(n_layer, n_node, p_list)
    elif f_dup == 1:  #np.all(np.array(f_dup) == 1):       
        layer_link_list = gen_dup_net(n_layer, n_node, p_list[0])
    else:
        layer_link_list = []
        for i in range(n_layer):
            G = erdos_renyi_graph(n_node, p_list[i])     # may be unconnected
            links = [[ele[0], ele[1], 'L{}'.format(i+1)] for ele in G.edges]
            if i >= 1:
                # randomly select a fraction of links in the current layer to be replaced
                    # the new network maynot be totally random now
                replaced_links = random.sample(links, int(len(links)*f_dup))
                links = [i for i in links if i not in replaced_links]
                # add the randomly selected links from the previous layer
                dup_links = random.sample(layer_link_list[i-1], int(len(links)*f_dup))
                # change the layer id to current layer
                dup_links = [[ele[0], ele[1], 'L{}'.format(i+1)] for ele in dup_links]
                # remove duplicated links in current layer
                dup_links = [ele for ele in dup_links if ele not in links]
                links += dup_links              
            layer_link_list.append(links)    
        # flatten the list
        layer_link_list = [x for sub in layer_link_list for x in sub]            
    return layer_link_list

def gen_dup_net(n_layer, n_node, p):
    layer_link_list = []
    G = erdos_renyi_graph(n_node, p)     # may be unconnected
    for i in range(n_layer):
        links = [[ele[0], ele[1], 'L{}'.format(i+1)] for ele in G.edges]
        layer_link_list.append(links)
    layer_link_list = [x for sub in layer_link_list for x in sub]    
    return layer_link_list

def gen_non_overlap_net(n_layer, n_node, p_list):
    layer_link_list = []
    n_node_divide = round(n_node/n_layer)
    for i in range(n_layer):
        G = erdos_renyi_graph(n_node_divide, p_list[i])     # may be unconnected
        links = [[ele[0]+i*n_node_divide, ele[1]+i*n_node_divide, 'L{}'.format(i+1)] for ele in G.edges]   
        layer_link_list.append(links)
    layer_link_list = [x for sub in layer_link_list for x in sub]    
    return layer_link_list

    # with open('../data/{}_net_layer_list_{}layers_{}nodes.pkl'.format(net_type, n_layer, n_node), 'wb') as f:
    #     pickle.dump(G_layer_list, f) 
# TODO: remove nodes and associated links in some layers to model real dark networks

def main():
    # n_node_list = [50, 100]
    n_node_list = [400]
    n_layer_list = [2, 3, 4, 5, 6]        
    # link_prob = [0.05, 0.03, 0.02, 0.02, 0.02]
    link_prob = [x*0.001 for x in [5, 4, 3.5, 3, 3, 4]]
    # net_type_list = ['non-dup'] #'dup'] #['rand'] #, 'power']
    f_dup_list = [round(i*0.2, 2) for i in range (0, 6)]  #[i/10 for i in range(11)]
    for f_dup in f_dup_list:
        for n_node in n_node_list:
            for n_layer in n_layer_list:
                layer_link_list = gen_overlap_net(n_layer, n_node, link_prob, f_dup)
                get_layer(n_layer, n_node, layer_link_list)
                link_df = pd.DataFrame(layer_link_list, columns=['From', 'To', 'Relation'])
                # link_df.to_excel("../data/rand_net_{}layers_{}nodes.xlsx".format(n_layer, n_node),
                #                   index=False) 
                link_df.to_csv("../data/rand_net_fdup_{}_{}layers_{}nodes.csv".format(f_dup, n_layer, n_node),
                               index=False) 
if __name__ == '__main__':
    main()