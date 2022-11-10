# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx



def rescale(x,a,b,c,d):
    return c + (x-a)/(b-a)*(d-c)

def draw_G_comp(G):
    ''' plot network. Node size proportional to degree and node color by component membership
    # get connected component ids
    # ref.: https://stackoverflow.com/questions/68235334/how-to-generate-the-component-id-in-the-networkx-graph/68235670#68235670
    '''
    # node size by degre
    deg = np.array(list(dict(G.degree).values()))
    node_size = rescale(deg, min(deg), max(deg), 80, 400)
    # node color by component
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    node_comp_id = [i for i,c in enumerate(Gcc, 1) for n in c]
    # plot
    plt.figure(figsize=(6,6))
    nx.draw(G, pos=nx.nx_agraph.graphviz_layout(G, prog="neato"),
            cmap=plt.get_cmap('rainbow'), node_color=node_comp_id,
            node_size = node_size)
    plt.show()
    
    # plot
    plt.figure(figsize=(6,6))
    nx.draw(G, pos=nx.spring_layout(G),
            cmap=plt.get_cmap('viridis'), 
            node_color=node_comp_id,
            node_size = node_size)
    plt.show()
    
    # plot
    plt.figure(figsize=(6,6))
    nx.draw(G, pos=nx.nx_agraph.graphviz_layout(G, prog="fdp"),
            cmap=plt.get_cmap('viridis'), 
            node_color=node_comp_id,
            node_size = node_size)
    plt.show()


G=nx.from_edgelist()