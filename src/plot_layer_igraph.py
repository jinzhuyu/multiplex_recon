# -*- coding: utf-8 -*-
"""
plot networks with different colors for connected components

@author: Jin-Zhu Yu
"""


import matplotlib.pyplot as plt
import numpy as np
import igraph as ig # 0.9.1
import cairo   # no version info
import plotly  #5.11.0
# need to install cairo to use ig: conda install -c conda-forge pycairo
# conda install -c plotly plotly
# print(ig.__version__)
# 0.9.1


def rescale(x,a,b,c,d):
    return c + (x-a)/(b-a)*(d-c)

        ig.plot(
            components,
            palette=ig.RainbowPalette(n=len(components)),
            vertex_size=new_size,
            edge_width=2,
        )
        
        ig.plot(g, vertex_size=5, bbox=(300, 300),edge_width=1)


def find_comp_id(layer_link_list, relation_list, n_node_total, n_layer, is_save=False):
    
    for i_lyr in range(1):
        g = ig.Graph()
        link_list = layer_link_list[i_lyr]
        # n_node = max([item for sublist in link_list for item in sublist])
        g.add_vertices(list(range(n_node_total)))
        g.add_edges(link_list)
        
        deg_list = g.degree()
        node_iso = [ix for ix in range(n_node_total) if deg_list[ix]==0]
        g.delete_vertices(node_iso)
        
        deg_arr = np.array(g.degree())
        new_size = rescale(deg_arr, min(deg_arr), max(deg_arr), 5, 25)
        components = g.components(mode='strong')
        comp_id = [i for i in range(len(components))]
        link_and_comp_id = 
        
def gen_net(layer_link_list, relation_list, n_node_total, n_layer, is_save=False):

    n_col = 2
    n_row = int( (n_layer + 1 ) / n_col)
    fig, ax = plt.subplots(n_row, n_col, figsize=(11*n_col, 10*n_row), dpi=500*n_row)
    
    for i_lyr in range(1):
        g = ig.Graph()
        link_list = layer_link_list[i_lyr]
        # n_node = max([item for sublist in link_list for item in sublist])
        g.add_vertices(list(range(n_node_total)))
        g.add_edges(link_list)
        
        deg_list = g.degree()
        node_iso = [ix for ix in range(n_node_total) if deg_list[ix]==0]
        g.delete_vertices(node_iso)
        
        deg_arr = np.array(g.degree())
        new_size = rescale(deg_arr, min(deg_arr), max(deg_arr), 5, 25)
        components = g.components(mode='strong')
        
        if n_row == 1:
            if n_col == 1:
                ax_ix = ax
            else:
                ax_ix = ax[i_lyr]
        else:
            row_ix = i_lyr // n_col
            col_ix = i_lyr % n_col
            ax_ix = ax[row_ix, col_ix]        
        ig.plot(
            components,
            palette=ig.RainbowPalette(n=len(components)),
            target=ax_ix,
            vertex_size=5,
            edge_width=2,
            layout=g.layout('fr'),
            # https://igraph.readthedocs.io/en/0.10.2/tutorial.html#layout-algorithms
        )
        
    if n_row == 1:
        for ii in range(n_col):
            ax[ii].axis('off')
    else:
        for ii in range(n_row):
            for jj in range(n_col):
                ax[ii,jj].axis('off')
    if is_save:
        fname = '../output/{}_net_layer_list_{}layers_{}nodes_layers'.format(net_name, n_layer, n_node_total)
        # plt.savefig(fname + '.png', bbox_inches='tight')
        plt.savefig(fname + '.pdf', bbox_inches='tight')
    plt.show()


gen_net(layer_link_list, relation_list, n_node_total, n_layer, is_save=True)
