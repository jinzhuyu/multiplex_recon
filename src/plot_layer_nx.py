# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import networkx as nx
# import matplotlib.transforms as mtransforms
import string

from my_utils import load_data

def rescale(x,a,b,c,d):
    return c + (x-a)/(b-a)*(d-c)

def draw_net_comp(G, n_row, n_col):
    ''' plot network. Node size proportional to degree and node color by component membership
    # get connected component ids
    # https://stackoverflow.com/questions/68235334/how-to-generate-the-component-id-in-the-networkx-graph/68235670#68235670
    '''
    # node size by degre
    deg = np.array(list(dict(G.degree).values()))
    if n_col == 3 and n_row == 1:
        size_min = 30
    if n_col == 2 and n_row == 1:
        size_min = 35
    if n_col == 2 and n_row == 2:
        size_min = 10
    size_max = size_min * 9
    node_size = rescale(deg, min(deg), max(deg), size_min, size_max)
    # node color by component size
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    cmap = cm.get_cmap('viridis_r')
    if len(Gcc) == 1: # all nodes are connected
        print('--- All nodes are connected')
        node_color = cmap.colors[-1]
    else:
        node_comp_size = [(n, len(c)) for i,c in enumerate(Gcc, 1) for n in c]
        # change to the sequence of node ids of G
        node_id = list(G.nodes())
        node_comp_size_sort = [x[1] for i in node_id  for x in node_comp_size if x[0]==i]
        node_color = node_comp_size_sort    
    # plot
    nx.draw(G, pos=nx.nx_agraph.graphviz_layout(G, prog="neato"), 
            node_color=node_color,
            node_size=node_size,
            edge_color='grey',
            cmap = cmap)

def draw_layers(layer_link_list, relation_list, net_name, n_layer, n_node_total, is_save_fig=False):
   
    # subplots with labels
    if n_layer == 2 or n_layer == 4:
        n_col = 2
        cbar_shrink = 0.91
    if n_layer == 3:
        n_col = 3
        cbar_shrink = 0.95
    n_row = n_layer // n_col +  n_layer % n_col
    if n_row == 1:
        base_fig_size = 4
        font_size = 15
        wspace=-0.01
        hspace=-0.025
    if n_row == 2:
        base_fig_size = 6
        font_size = 20 
        wspace=-0.05
        hspace=-0.025
    if n_col == 3:
        font_size = 20 
    fig, axes = plt.subplots(n_row, n_col, figsize=(base_fig_size*n_col, base_fig_size*n_row))  
    axes_flat = axes.flat      
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.subplots_adjust(left=-0.05, right=1.2, top=0.99, bottom=-0.2)
    # plot curves    
    for n, ax in enumerate(axes_flat):
        G = nx.from_edgelist(layer_link_list[n])
        plt.sca(ax)
        im = draw_net_comp(G, n_row, n_col) 
        # ax.set_title(string.ascii_uppercase[n] + '. {}'.format(relation_list[n]), 
        #              size=font_size, pad=-0.03)
        ax.text(0.083, 0.96, string.ascii_uppercase[n] + '. {}'.format(relation_list[n]),
                transform=ax.transAxes, size=font_size) 
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                 orientation='horizontal',
                 shrink=cbar_shrink, aspect=90, pad=-0.02,
                 ticks=[0, 1])
    cbar.ax.set_xticklabels(['Large', 'Small'])
    cbar.ax.tick_params(labelsize=font_size-2)
    cbar.ax.get_xaxis().labelpad = 0
    cbar.ax.set_xlabel('Size of connected component', size=font_size-2)
    #save fig
    if is_save_fig:
        file_name = '../output/layers_plot/{}_net_{}layers_{}nodes'.format(net_name, n_layer, n_node_total)
        # plt.savefig(file_name +'.png', dpi=800)
        plt.savefig(file_name +'.pdf', dpi=500)
    plt.show()

def draw_layers_ext(net_name, n_layer, n_node_total):
    file_name = '{}_net_{}layers_{}nodes'.format(net_name, n_layer, n_node_total)
    layer_link_list, relation_list = load_data('../data/{}.csv'.format(file_name))
    draw_layers(layer_link_list, relation_list, net_name, n_layer, n_node_total,
                is_save_fig=True)

def main():
    
    net_name = 'drug'
    n_node_total, n_layer = 2196, 4 
    draw_layers_ext(net_name, n_layer, n_node_total)


    net_name = 'mafia'
    n_node_total, n_layer = 143, 2
    draw_layers_ext(net_name, n_layer, n_node_total)
    
    net_name = 'london_transport'
    n_node_total = 356
    n_layer = 3
    draw_layers_ext(net_name, n_layer, n_node_total)

    net_name = 'elegan'
    n_node_total = 279
    n_layer = 3
    draw_layers_ext(net_name, n_layer, n_node_total)

# if __name__ == 'main':
#     main()