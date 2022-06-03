# -*- coding: utf-8 -*-

import os
os.chdir('c:/code/illicit_net_resil/src')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from copy import deepcopy
# import pandas as pd
# from scipy.stats import poisson
import networkx as nx
import pickle

# import sys
# sys.path.insert(0, './xxxx')
# from xxxx import *


class MultiNet():
    def __init__(self):

        vars = locals() # dict of local names
        self.__dict__.update(vars) # __dict__ holds an object's attributes
        del self.__dict__["self"]  # `self` is not needed anymore

    def load_data(self, dir='../data/links.xlsx'):
        self.link_df = pd.read_excel(dir, sheet_name='LINKS IND AND GROUP')

    def gen_net(self):
        link_df = self.link_df
        node_id = pd.concat([link_df['Actor_A'], link_df['Actor_B']], ignore_index=True).unique().tolist()
        node_id.sort()
        
        G = nx.MultiGraph()
        layer_list = link_df['Type_relation'].unique()   
        for _, row in link_df.iterrows():
            G.add_edge(row['Actor_A'], row['Actor_B'], label=row['Type_relation'])
        
        self.G = G

        
    def plot_subgraph(self, is_save_fig=True): 
        # TODO: use color to represent layers or groups, or nodes in multiple layers
        # different node size
        font_size = 13
        fig, axes = plt.subplot_mosaic([['A', 'B'], ['C', 'D']],
                                      constrained_layout=True, figsize=(4*1.9, 4*2/3*1.9), dpi=300)        
        for label, ax in axes.items():
            # label physical distance to the left and up:
            trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
            if label in ['B', 'D']:
                x_pos = 0.13
            else:
                x_pos = 0.13
            ax.text(x_pos, 0.93, label, transform=ax.transAxes + trans,
                    fontsize=font_size, va='bottom')
        fig.subplots_adjust(hspace=0.28)  
        fig.subplots_adjust(wspace=0.19)
        h = 0
        layers_selected = ['Co-Offenders', 'Kinship', 'Formal Criminal Organization', 'Legitimate']
        ax_labels = [ele[0] for ele in list(axes.items())]
        for idx in range(len(ax_labels)):
            G_sub = nx.Graph()
            selected_edges = [(u,v) for (u,v,e) in self.G.edges(data=True) if e['label']==layers_selected[h]]
            G_sub.add_edges_from(selected_edges)
            plt.sca(axes[ax_labels[idx]])
            nx.draw(G_sub, node_size=2)
            print(layers_selected[h], ': ', len(selected_edges))
            h += 1
        # save fig
        plt.tight_layout()
        if is_save_fig:
            plt.savefig('../output/each_layer_net.pdf', dpi=800)
            plt.savefig('../output/each_layer_net.png', dpi=800)
        plt.show()       
       
multi_net = MultiNet()
self = multi_net

plt.figure(figsize=(6,4), dpi=300)
nx.draw(self.G, node_size=2)
plt.savefig('../output/all_layers.pdf', dpi=800)
plt.show()

multi_net.load_data()
multi_net.gen_net()

# https://stackoverflow.com/questions/17751552/drawing-multiplex-graphs-with-networkx