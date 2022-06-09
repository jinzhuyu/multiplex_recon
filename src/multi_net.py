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


class MultiNet:
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
        
        plt.figure(figsize=(6,4), dpi=300)
        nx.draw(self.G, node_size=2)
        plt.savefig('../output/all_layers.pdf', dpi=800)
        plt.show()
       
multi_net = MultiNet()
self = multi_net
multi_net.load_data()
multi_net.gen_net()

# https://stackoverflow.com/questions/17751552/drawing-multiplex-graphs-with-networkx




class Reconstruct:
    
    def recon_multiplex(M_list, PONset_list, n_node, itermax=1000, eps=1e-6):
        '''     
        Parameters
        ----------
        M1, M2: the adjacency matrices of layer 1, 2 in a multiplex
        PONset1, PONset2 : node set containing the index of nodes in the subgraphs
        n_node : number of nodes in each layer. nodes are shared over layers.

        Returns
        -------
        Q1,Q2: the reconstructed adjacency matrices of layer 1, 2 in the multiplex network

        '''
        
        # TODO: what if there are many layers instead of 2
            # the input M and PONset should be list
            # n_node can be inferred from input
        
        # generate the ground truth of a multiplex network
        n_node = self.n_node
        n_layer = len(self.M_list)
        
        self.A2 = np.zeros([n_node,n_node])  
    
        # layers should have same nodeset. Use the Union of sets of nodes in each layer.
        def get_layer_adj(self, M):
            n_node = self.n_node
            A = np.zeros([n_node, n_node]) 
            for k in range(n_node):
                i = M[k, 1] 
                j = M[k, 2] 
                A[i, j] = 1 
                A[j, i] = 1
            return A
        
        layer_adj_list = []
        for ix in range(n_layer):
            layer_adj_list.append(get_layer_adj(M_list[idx]))
    
        # generate the aggregate network
        J_N = np.ones([self.n_node, self.n_node])
        self.A0 = J_N - (J_N - self.A1)*(J_N - self.A2) 
        
        #initial guess of the network model parameters
        deg_seq_list = []
        deg_seq_last_list = []
        for idx in range(n_layer):
            self.deg_seq_list.append( np.random.randint(low=1, high=self.n_node+1, size=self.n_node) )
            self.deg_seq_last_list.append( np.zeros(self.n_node) ) 
    
        for iter in range(self.itermax):
    
            Q1 = np.zeros(n_node) 
            Q2 = np.zeros(n_node) 
            DegreeSum1 = np.sum(self.deg_seq_1)-1 
            DegreeSum2 = np.sum(self.deg_seq_2)-1 
    
            #calculate link reliabilities by configuration model
            def cal_link_prob(self):
                for i in range(n_node):
                    for j in range(n_node):
                        dem = 1-(1-self.deg_seq_1(i)*self.deg_seq_1(j)/DegreeSum1)*(1-self.deg_seq_2(i)*self.deg_seq_2(j)/DegreeSum2) 
                        if dem==0:
                            Q1[i, j]=0 
                            Q2[i, j]=0 
                        if dem != 0:
                            Q1[i, j]=self.A0[i, j]*self.deg_seq_1(i)*self.deg_seq_1(j)/DegreeSum1/dem 
                            Q2[i, j]=self.A0[i, j]*self.deg_seq_2(i)*self.deg_seq_2(j)/DegreeSum2/dem 
    
            #avoid probability overflow in configuration model
            def remove_error_prob(self, Q):
                Q[Q<0]=0 
                Q[Q>1]=1 
                
            self.Q1 = self.remove_error_prob(self.Q1)
            self.Q2 = self.remove_error_prob(self.Q2)
    
    
            #patial observation in layer 1
            for s in range(len(self.PONset1)):
                for t in range(len(self.PONset1)):
    
                    i = self.PONset1(s) 
                    j = self.PONset1(t) 
                    self.Q1[i, j] = self.A1[i, j] 
    
                    #OR-aggregate mechanism
                    if self.A0[i, j] == 1:
                        if Q1[i, j] == 1:
                            if Q2[i, j]!=0 and Q2[i, j]!=1:
                                Q2[i, j]=self.deg_seq_2(i)*self.deg_seq_2(j)/DegreeSum2
                        if Q1[i, j]==0:
                            Q2[i, j] = 1 
   
            #patial observation in layer 2
            for s in range(len(self.PONset1)):
                for t in range(len(self.PONset1)):
    
                    i = self.PONset2(s) 
                    j = self.PONset2(t) 
                    self.Q2[i, j] = self.A2[i, j] 
    
                    #OR-aggregate mechanism
                    if self.A0[i, j]==1:
                        if Q2[i, j]==1:
                            if Q1[i, j]!=0 and Q1[i, j]!=1:
                                Q1[i, j]=self.deg_seq_1(i)*self.deg_seq_1(j)/DegreeSum1 
                        if Q2[i, j]==0:
                            Q1[i, j] = 1 
   
            #avoid probability overflow in configuration model
            self.Q1 = self.remove_error_prob(self.Q1)
            self.Q2 = self.remove_error_prob(self.Q2)   
    
            #update network model parameters
            self.deg_seq_1 = np.sum(self.Q1) 
            self.deg_seq_2 = np.sum(self.Q2) 
    
    
            #convergence check
            if np.sum(np.abs(self.deg_seq_last_1-self.deg_seq_1))<self.eps and\
               np.sum(np.abs(self.deg_seq_last_2-self.deg_seq_2))<self.eps: 
                break
            self.deg_seq_last_1 = self.deg_seq_1 
            self.deg_seq_last_2 = self.deg_seq_2 
    
        
    
    
    
    
    
    
    
    
    
    
    
    