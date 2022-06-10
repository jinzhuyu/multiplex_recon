# -*- coding: utf-8 -*-

import os
os.chdir('c:/code/illicit_net_resil/src')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from functools import reduce
# from copy import deepcopy
# import pandas as pd
# from scipy.stats import poisson
import networkx as nx
# import pickle


# import sys
# sys.path.insert(0, './xxxx')
# from xxxx import *


class MultiNet:
    def __init__(self):

        vars = locals() # dict of local names
        self.__dict__.update(vars) # __dict__ holds an object's attributes
        del self.__dict__["self"]  # `self` is not needed anymore

    def load_data(self, path='../data/links.xlsx'):
        self.link_df = pd.read_excel(path, sheet_name='LINKS IND AND GROUP')

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
       
# multi_net = MultiNet()
# self = multi_net
# multi_net.load_data()
# multi_net.gen_net()

# https://stackoverflow.com/questions/17751552/drawing-multiplex-graphs-with-networkx




class Reconstruct:
    
    def __init__(self, M_list, PON_idx_list, n_node, itermax=1000, eps=1e-6):
        '''     
        Parameters
        ----------
        M1, M2: the adjacency link array of layer 1, 2 in a multiplex
        PON_idx1, PON_idx2 : node set containing the index of nodes in the subgraphs.
        n_node : number of nodes in each layer. nodes are shared over layers
        eps: error tolerance at convergence

        Returns
        -------
        Q1,Q2: the reconstructed adjacency matrices of layer 1, 2 in the multiplex network
        
        TODO: a more common case is that we have incomplete topology of several layers. the aggregated topology is not available.

        '''    
        # TODO: n_node can be inferred from input        
        vars = locals() # dict of local names
        self.__dict__.update(vars) # __dict__ holds an object's attributes
        del self.__dict__["self"]  # `self` is not needed anymore
        
        self.n_layer = len(self.M_list)
        self.get_layer_adj_list()
        self.get_agg_net()
    # layers should have same nodeset. Use the Union of sets of nodes in each layer.    
    
    # functions for generating the ground truth of a multiplex network  
    def get_layer_adj_list(self):
        def get_layer_adj(link_arr):
            n_node = self.n_node
            n_link = len(link_arr)
            A = np.zeros([n_node, n_node]) 
            for k in range(n_link):
                i = link_arr[k, 0] 
                j = link_arr[k, 1] 
                A[i, j] = 1 
                A[j, i] = 1        
            return A
        
        self.layer_adj_list = []
        for idx in range(self.n_layer):
            self.layer_adj_list.append(get_layer_adj(self.M_list[idx]))
            
    
    def get_agg_net(self):
        # get the aggregate network using the OR agggregation
        J_N = np.ones([self.n_node, self.n_node])
        layer_adj_neg_list = [J_N - ele for ele in self.layer_adj_list]
        self.A0 = J_N - reduce(np.multiply, layer_adj_neg_list)


    # functions used in learn layer adj        
    # avoid probability overflow in configuration model
    # prob overflow can be avoided automatically if degree gusses are integers
    def avoid_prob_overflow(self, Q_list):
        for Q in Q_list:
            Q[Q<0] = 0 
            Q[Q>1] = 1 # are there Q >1?
            if (len(Q[Q>1]) + len(Q[Q<0])) >= 1:
                print('There are prob overflows in Q')
        return Q_list

    #calculate link reliabilities by configuration model
    def cal_link_prob_deg(self, Q_list):
        ''' calculate link probability between two nodes using their degrees
        '''
        n_node = self.n_node
        for i in range(n_node):
            for j in range(n_node):
                # Page 25 in the SI
                temp = [1-self.deg_seq_list[ele][i]*self.deg_seq_list[ele][j]/(self.deg_sum_list[ele]-1)\
                        for ele in range(self.n_layer)]
                agg_link_prob = 1 - np.prod(temp)
                for idx, Q in enumerate(Q_list):
                    if agg_link_prob == 0:
                        Q[i, j] = 0
                    else:
                        # single link prob using degree of two nodes: page 27 in SI
                        single_link_prob = self.deg_seq_list[idx][i]*self.deg_seq_list[idx][j]\
                                           /(self.deg_sum_list[idx]-1)
                        Q[i, j] = self.A0[i, j]*single_link_prob/agg_link_prob                                          
        Q_list = self.avoid_prob_overflow(Q_list)
        return Q_list
    
    def cal_link_prob_PON(self, Q_list):
        # calculate link probability using partial observed nodes in each layer
        # TODO: avoid multiple nested for loops
        for ns_idx, node_set in enumerate(self.PON_idx_list):
            n_node_this = len(node_set)
            for s in range(n_node_this):
                for t in range(n_node_this):
                    i, j = node_set[s], node_set[t]
                    Q_list[ns_idx][i,j] = self.layer_adj_list[ns_idx][i, j] # TODO: why is the ground truth used here
    
                    # OR-aggregate mechanism: page 25 in SI
                    if self.A0[i, j] == 1:
                        other_layer_idx = [ele for ele in range(self.n_layer) if ele != ns_idx]
                        single_link_prob_arr = np.zeros(self.n_layer)
                        for idx1 in other_layer_idx:
                            if Q_list[idx1][i, j] not in [0, 1]: # TODO: why not =0 and not =1
                                single_link_prob = self.deg_seq_list[idx1][i]*self.deg_seq_list[idx1][j]\
                                                   /(self.deg_sum_list[idx1] - 1)
                                Q_list[idx1][i, j] = single_link_prob
                                single_link_prob_arr[idx1] = single_link_prob
                        if Q_list[ns_idx][i,j] == 0:
                            # make at least one Q_ij = 1 to make A0_ij = 1
                            # normalize each single link prob by the max
                                # so that the max automatically becomes the chosen 1
                            max_single_prob = np.max(single_link_prob_arr)
                            for idx1 in other_layer_idx:
                                if max_single_prob == 0:
                                    raise Exception('max_single_prob is 0!')
                                Q_list[idx1][i, j] = single_link_prob_arr[idx1] / max_single_prob
        Q_list = self.avoid_prob_overflow(Q_list)
        return Q_list
    
    def learn_layer_adj(self):     
        #initialize the network model parameters
        self.deg_seq_list = [np.random.randint(1, self.n_node+1, size=self.n_node)
                             for idx in range(self.n_layer)]
        self.deg_seq_last_list = [np.zeros(self.n_node) for idx in range(self.n_layer)] 
    
        for iter in range(self.itermax):
            if iter % 100 == 0: 
                print('\n====== iter: {}'.format(iter))
            # init
            n_node = self.n_node            
            self.Q_list = [np.zeros([n_node, n_node]) for idx in range(self.n_layer)]
            self.deg_sum_list = [np.sum(ele) for ele in self.deg_seq_list]
    
            #calculate link prob by configuration model
            self.Q_list = self.cal_link_prob_deg(self.Q_list)

            # update link prob using partial node sets
            self.Q_list = self.cal_link_prob_PON(self.Q_list)        
       
            #update network model parameters
            self.deg_seq_list  = [np.sum(ele, axis=0) for ele in self.Q_list]
            # print('deg_seq_list', self.deg_seq_list)

            #convergence check
            cond = [np.sum(np.abs(self.deg_seq_last_list[ele]-self.deg_seq_list[ele]))<self.eps\
                    for ele in range(self.n_layer)]            
            if all(cond):
                print('\nConverges at iter: {}'.format(iter))
                break
            else:
                if iter == self.itermax:
                    print('\nConvergence NOT achieved at the last iteration')
            
            self.deg_seq_last_list = self.deg_seq_list

    def print_result(self):
        
        n_digit = 0
        self.Q_list = [np.round(ele, n_digit) for ele in self.Q_list]
        self.deg_seq_last_list = [np.round(ele, n_digit) for ele in self.deg_seq_last_list]  
        
        n_space = 2
        n_dot = 20
        print('\nDegree sequence')
        for idx in range(self.n_layer):   
            if idx>0:
                print(' ' * n_space, '-'*n_dot)
            npprint(self.deg_seq_last_list[idx], n_space)  
            
        print('\nReconstructed adj mat')
        for idx in range(self.n_layer):   
            if idx>0:
                print(' ' * n_space, '-'*n_dot)
            npprint(self.Q_list[idx], n_space)  
        
        # true adj mat
        adj_list = [np.zeros([self.n_node, self.n_node]) for i in range(self.n_layer)]
        for idx in range(self.n_layer):
            for ele in self.M_list[idx].tolist():
                adj_list[idx][ele[0], ele[1]] = 1 
        print('\nTrue adj mat')
        for idx in range(self.n_layer): 
            if idx>0:
                print(' ' * n_space, '-'*n_dot)           
            npprint(adj_list[idx], n_space)    


def npprint(A, n_space=2):
     assert isinstance(A, np.ndarray), "input of npprint must be array like"
     if A.ndim==1 :
         print(' '*n_space, A)
     else:
         for i in range(A.shape[1]):
             npprint(A[:,i])
             
def main(): 
    
    # import data
    path = '../data/toy_net/layer_links.xlsx'
    layer_df_list = [pd.read_excel(path, sheet_name='layer_{}'.format(i)) for i in [1,2]]
    layer_link_list = [ele.to_numpy() for ele in layer_df_list]
    
    # initilize    
    n_node = max([np.amax(ele) for ele in layer_link_list]) + 1
    
    # choose observed nodes
    PON_idx_list =[[0,1,2], [0,4,5]]    

    reconst = Reconstruct(M_list=layer_link_list, PON_idx_list=PON_idx_list,
                          n_node=n_node, itermax=1000, eps=1e-6)        
    reconst.learn_layer_adj()
    
    # show results    
    reconst.print_result()
    
    
main() 

# self = reconst

