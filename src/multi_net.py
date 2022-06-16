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


class DrugNet:
    def __init__(self):

        vars = locals() # dict of local names
        self.__dict__.update(vars) # __dict__ holds an object's attributes
        del self.__dict__["self"]  # `self` is not needed anymore

        self.load_data()
        self.gen_net()
        self.get_layer_links_list()
        self.get_n_node()

    def load_data(self, path='../data/links.xlsx'):
        self.link_df = pd.read_excel(path, sheet_name='LINKS IND AND GROUP')

    def gen_net(self):
        link_df = self.link_df
        node_id = pd.concat([link_df['Actor_A'], link_df['Actor_B']], ignore_index=True).unique().tolist()
        node_id.sort()
        self.node_id = node_id
        
        G = nx.MultiGraph()
        self.layer_id_list = link_df['Type_relation'].unique() .tolist()  
        for _, row in link_df.iterrows():
            G.add_edge(row['Actor_A'], row['Actor_B'], label=row['Type_relation'])
        
        self.G = G

    
    def get_subgraph_list(self):
        '''get each layer as a subgraph 
        '''
        self.sub_graph_list = []
        for idx in range(len(self.layer_id_list)):
            G_sub = nx.Graph()
            edges_this_layer = [(u,v) for (u,v,e) in self.G.edges(data=True)\
                              if e['label']==self.layer_id_list[idx]]
            G_sub.add_edges_from(edges_this_layer)
            self.sub_graph_list.append(G_sub)
        
    def get_layer_links_list(self):
        '''
            layer_links_list: a list containing 2d link array for each layer
        '''
        
        # a list of the 2d link array for each layer
        self.layer_links_list = []
        for idx in range(len(self.layer_id_list)):
            edges_this_layer = [[u,v] for (u,v,e) in self.G.edges(data=True)\
                                 if e['label']==self.layer_id_list[idx]]
            self.layer_links_list.append(edges_this_layer)
                        
    def get_n_node(self):
        node_id = set(np.concatenate(self.layer_links_list).ravel())
        self.n_node = len(node_id) 

    def get_node_list(self):
        node_id_list = [list(set(np.concatenate(self.layer_links_list[i])))
                        for i in range(len(self.layer_id_list))]
        self.node_id_list = len(node_id_list)
        
        
        
        # select a fraction of nodes as truly observed nodes
        
        # append nodes that are present in the aggregate net, but not in the current layer
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
    def plot_layer(self, is_save_fig=True): 
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
        counter = 0
        layers_selected = ['Co-Offenders', 'Kinship', 'Formal Criminal Organization', 'Legitimate']
        ax_labels = [ele[0] for ele in list(axes.items())]
        for idx in range(len(ax_labels)):
            G_sub = nx.Graph()
            selected_edges = [(u,v) for (u,v,e) in self.G.edges(data=True) if e['label']==layers_selected[counter]]
            G_sub.add_edges_from(selected_edges)
            plt.sca(axes[ax_labels[idx]])
            nx.draw(G_sub, node_size=2)
            print(layers_selected[counter], ': ', len(selected_edges))
            counter += 1
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
               
       
# drug_net = DrugNet()
# drug_net.load_data()
# drug_net.gen_net()
# self = drug_net
# https://stackoverflow.com/questions/17751552/drawing-multiplex-graphs-with-networkx


# drug_net = Reconstruct(layer_links_list=multi_net.layer_links_list,
#                        PON_idx_list=)


class Reconstruct:
    
    def __init__(self, layer_links_list, PON_idx_list, itermax=1000, eps=1e-5, **kwargs):
        '''     
        Parameters
        ----------
        layer_links_list: a list containing 2d link array for each layer
        PON_idx1, PON_idx2 : node set containing the index of nodes in the subgraphs.
        n_node : number of nodes in each layer. nodes are shared over layers
        eps: error tolerance at convergence

        Returns
        -------
        Q1,Q2: the recovered adjacency matrices of layer 1, 2 in the multiplex network
        '''    
        super().__init__(**kwargs) # inherite parent class's method        
        vars = locals() # dict of local names
        self.__dict__.update(vars) # __dict__ holds an object's attributes
        del self.__dict__["self"]  # `self` is not needed anymore
        
        self.n_layer = len(self.layer_links_list)
        self.get_n_node()
        self.get_true_adj_list()
        self.get_agg_adj()
 
    # layers should have same nodeset. Use the Union of sets of nodes in each layer.    
    
    # functions for generating the ground truth of a multiplex network
    def get_n_node(self):
        node_id_all = set(np.concatenate(self.layer_links_list).ravel().tolist())
        self.n_node = len(node_id_all)
        
    def get_true_adj_list(self):
        def get_true_adj(link_arr):
            n_node = self.n_node
            n_link = len(link_arr)
            A = np.zeros([n_node, n_node]) 
            for k in range(n_link):
                i = link_arr[k, 0] 
                j = link_arr[k, 1] 
                A[i,j] = 1 
                A[j,i] = 1        
            return A
        
        self.true_adj_list = []        
        for idx in range(self.n_layer):
            self.true_adj_list.append(get_true_adj(self.layer_links_list[idx]))  
    
    def get_agg_adj(self):
        # get the aggregate network using the OR agggregation
        J_N = np.ones([self.n_node, self.n_node])
        true_adj_neg_list = [J_N - ele for ele in self.true_adj_list]
        self.agg_adj = J_N - reduce(np.multiply, true_adj_neg_list)


    # functions used in learn layer adj        
    # avoid probability overflow in configuration model
    # prob overflow can be avoided automatically if degree gusses are integers
    def avoid_prob_overflow(self, pred_adj_list):
        for Q in pred_adj_list:
            Q[Q<0] = 0 
            Q[Q>1] = 1 # are there Q >1?
            if (len(Q[Q>1]) + len(Q[Q<0])) >= 1:
                print('There are prob overflows in Q')
        return pred_adj_list

    #calculate link reliabilities by configuration model
    def cal_link_prob_deg(self, pred_adj_list):
        ''' calculate link probability between two nodes using their degrees
        '''
        n_node = self.n_node
        for i in range(n_node):
            for j in range(n_node):
                # Page 25 in the SI
                temp = [1-self.deg_seq_list[ele][i]*self.deg_seq_list[ele][j]/\
                        (self.deg_sum_list[ele]-1) for ele in range(self.n_layer)]
                agg_link_prob = 1 - np.prod(temp)
                for idx, Q in enumerate(pred_adj_list):
                    if agg_link_prob == 0:
                        Q[i,j] = 0
                    else:
                        # single link prob using degree of two nodes: page 27 in SI
                        single_link_prob = self.deg_seq_list[idx][i]*self.deg_seq_list[idx][j]\
                                           /(self.deg_sum_list[idx]-1)
                        Q[i,j] = self.agg_adj[i,j]*single_link_prob/agg_link_prob                                          
        pred_adj_list = self.avoid_prob_overflow(pred_adj_list)
        return pred_adj_list
    
    def cal_link_prob_PON(self, pred_adj_list):
        # calculate link probability using partial observed nodes in each layer
        # TODO: avoid multiple nested for loops
        for curr_lyr, node_set in enumerate(self.PON_idx_list):
            n_node_temp = len(node_set)
            for s in range(n_node_temp):
                for t in range(n_node_temp):
                    i, j = node_set[s], node_set[t]
                    # TODO: the following indicates that links among the observed nodes are also observed.
                        # so the subgraphs are actually vertex-induced subgraph
                    # pred_adj_list[curr_lyr][i,j] = self.true_adj_list[curr_lyr][i, j]
                    # suppose only a portion of links among observed nodes are observed
                    pred_adj_list[curr_lyr][i,j] = self.true_adj_list[curr_lyr][i,j]
    
                    # OR-aggregate mechanism: page 25 in SI
                    if self.agg_adj[i,j] == 1:
                        other_layer_idx = [ele for ele in range(self.n_layer) if ele != curr_lyr]
                        single_link_prob_arr = np.zeros(self.n_layer)
                        # calculate predicted link [i,j] probability in other layers
                        for lyr_idx in other_layer_idx:
                            single_link_prob = self.deg_seq_list[lyr_idx][i]  \
                                               *self.deg_seq_list[lyr_idx][j] \
                                               /(self.deg_sum_list[lyr_idx] - 1)
                            single_link_prob_arr[lyr_idx] = single_link_prob
                        # determine the actual predicted link [i,j] probability in other layers
                        if pred_adj_list[curr_lyr][i,j] == 1:
                            for lyr_idx in other_layer_idx:
                                if pred_adj_list[lyr_idx][i,j] not in [0, 1]: # TODO: why not =0 and not =1:
                                    pred_adj_list[lyr_idx][i,j] = single_link_prob_arr[lyr_idx]
                        if pred_adj_list[curr_lyr][i,j] == 0:
                            # make at least one Q_ij = 1 to make A0_ij = 1
                            if len(other_layer_idx) >= 2:
                                max_single_prob = np.max(single_link_prob_arr)
                                if max_single_prob != 0:
                                    # normalize each single link prob by the max
                                    # so that the max automatically becomes the chosen 1
                                    for lyr_idx in other_layer_idx:
                                        pred_adj_list[lyr_idx][i,j] = single_link_prob_arr[lyr_idx] \
                                                                      /max_single_prob
                                else: # TODO: randomly select one?
                                    rand_idx = np.random.choice(other_layer_idx)
                                    pred_adj_list[rand_idx][i,j] = 1
                            else: # two layers in total
                                rand_idx = other_layer_idx[0]
                                pred_adj_list[rand_idx][i,j] = 1
        pred_adj_list = self.avoid_prob_overflow(pred_adj_list)
        return pred_adj_list
    
    def cal_adj_MAE(self):
        ''' MAE: average percentage of incorrect links
        '''
        n_digit = 0
        self.pred_adj_list_round = [np.round(ele, n_digit) for ele in self.pred_adj_list]
        self.deg_seq_last_list_round = [np.round(ele, n_digit) for ele in self.deg_seq_last_list]  
        
        self.adj_MAE_list = [np.mean(np.abs(self.pred_adj_list_round[idx]-self.true_adj_list[idx]))\
                             for idx in range(self.n_layer)]
        self.adj_MAE = np.mean(np.array(self.adj_MAE_list))
        
    def predict_adj(self):     
        #initialize the network model parameters
        self.deg_seq_list = [np.random.uniform(1, self.n_node+1, size=self.n_node)
                             for idx in range(self.n_layer)]
        self.deg_seq_last_list = [np.zeros(self.n_node) for idx in range(self.n_layer)] 
    
        for iter in range(self.itermax):
            # if (iter+1) % 1000 == 0: 
            #     print('  === iter: {}'.format(iter+1))
            # init
            n_node = self.n_node            
            self.pred_adj_list = [np.zeros([n_node, n_node]) for idx in range(self.n_layer)]
            self.deg_sum_list = [np.sum(ele) for ele in self.deg_seq_list]
    
            #calculate link prob by configuration model
            self.pred_adj_list = self.cal_link_prob_deg(self.pred_adj_list)

            # update link prob using partial node sets and all links among observed nodes
            self.pred_adj_list = self.cal_link_prob_PON(self.pred_adj_list)        
       
            #update network model parameters
            self.deg_seq_list  = [np.sum(ele, axis=0) for ele in self.pred_adj_list]
            # print('deg_seq_list', self.deg_seq_list)

            # check convergence of degree sequence
            cond = [np.sum(np.abs(self.deg_seq_last_list[ele]-self.deg_seq_list[ele])) < self.eps\
                    for ele in range(self.n_layer)]            
            if all(cond):
                print('\nConverges at iter: {}'.format(iter))
                break
            else:
                if iter + 1 == self.itermax:
                    print('\nConvergence NOT achieved at the last iteration')
            
            self.deg_seq_last_list = self.deg_seq_list
            
        self.cal_adj_MAE()
    
    
    def print_result(self): 
        def round_list(any_list, n_digit=4):
            return [np.round(ele, n_digit) for ele in any_list]
        
        n_space = 2
        n_dot = 19
        n_digit = 4
        print('\nDegree sequence')
        for idx in range(self.n_layer):   
            if idx > 0:
                print(' ' * n_space, '-'*n_dot*2)
            npprint(round_list(self.deg_seq_last_list)[idx], n_space)  
        
        print('\n')
        for idx in range(self.n_layer):   
            if idx > 0:
                print(' ' * n_space, '-'*n_dot)
            npprint(self.deg_seq_last_list_round[idx], n_space) 
            
        print('\nRecovered adj mat')
        for idx in range(self.n_layer):   
            if idx > 0:
                print(' ' * n_space, '-'*n_dot)
            npprint(self.pred_adj_list_round[idx], n_space)  
        
        # true adj mat
        print('\nTrue adj mat')
        for idx in range(self.n_layer): 
            if idx > 0:
                print(' ' * n_space, '-'*n_dot)           
            npprint(self.true_adj_list[idx], n_space)  
        print('\nadj_MAE: ', round_list(self.adj_MAE_list))
        
        
        
    def gen_sub_graph(self, sample_meth='random_unif'):
        ''' 
        input:
            sample_meth: 'random_unif' - uniformly random or 'random_walk'          
        TODO:
            the following two lines should be modified the observed links should be among observed nodes
            'if np.random.uniform(0, 1) < self.frac_obs_link:
                 pred_adj_list[curr_lyr][i,j] = self.true_adj_list[curr_lyr][i,j]'
        '''                        
        
        
def npprint(A, n_space=2):
     assert isinstance(A, np.ndarray), "input of npprint must be ndarray"
     if A.ndim==1 :
         print(' '*n_space, A)
     else:
         for i in range(A.shape[1]):
             npprint(A[:,i])



def main_drug(): 
    
    # import data
    drug_net = DrugNet()
    drug_net.load_data()
    drug_net.gen_net()
    drug_net.get_layer_links_list()
    
    frac_obs_node = [0.1*i for i in range(8, 9)]
    n_node_obs = [int(ele*drug_net.n_node) for ele in frac_obs_node ]     
    n_fold = 1
    MAE_list = [[] for i in range(n_fold)]
     
    for i_fd in range(n_fold):   
        for idx in range(len(frac_obs_node)):        
            PON_idx_list = [np.random.choice(drug_net.n_node, n_node_obs[idx], replace=False).tolist()\
                            for i in range(len(drug_net.layer_links_list))]

            reconst = Reconstruct(layer_links_list=drug_net.layer_links_list,
                                  PON_idx_list=PON_idx_list,
                                  itermax=int(1e4), eps=1e-6)        
            reconst.predict_adj()
            MAE_list[i_fd].append(reconst.adj_MAE)
            # # show results    
            # reconst.print_result()
    
    mean_MAE = np.mean(np.array(MAE_list), axis=0)
    plt.figure(figsize=(4.8, 4.8*3/4))
    plt.plot(frac_obs_node, mean_MAE)
    plt.xlabel('Fraction of observed nodes')
    plt.ylabel('MAE of adjacency matrices')
    plt.xticks([frac_obs_node[idx] for idx in range(len(frac_obs_node)) if idx%2 == 1])
    plt.ylim(top=max(mean_MAE)+0.01)
    plt.show()
    
    print('\n mean MAE: ', mean_MAE)
    
   
    
# main_drug()


            
def main_toy(): 
    
    # import data
    path = '../data/toy_net/layer_links.xlsx'
    layer_df_list = [pd.read_excel(path, sheet_name='layer_{}'.format(i)) for i in [1,2]]
    layer_links_list = [ele.to_numpy() for ele in layer_df_list]
    
    # initilize
    node_id_list = [set(np.concatenate(ele)) for ele in layer_links_list]    
    n_node_list = [len(ele) for ele in node_id_list]
    frac_obs_node = [0.1*i for i in range(2,3)]
    n_node_obs_list = [[int(i*n_node_list[j]) for i in frac_obs_node] \
                       for j in range(len(layer_links_list))]   
    n_fold = 1
    MAE_list = [[] for i in range(n_fold)]
    
    for i_fd in range(n_fold):
   
        for idx in range(len(frac_obs_node)):
            # PON_idx_list = [[0,1,2], [0,4,5]]
            PON_idx_list = [np.random.choice(n_node_list[i], n_node_obs_list[i][idx], replace=False).tolist()\
                            for i in range(len(layer_df_list))]

            reconst = Reconstruct(layer_links_list=layer_links_list,
                                  PON_idx_list=PON_idx_list,
                                  itermax=int(1e4), eps=1e-6)        
            reconst.predict_adj()
            MAE_list[i_fd].append(reconst.adj_MAE)
            # # show results    
            reconst.print_result()
    
    mean_MAE = np.mean(np.array(MAE_list), axis=0)
    print('\n mean MAE: ', np.round(mean_MAE, 4))   
    
    plt.figure(figsize=(4.8, 4.8*3/4))
    plt.plot(frac_obs_node, mean_MAE, marker='o')
    plt.xlabel('Fraction of observed nodes')
    plt.ylabel('MAE of adjacency matrices')
    plt.xticks([frac_obs_node[idx] for idx in range(len(frac_obs_node)) if idx%2 == 1])
    plt.ylim(top=max(mean_MAE)+0.01)
    plt.show()
    

    
   
    
# main_toy() 




def sample_deg_corr(G, f=0.1, edges=None, probs=None):
    '''
    Parameters:
    ----------
    G:         network
    edges:     connections
    probs:     degree correlation related probabilities
    f:         fraction of edges to be sampled
    '''
    # prob = np.array([G.degree(u)**alpha for u in G])
    # prob = prob/prob.sum()
    if edges is None:
        edges = sorted(G.edges())

    m = G.number_of_edges()
    indices=np.random.choice(range(m),size=int(m*f),replace=False,p=probs)
    # returns a copy of the graph G with all of the edges removed
    G_observed=nx.create_empty_copy(G)
    G_observed.add_edges_from(edges[indices])
    return G_observed


def sample_random_walk(G,nodes,f=0.1):
    '''
    Parameters:
    ----------
    G:         network
    nodes:     vertices
    f:         fraction of edges to be sampled
    '''
    seed_node = np.random.choice(nodes,size=1)[0]
    G_observed=nx.create_empty_copy(G)
    m = G.number_of_edges()
    size, sample,distinct=int(m*f),[seed_node],set()

    while len(distinct) < 2*size:
        u = sample[-1]
        neighbors = list(G.neighbors(u))
        v = np.random.choice(neighbors)
        sample.append(v)
        distinct.add((u,v))
        distinct.add((v,u))
        G_observed.add_edge(u,v)
    return G_observed

           
def sample_snow_ball(G,nodes,f=0.1):
    '''
    Parameters:
    ----------
    G:         network
    nodes:     vertices
    f:         fraction of edges to be sampled
    '''
    m = G.number_of_edges()
    size = int(m*f)

    while True:
        seed_node = np.random.choice(nodes,size=1)[0]
        G_observed=nx.create_empty_copy(G)
        visited=set()

        members = [seed_node]
        sz_sampled = 0
        found = False
        while len(members)>0:
            u = members[0]
            del members[0]

            if u in visited:
                continue

            visited.add(u)

            for v in G.neighbors(u):
                if v in visited:
                    continue

                members.append(v)
                G_observed.add_edge(u,v)
                sz_sampled += 1
                if sz_sampled == size:
                    return G_observed
    return G_observed




class Utils:
    def format_fig(size_scale=1):
    
        # from matplotlib import pyplot as plt
        SMALL = 13*size_scale
        MEDIUM = 15*size_scale
        LARGE = 16*size_scale
        
        # plt.style.use('classic')
        
        plt.rcParams["font.family"] = "Arial"  #Comic Sans MS, Arial, Helvetica Neue
        plt.rcParams['font.weight']= 'normal'
        plt.rcParams['figure.figsize'] = (6, 6*3/4)
        plt.rcParams['figure.titlesize'] = LARGE   
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['figure.dpi'] = 300
    
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.axisbelow'] = True
    
        plt.rcParams['axes.titlepad'] = LARGE + 2  # title to figure
        plt.rcParams['axes.labelpad'] = 3.5 # x y labels to figure
        plt.rc('axes', titlesize=MEDIUM, labelsize=MEDIUM, linewidth=1.25)    # fontsize of the axes title, the x and y labels
        
        
        plt.rcParams['ytick.right'] = False
        plt.rcParams['xtick.top'] = False
        # plt.rcParams['xtick.minor.visible'] = True
        # plt.rcParams['ytick.minor.visible'] = True
    
        plt.rc('lines', linewidth=1.8, markersize=3) #, markeredgecolor='none')
        
        plt.rc('xtick', labelsize=SMALL)
        plt.rc('ytick', labelsize=SMALL)
        
        # plt.rcParams['xtick.major.size'] = 5
        # plt.rcParams['ytick.major.size'] = 5
    
        plt.rcParams['axes.formatter.useoffset'] = False # turn off offset
        # To turn off scientific notation, use: ax.ticklabel_format(style='plain') or
        # plt.ticklabel_format(style='plain')
        
        plt.rcParams['legend.fontsize'] = SMALL
        plt.rcParams["legend.fancybox"] = True
        plt.rcParams["legend.loc"] = "best"
        plt.rcParams["legend.framealpha"] = 0.5
        plt.rcParams["legend.numpoints"] = 1
    
        
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.dpi'] = 800
        
        
    def def_linestyles():
        linestyles = [('solid',               (0, ())),
                      ('dashdotted',          (0, (3, 5, 1, 5))),
                      ('dotted',              (0, (1, 5))),
                      ('densely dotted',      (0, (1, 1))),
                  
                      ('loosely dashed',      (0, (5, 10))),
                      ('dashed',              (0, (5, 5))),
                      ('densely dashed',      (0, (5, 1))),
                  
                      ('loosely dashdotted',  (0, (3, 10, 1, 10))),
                      ('loosely dotted',      (0, (1, 10))),
                      ('densely dashdotted',  (0, (3, 1, 1, 1))),
                  
                      ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
                      ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
                      ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
        return linestyles

Utils.format_fig()
