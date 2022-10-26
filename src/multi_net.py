# -*- coding: utf-8 -*-

'''
#  link density cam be derived by number of nodes and average degree, so average degreee can be removed from feature list
# simplify the problem 
    # two layers or just one layer   
    # use synthetic random network to control a given feature, such as edge overlap rate
    # extreme cases
# same observation rate (e.g., 0.5), get the results
# augment drug networks data with other combinations of 2 and 3 layers

# relation between cond entropy ratio (rho_CH) and accuracy (MCC): MCC \propto 1 - beta for each network * rho_CH
'''

import os
# os.chdir('c:/code/illicit_net_resil/src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from random import sample
# import matplotlib.transforms as mtransforms
# from matplotlib import cm
# import networkx as nx
import multiprocessing as mp 
# from functools import reduce
from itertools import permutations, product
# from math import comb
from copy import deepcopy
# from pickle import load
# from prg import prg
from time import time
import numba
# conda install -c numba numba

# metrics
from sklearn.metrics import auc, confusion_matrix
from sklearn.metrics import precision_score, recall_score, precision_recall_curve 
from sklearn.metrics import matthews_corrcoef, accuracy_score
# from sklearn.metrics import f1_score #balanced_accuracy_score 
# from sklearn.metrics.cluster import fowlkes_mallows_score # geometric mean (G-mean)
from imblearn.metrics import geometric_mean_score
# conda install -c conda-forge imbalanced-learn


# other methods
# from matrix_completion import svt_solve
# conda install -c conda-forge cvxpy
# pip install matrix-completion

from my_utils import plotfuncs, npprint, copy_upper_to_lower

# from stellargraph import StellarGraph
# from stellargraph.data import UnsupervisedSampler
# from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
# from stellargraph.layer import Attri2Vec, link_classification
# from tensorflow import keras
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score



'''
# TODO: predictive accuracy measured by higher-order effects / dynamical processes on networks

# import sys
# sys.path.insert(0, './xxxx')
# from xxxx import *
# TODO: 
    # : when the adj[i,j] is observed, how to save computational cost???
    # : avoid indexing those virtual observed nodes and lnks
    # : the current complexity is N_iter*(N_all^2 + N_obs^2)
    # : includethe list of observed links as an input
    
    #: use other metric for tolorance
    #: use numpy array instead of nested list in cal_link_prob deg and PON_list

TODO: why is recall declining. tp / (tp + fn)                                                   
TODO: reduce false negative based on EM: 
    when agg=1, (1) no other layers link prob > 0.5, p_link_other = p_link_other / sum(p_link_other)
                (2) some layer link prob > 0.5,
'''        

class Reconstruct:
    def __init__(self, layer_link_list, node_attr_df=None, real_virt_node_obs=None, #net_layer_list=None, #layer_link_unobs_list=None,
                 real_node_obs=None, layer_real_node=None, n_node_total=None, itermax=100, err_tol=1e-6, 
                 simil_index_list=['jaccard_coefficient', 'preferential_attachment',
                                   'common_neighbor_centrality', 'adamic_adar_index'],
                 **kwargs):
        '''     
        Parameters
        ----------
        layer_link_list: a list containing 2d link array for each layer
        PON_idx1, PON_idx2 : node set containing the index of nodes in the subgraphs.
        n_node_total : number of nodes in each layer. nodes are shared over layers
        err_tol: error tolerance at convergence

        Returns
        -------
        Q1,Q2: the recovered adjacency matrices of layer 1, 2 in the multiplex network
        '''    
        super().__init__(**kwargs) # inherite parent class's method if necessary       
        vars = locals()
        self.__dict__.update(vars)
        del self.__dict__["self"]
        
    def main(self):
            
        self.n_layer = len(self.layer_link_list)
        self.get_layer_link_obs()
        self.get_layer_link_unobs()
        self.get_adj_true_arr()        
        
        self.get_num_link_by_layer()  
        self.get_n_link_obs()
        self.get_obs_link_mask()
        self.get_link_unobs_mask()
        
        self.lyr_pair_list = list(permutations(list(range(self.n_layer)), r=2))
                  
        self.run_all_models()
        self.get_metric_value()
    
    # functions for generating the ground truth of a multiplex network      
    # def get_subgraph_list(self):
    #     '''get each layer as a subgraph 
    #     '''
    #     self.sub_graph_list = []
    #     for idx in range(len(self.layer_id_list)):
    #         G_by_layer_sub_sg = nx.Graph()
    #         edges_this_layer = [(u,v) for (u,v,e) in self.G.edges(data=True)\
    #                           if e['label']==self.layer_id_list[idx]]
    #         G_by_layer_sub_sg.add_edges_from(edges_this_layer)
    #         self.sub_graph_list.append(G_by_layer_sub_sg)
        
    # def get_layer_link_list(self):
    #     '''layer_link_list: a list containing 2d link array for each layer
    #     '''      
    #     # a list of the 2d link array for each layer
    #     self.layer_link_list = []
        
    #     for idx in range(self.n_layer):
    #         edges_this_layer = [[u,v] for (u,v,e) in self.G.edges(data=True)\
    #                              if e['label']==self.layer_id_list[idx]]
    #         self.layer_link_list.append(edges_this_layer)
                        
    # def get_layer_node_list(self):
    #     self.node_set_agg = set(np.concatenate(self.layer_link_list).ravel())
    #     self.n_node_total = len(self.node_set_agg) 

    #     layer_node = [set(np.concatenate(self.layer_link_list[i]))
    #                       for i in range(len(self.layer_id_list))]
    #     self.layer_n_node = [len(ele) for ele in layer_node]
    #     self.layer_real_node = [list(ele) for ele in layer_node]
    #     # node in the aggregate net but not a layer
    #     self.layer_virt_node = [list(self.node_set_agg.difference(ele)) for ele in layer_node] 
              
    def get_adj_true_arr(self):
        def get_adj_true(link_arr):
            if not isinstance(link_arr, list):
                link_list = link_arr.tolist()
            else:
                link_list = link_arr
            A = np.zeros([self.n_node_total, self.n_node_total])
            link_arr = np.array(link_list)
            A[link_arr[:,0], link_arr[:,1]] = 1
            return A        
        self.adj_true_arr = np.zeros((self.n_layer, self.n_node_total, self.n_node_total))
        for i_lyr in range(self.n_layer):              
            self.adj_true_arr[i_lyr] = get_adj_true(self.layer_link_list[i_lyr])
            tri_up_idx = np.triu_indices(self.n_node_total, k=1)
            self.adj_true_arr[i_lyr][tri_up_idx[::-1]] = \
                self.adj_true_arr[i_lyr][tri_up_idx]
    
    def get_agg_adj(self):
        '''get the aggregate adj mat using the OR agggregation
        '''
        adj_true_neg = np.ones((self.n_layer, self.n_node_total, self.n_node_total)) - self.adj_true_arr
        self.agg_adj = np.ones((self.n_node_total, self.n_node_total)) - np.prod(adj_true_neg, axis=0)
        self.agg_adj_3d = np.repeat(self.agg_adj[np.newaxis, :, :], self.n_layer, axis=0) 

    def get_obs_link_mask(self):
        self.obs_link_mask_list = []
        for i_lyr in range(self.n_layer):
            obs_idx = get_permuts_half_numba(np.array(self.real_virt_node_obs[i_lyr])).astype(int).tolist()
            obs_idx = np.array(obs_idx).T
            if len(obs_idx) != 0:
                mask = (obs_idx[0], obs_idx[1])
            else:
                mask = []
            self.obs_link_mask_list.append(mask)

    def get_adj_obs(self, adj_pred_arr):
        '''get the observed adj mat from observed links
        '''
        tri_up_idx = np.triu_indices(self.n_node_total, k=1)
        for i_lyr in range(self.n_layer):
            mask = self.obs_link_mask_list[i_lyr]
            if mask:  # not empty
                adj_pred_arr[i_lyr][mask] = self.adj_true_arr[i_lyr][mask]
                # print('------ # of actual links observed in {} layer: {}'.\
                #       format(i_lyr, (adj_pred_arr[i_lyr]==1).sum()))
                # print('--------- # of possible links among observed nodes by len(mask[0]): ', len(mask[0]))
                adj_pred_arr[i_lyr][tri_up_idx[::-1]] = adj_pred_arr[i_lyr][tri_up_idx]
        return adj_pred_arr
    
    #TODO: should the initial adj_pred_arr = np.ones??? The FPs are very high

    def get_agg_adj_obs(self):
        '''get the aggregate adj mat using the OR agggregation
           according to the observed links in regular layers
        '''
            # if all are 0, then all negs are 1, then 1- prod=0
            # if one is 1, then one neg is 0, then 1-prod=1
        adj_pred_arr = self.get_adj_obs(np.zeros((self.n_layer, self.n_node_total, self.n_node_total)))        
        # np.sum(adj_pred_arr[0]!=self.adj_true_arr[0]) / self.n_node_total**2
        adj_obs_neg = np.ones((self.n_layer, self.n_node_total, self.n_node_total)) - adj_pred_arr
            
        self.agg_adj = np.ones((self.n_node_total, self.n_node_total)) - np.prod(adj_obs_neg, axis=0)
        self.agg_adj_3d = np.repeat(self.agg_adj[np.newaxis, :, :], self.n_layer, axis=0)

    def get_layer_link_obs(self):
        ''' Observed links are those who start and end nodes are both observed
            Due to symmetry, only unobserved links in the upper half are selected 
        '''
        self.layer_true_link_obs = []
        self.layer_possib_link_obs = []
        self.layer_possib_link_unobs = []
        for i_lyr in range(self.n_layer): 
            true_link_obs = [ele for ele in self.layer_link_list[i_lyr]\
                            if (ele[0] in self.real_virt_node_obs[i_lyr] and \
                                ele[1] in self.real_virt_node_obs[i_lyr])]
            self.layer_true_link_obs.append(true_link_obs)
            # len(true_link_obs)
            # foobar
            # layer_possib_links = get_permuts_half_numba(np.arange(self.n_node_total)).astype(int).tolist()
            # print('--- num layer_possib_links: ', len(layer_possib_links))
            # layer_possib_links_obs = [ele for ele in layer_possib_links\
            #                           if (ele[0] in self.real_virt_node_obs[i_lyr] and \
            #                               ele[1] in self.real_virt_node_obs[i_lyr])]
            # self.layer_possib_link_obs.append(layer_possib_links_obs)
            layer_possib_links_obs = get_permuts_half_numba(
                np.array(self.real_node_obs[i_lyr])).astype(int).tolist()
            layer_possib_links_obs = [
                [ele[0], ele[1]] if (ele[0] < ele[1]) else [ele[1], ele[0]] \
                for ele in layer_possib_links_obs]
            self.layer_possib_link_obs.append(layer_possib_links_obs)
            # len(layer_possib_links_obs)
            
        #     layer_possib_link_unobs = [ele for ele in layer_possib_links\
        #                                if ele not in layer_possib_links_obs]
        #     self.layer_possib_link_unobs.append(layer_possib_link_unobs)
            
        # self.n_possib_link_unobs = np.array([len(sub) for sub in self.layer_possib_link_unobs]) 
        # print('--- n_possib_link_unobs: ', self.n_possib_link_unobs)

    def get_layer_link_unobs(self):
        ''' Due to symmetry, only unobserved links in the upper half are selected 
        '''
        self.layer_possib_link_unobs = []
        for i_lyr in range(self.n_layer):
            # unobserved links aomng unobserved real nodes
            real_virt_node_unobs = np.array([ele for ele in range(self.n_node_total) \
                                            if ele not in self.real_virt_node_obs[i_lyr]])
            # both nodes are unobserved
            link_unobs_1 = get_permuts_half_numba(real_virt_node_unobs).astype(int).tolist()
            # ubobserved links have at least one unobserved node (if vertex-induced, use links among observed nodes orig)
            # unobserved links between real nodes observed and real nodes unobserved
            link_unobs_2 = list(product(self.real_node_obs[i_lyr], real_virt_node_unobs))
            link_unobs_2 = [[ele[0], ele[1]] if (ele[0] < ele[1]) else [ele[1], ele[0]] \
                            for ele in link_unobs_2]
            # layer_link_unobs = [ele for ele in link_posbl if \
            #                     (ele[0] in layer_node_unobs or ele[1] in layer_node_unobs)]
            self.layer_possib_link_unobs.append(np.array(link_unobs_1 + link_unobs_2))  
        self.n_possib_link_unobs = np.array([len(sub) for sub in self.layer_possib_link_unobs])
        # print('------ n_possib_link_unobs: ', self.n_possib_link_unobs) #n_possib_link_unobs:  [1148901  438703]
    
           
    def get_num_link_by_layer(self):    
        # n_node_true_by_layer = np.array([len(x) for x in self.layer_real_node])
        # n_node_obs_by_layer = np.array([len(x) for x in self.real_node_obs])
        # # n_link_obs_by_layer = self.n_link_obs
        # print('\n------ n_node_true_by_layer, n_node_obs_by_layer, n_link_obs_by_layer')
        # print('----------', n_node_true_by_layer, n_node_obs_by_layer, self.n_link_obs)
        # n_link_estimate_by_layer = (self.n_link_obs*n_node_true_by_layer/ n_node_obs_by_layer).astype(int)
        # print('---------- estimated n_link_by_layer', n_link_estimate_by_layer)
        self.n_link_total_by_layer = np.array([len(x) for x in self.layer_link_list])
        # print('------ true n_link_by_layer', self.n_link_total_by_layer)

    
    def get_n_link_obs(self):
        ''' Due to symmetry, only unobserved links in the upper half is selected 
        '''
        # self.n_link_obs = []
        # # permuts = list(permutations(range(self.n_node_total), r=2))
        # # permuts_half = [ele for ele in permuts if ele[1] > ele[0]]
        # for i_lyr in range(self.n_layer):
        #     # node_obs = self.real_virt_node_obs[i_lyr]
        #     real_virt_link_obs = get_permuts_half_numba(np.array(self.real_node_obs[i_lyr])).astype(int).tolist()
        #     real_virt_link_obs = np.array(real_virt_link_obs).T
        #     n_link_obs_temp = (self.adj_true_arr[i_lyr][(real_virt_link_obs[0], real_virt_link_obs[1])] == 1).sum()
        #     self.n_link_obs.append(n_link_obs_temp)
        #     self.n_link_left.append(self.n_link_total_by_layer[i_lyr] - n_link_obs_temp)
                    
        # smaller than actual n_link_obs
        # print('------ n_link_obs:', self.n_link_obs)       
        self.n_link_left = []
        adj_true_unobs_list = [[] for _ in range(self.n_layer)]
        for i_lyr in range(self.n_layer):
            # print('------ layer: ', i_lyr)
            for [i,j] in self.layer_possib_link_unobs[i_lyr]:
                adj_true_unobs_list[i_lyr].append(self.adj_true_arr[i_lyr][i,j])
            self.n_link_left.append((np.array(adj_true_unobs_list[i_lyr])==1).sum())
        self.adj_true_unobs = np.concatenate(adj_true_unobs_list)
        
        # print('------ true n_link_left by self.adj_true_unobs==1: ', self.n_link_left)

    def update_agg_adj(self):
        # update aggregate adj using the predicted adj of each layer by OR mechanism                                      
        self.agg_adj = 1 - np.prod(1 - self.adj_pred_arr, axis=0) 
        self.agg_adj[self.agg_adj<0] = 0 
        self.agg_adj[self.agg_adj>1] = 1
        
        # update the degree sequence of aggregate adj
        # this can be skipped
        self.agg_deg_seq = np.sum(self.agg_adj, axis=1)
        # update aggregate adj using the degree sequence
        self.agg_adj = self.agg_deg_seq[:, None]*self.agg_deg_seq[:].T / (sum(self.agg_deg_seq) - 1)          
        
        # rectify agg adj entries where associated links are observed in any layer       
        # this seems to be the major reason behind performance improvement
        for i_lyr in range(self.n_layer):
            mask = self.obs_link_mask_list[i_lyr]
            self.agg_adj[mask] = 1 - np.prod(1 - self.adj_true_arr, axis=0)[mask]     
        self.agg_adj[self.agg_adj<0] = 0 
        self.agg_adj[self.agg_adj>1] = 1
        
        # print('\n------ self.agg_adj: ', self.agg_adj)
    
    def cal_link_prob_deg(self):
        ''' calculate link probability between two nodes using their degrees (configuration model)
            used as prior link probability
            entries where the respective aggregate adj is 1 are updated
        '''        
        self.sgl_link_prob_3d = np.zeros((self.n_layer, self.n_node_total, self.n_node_total))
        for i in range(self.n_layer):
            self.sgl_link_prob_3d[i,:,:] = self.deg_seq_arr[i, :, None]*self.deg_seq_arr[i,:].T \
                                           / (self.deg_sum_arr[i] -1)                                       
        # to prevent p = 1
        epls = 1e-5
        self.sgl_link_prob_3d[self.sgl_link_prob_3d>=1] = 1 - epls
        
        agg_link_prob = 1 - np.prod(1 - self.sgl_link_prob_3d, axis=0)              
        agg_link_prob_3d = np.repeat(agg_link_prob[np.newaxis, :, :], self.n_layer, axis=0)
        
        agg_link_prob_3d[agg_link_prob_3d==0] = np.nan  # avoid "Runtime error: divided by 0"
        self.adj_pred_arr = self.agg_adj_3d * self.sgl_link_prob_3d / agg_link_prob_3d
        # self.adj_pred_arr = self.sgl_link_prob_3d
        self.adj_pred_arr = np.nan_to_num(self.adj_pred_arr) 
        
        # ratio = np.nan_to_num(self.agg_adj_3d * self.sgl_link_prob_3d / agg_link_prob_3d)
        # print('\n------\nagg_adj_3d / agg_link_prob_3d ', ratio[ratio >0] )
        
        self.adj_pred_arr[self.adj_pred_arr<0] = 0 
        self.adj_pred_arr[self.adj_pred_arr>1] = 1
    
    def cal_link_prob_PON(self):
        
        '''update link probability using the set of observed nodes in each layer
        '''
        self.adj_pred_arr = self.get_adj_obs(self.adj_pred_arr)        
        agg_link_prob_list = []
        for i_curr in range(self.n_layer):
            agg_link_prob_temp = 1 - np.prod(1 - np.delete(self.sgl_link_prob_3d, i_curr, axis=0), axis=0)
            agg_link_prob_temp[agg_link_prob_temp==0] = np.nan  # avoid Runtime error: divided by 0
            agg_link_prob_list.append(agg_link_prob_temp)
            
        for lyr_pair in self.lyr_pair_list:
            i_curr, i_othr = lyr_pair[0], lyr_pair[1]
            # if a link is present in the aggregate network
                # if link is present in current layer and link is not between observed nodes
            mask_1 = self.adj_pred_arr[i_curr, :, :] == 1 & \
                     np.logical_not(np.isin(self.adj_pred_arr[i_othr, :, :], [0, 1]))
            # self.adj_pred_arr[i_othr, mask_1] = self.agg_adj[mask_1] * self.sgl_link_prob_3d[i_othr, mask_1]
            self.adj_pred_arr[i_othr, mask_1] = self.sgl_link_prob_3d[i_othr, mask_1]
                # if link is not present in current layer and link is not between observed nodes                               
            agg_link_prob = agg_link_prob_list[i_curr]
            # agg_link_prob[i,j] = 0 means: sgl_link_prob_3d[i_othr, i, j] over other lyrs are all zeros
            mask_0 = self.adj_pred_arr[i_curr, :, :] == 0 & \
                     np.logical_not(np.isin(self.adj_pred_arr[i_othr, :, :], [0, 1])) 
            self.adj_pred_arr[i_othr, mask_0] = self.agg_adj[mask_0] * self.sgl_link_prob_3d[i_othr, mask_0]/ \
                                                agg_link_prob[mask_0]
            # self.adj_pred_arr[i_othr, mask_0] = self.sgl_link_prob_3d[i_othr, mask_0]
            self.adj_pred_arr[i_othr, mask_0] = np.nan_to_num(self.adj_pred_arr[i_othr, mask_0])
            
            # ratio = np.nan_to_num(self.agg_adj[mask_0] / agg_link_prob[mask_0])
            # print('\n------\nagg_adj_3d[mask_0] / agg_link_prob_3d[mask_0] ', ratio[ratio>0])
                                    
        self.adj_pred_arr[self.adj_pred_arr<0] = 0 
        self.adj_pred_arr[self.adj_pred_arr>1] = 1
                 
            # mask_1 = self.agg_adj_3d == 1 & self.adj_pred_arr[self.adj_pred_arr>1] = 1
    
    def predict_adj_EM(self, is_agg_topol_known=False, is_update_agg_topol=True):
        '''predict the adj mat of each layer using EM algorithm 
            if is_update_agg_topol=True, the the aggregate topology is updated at every iteration
        '''
        print('\n--- EM without aggregate adj')
        if is_agg_topol_known:
            # use true aggregate adj
            self.get_agg_adj() 
            is_update_agg_topol = False
        else:
            # without using aggregate adj
            self.get_agg_adj_obs()
        #initialize the network model parameters
        self.deg_seq_arr = np.random.uniform(1, self.n_node_total+1, size=(self.n_layer, self.n_node_total))
        self.adj_pred_arr_last = np.zeros((self.n_layer, self.n_node_total, self.n_node_total))
        self.mae_link_prob = []
        for iter in range(self.itermax):
                                               
            self.deg_sum_arr = np.sum(self.deg_seq_arr, axis=1) #[np.sum(ele) for ele in self.deg_seq_list]    
            
            #calculate link prob by configuration model
            self.cal_link_prob_deg()
            # update link prob using partial node sets and all links among observed nodes
            self.cal_link_prob_PON()  
            
            #update network model parameter: degree sequence
            self.deg_seq_arr = np.sum(self.adj_pred_arr, axis=1)
            
            # update aggregate adj
            if is_update_agg_topol:
                self.update_agg_adj()

            # check convergence of degree sequence
            mae_link_prob = np.abs(self.adj_pred_arr_last - self.adj_pred_arr)
            self.mae_link_prob.append(np.mean(mae_link_prob))
            # if np.all(mae < self.err_tol) :
            if np.all(mae_link_prob < self.err_tol):
                # print('\nConverges at iter: {}'.format(iter))
                break
            # else:
            #     if iter == self.itermax-1:
            #         print('\nNOT converged at the last iteration. MAE: {}\n'.\
            #               format(np.sum(mae_link_prob)))            
            
            self.adj_pred_arr_last = self.adj_pred_arr
            # self.adj_pred_arr_round = np.round(self.adj_pred_arr, 0)
            # self.deg_seq_last_arr_round = np.round(self.deg_seq_last_arr)   
        self.deg_seq_last_arr = self.deg_seq_arr
        # self.adjust_link_prob()
        return self.adj_pred_arr_last, self.sgl_link_prob_3d
    
    def get_link_unobs_mask(self):
        self.link_unobs_mask = []
        for i_lyr in range(self.n_layer):
            self.link_unobs_mask.append((self.layer_possib_link_unobs[i_lyr][:, 0],
                                         self.layer_possib_link_unobs[i_lyr][:, 1]))
            
    def adjust_link_prob(self):
        '''adjust link probability in each layer so that the predicted number of links
           matches the true number of unobserved links
        '''
        print('\n--- update_link_prob_link_no')
        for i_lyr in range(self.n_layer):
            print('\n------ layer: ', i_lyr)
            link_unobs_mask = self.link_unobs_mask[i_lyr]
            unobs_link_prob_temp =  self.adj_pred_arr_last[i_lyr][link_unobs_mask] 
            # print('------ len(unobs_link_prob_temp): ', len(unobs_link_prob_temp))
            top_idx = np.argsort(-unobs_link_prob_temp)[:self.n_link_left[i_lyr]] 
            mask_select = (link_unobs_mask[0][top_idx], link_unobs_mask[1][top_idx])
            top_unobs_link_prob_min = np.min(self.adj_pred_arr_last[i_lyr][mask_select])
            # print('------ min of top unobserved link prob: ', top_unobs_link_prob_min)
            if top_unobs_link_prob_min < 1:
                # left move unobs link prob so that the top min is slightly > 0.5
                self.adj_pred_arr_last[i_lyr][link_unobs_mask] -= top_unobs_link_prob_min - (0.5 + 1e-5)
            else:
                # TODO; why is the minimum prob of unobs links = 1?
                # keep the top prob as 1 and reduce the value of the rest prob to <= 0.5
                self.adj_pred_arr_last[i_lyr][link_unobs_mask] -= 0.5
                self.adj_pred_arr_last[i_lyr][mask_select] = 1             
        
        self.adj_pred_arr_last[self.adj_pred_arr_last<0] = 0 
        self.adj_pred_arr_last[self.adj_pred_arr_last>1] = 1

    
    def update_EM_add(self):
        self.adj_pred_arr_add = deepcopy(self.adj_pred_arr_last)
        #TODO: use mask            
        mask = (self.agg_adj == 1) & (np.sum(self.adj_pred_arr_add < 0.5, axis=0) == 2)
        print('------ agg=1 while no links in the associate position in all layers: ',
              np.sum(mask))
        adj_max_temp = np.max(self.adj_pred_arr_add, axis=0)
        for i_lyr in range(self.n_layer):
            # for i in range(self.n_node_total):
            #     for j in range(i+1, self.n_node_total):
            #         if self.agg_adj_3d[i_lyr, i, j] == 1 and (self.adj_pred_arr_add[:, i, j] < 0.5).all():
            #             print('------ agg=1 while no links in the associate position in each layer')
            #             self.adj_pred_arr_add[:, i, j] /= max(self.adj_pred_arr[:, i, j])
            self.adj_pred_arr_add[i_lyr][mask] /= adj_max_temp[mask]
        return self.adj_pred_arr_add

# n = 6
# bb = np.random.normal(0,1,(n,n))
# aa = np.random.normal(0,1,(2, n,n))
# mask = (bb >0.5) & (np.sum(aa < 0.5, axis=0) == 2)

# np.max(aa, axis=0)[mask]
    # using matrix factorization
    # def MF_nonbin(self):
    #     adj_pred_nm_nb = []
    #     for i_lyr in range(2):
    #         print('\n------ layer: ', i_lyr)
    #         is_obs_arr = np.zeros((self.n_node_total, self.n_node_total))
    #         # for this method, the lower triangle is considered as well
    #         link_obs_temp = np.array(self.layer_possib_link_obs[i_lyr])
    #         obs_idx_start = link_obs_temp.flatten('F')
    #         link_obs_temp[:, [0,1]] = link_obs_temp[:, [1,0]]
    #         obs_idx_end = link_obs_temp.flatten('F')
    #         # the mask indicating node obs is (is_obs_start, is_obs_end)
    #         is_obs_arr[(obs_idx_start, obs_idx_end)] = 1
    #         adj_pred_temp = svt_solve(self.adj_true_arr[i_lyr].astype(float),
    #                                   is_obs_arr, max_iterations=200)
    #         adj_pred_nm_nb.append(adj_pred_temp)
    #         print('------ count of 1 in the predicted mat: ', np.count_nonzero(adj_pred_temp == 1))
    #     return adj_pred_nm_nb

    def random_model(self):
        '''randomly select no of links left among the unobserved links
        '''
        print('\n--- Random model')
        adj_pred_arr = np.zeros((self.n_layer, self.n_node_total, self.n_node_total))
        adj_pred_arr = self.get_adj_obs(adj_pred_arr)
        for i_lyr in range(self.n_layer):
            link_unobs_mask = self.link_unobs_mask[i_lyr]
            top_idx = np.random.choice(len(link_unobs_mask[0]), self.n_link_left[i_lyr]) 
            mask_select = (link_unobs_mask[0][top_idx], link_unobs_mask[1][top_idx])
            adj_pred_arr[i_lyr][mask_select] = 1
        return adj_pred_arr

           
    def run_all_models(self):
        
        # print('\n--- Estimation based on similarity done\n')
        # adj_pred_arr_simil = self.pred_adj_simil()
        # adj_pred_arr_EM_wo_agg_adj, sgl_link_prob_3d_wo_agg_adj = self.predict_adj_EM(
        #     is_agg_topol_known=False, is_update_agg_topol=False)
        # # mae_list_no_agg_adj = self.mae_link_prob
        # for i_lyr in range(self.n_layer):
        #     print('------ # of predicted links in {} layer: {}'.\
        #           format(i_lyr, (adj_pred_arr_EM[i_lyr]>0.5).sum())) 

        # adj_pred_arr_EM_add = self.update_EM_add()
        adj_pred_arr_EM_wt_agg_adj, sgl_link_prob_3d_wt_agg_adj = self.predict_adj_EM(
            is_agg_topol_known=False, is_update_agg_topol=True)
        # mae_list_with_agg_adj = self.mae_link_prob
        # import matplotlib
        # matplotlib.use('Agg')
        # Plots.plot_link_mae(mae_list_no_agg_adj, self.mae_link_prob)

        adj_pred_arr_EM_wo_agg_adj, sgl_link_prob_3d_wo_agg_adj = \
            adj_pred_arr_EM_wt_agg_adj, sgl_link_prob_3d_wt_agg_adj
            
        # # mae_list_no_agg_adj = self.mae_link_prob
        
        # for i_lyr in range(self.n_layer):
        #     print('\n------ # of predicted links in {} layer: {}'.\
        #           format(i_lyr, (adj_pred_arr_EM_no_agg_adj[i_lyr]>0.5).sum()))         
                
        # adj_pred_arr_simil = self.pred_adj_simil()
        # print('\n--- Estimation based on neural networks\n')
        # adj_pred_arr_nn = self.rw_nn()
        adj_pred_arr_rm =  self.random_model()
        
        print('--- Done running all models\n')

        self.adj_pred_arr_list = [adj_pred_arr_EM_wt_agg_adj] + [adj_pred_arr_EM_wo_agg_adj] + [adj_pred_arr_rm]
        self.sgl_link_prob_list = [sgl_link_prob_3d_wo_agg_adj] + [sgl_link_prob_3d_wo_agg_adj] + [None]             

                
    def get_adj_true_unobs(self):
        adj_true_unobs_list = [[] for _ in range(self.n_layer)]
        for i_lyr in range(self.n_layer):
            # TODO: use mask
            # print('------ layer: ', i_lyr)
            for [i,j] in self.layer_possib_link_unobs[i_lyr]:
                adj_true_unobs_list[i_lyr].append(self.adj_true_arr[i_lyr][i,j])
        self.adj_true_unobs = np.concatenate(adj_true_unobs_list)
        
        # print('\n------ total true links left by self.adj_true_unobs==1: ', (np.array(self.adj_true_unobs)==1).sum(),'\n')


    def cal_cond_entropy(self, Z, P=None):
        ''' calculate conditional entropy: H(Z|X, Theta)
        Z is the predicted adj mat for all layers, [layer, adj_mat_row, adj_mat_col]
            Z = agg_adj*P/agg_link_prob
        P is the estimated link probability with the same dimension with Z
        e.g., cond_entropy_wt_agg_adj = self.cal_cond_entropy(Z=adj_pred_arr_EM_wt_agg_adj, P=sgl_link_prob_3d_wt_agg_adj)
        '''        
        if P is not None:
            # print('\n--- Conditional entropy of EM')
            # joint_post. P(Z, X|Theta) = prod over i <j * prod over layer
                # since the estimated Z contains X. It is reduced to P(Z|Theta)
            Z = np.round(Z)
            prod_over_layer = np.prod(np.multiply(P**Z, (1-P)**(1-Z)), axis=0)
            prod_over_layer_triu = prod_over_layer[np.triu_indices(self.n_node_total, k=1)]
            # print('------ P**Z==0 ', np.where(P**Z==0))
            # print('------ (1-P)**(1-Z)==0 ', np.where((1-P)**(1-Z)==0))
            # print('------ P<0.5, Z> 0.5 ',  np.where(np.logical_and(P<0.5, Z> 0.5)))
            # print('------ min', np.min(prod_over_layer_triu))
            # print('------ max', np.max(prod_over_layer_triu))
            # end0 = 500
            # print('------ first 200', prod_over_layer_triu[:end0])
            # from decimal import Decimal
            prod_over_layer_triu_log2 = np.log2(prod_over_layer_triu) #[np.log2(Decimal(x)) for x in prod_over_layer_triu[:end0]]
            # print('------ first 20 loggged', prod_over_layer_triu_log)
            joint_post_log2 = np.sum(prod_over_layer_triu_log2)
            # print('------ joint_post_log2: ', joint_post_log2)
            
            layer_prod_sum_log_list = []   # P(X|Theta) = prod over i <j * prod over layer 
            for i_lyr in range(self.n_layer):
                mask = self.obs_link_mask_list[i_lyr]
                if mask:  # not empty
                    layer_X_obs = Z[i_lyr][mask] #self.adj_true_arr[i_lyr][mask]
                    layer_P_obs = P[i_lyr][mask]
                    layer_prod_sum_log = np.sum(
                        np.log2(np.multiply(layer_P_obs**layer_X_obs, (1-layer_P_obs)**(1-layer_X_obs))))
                    layer_prod_sum_log_list.append(layer_prod_sum_log)
            margin_post_log2 = np.sum(layer_prod_sum_log_list)
            # print('------ margin_post_log2: ', margin_post_log2)
            if -joint_post_log2 + margin_post_log2 <= 0:
                print('------ -joint_post_log2: ', -joint_post_log2)
                print('------ margin_post_log2: ', margin_post_log2)
            # if margin_post != 0:    
            cond_entropy_log2 = joint_post_log2 + np.log2(-joint_post_log2 + margin_post_log2)
            IG_ratio = margin_post_log2 / joint_post_log2
            return cond_entropy_log2, IG_ratio
            # print('------ cond_entropy_log2: ', cond_entropy_log2)
            # else:
            #     raise(ValueError('marginal post is zero'))
        else:
            # print('\n--- Conditional entropy of random model')
            # margin_post_log2 = 0   # X is given, thus P(X|null model) = 1
            
            # joint_post_log2_list = []
            # for i_lyr in range(self.n_layer):
            #     link_unobs_mask = self.link_unobs_mask[i_lyr]
            #     joint_post_log2_list.append(np.log2(comb(len(link_unobs_mask[0]), self.n_link_left[i_lyr])) -\
            #                                 len(link_unobs_mask[0]))
            # joint_post_log2 = np.sum(joint_post_log2_list)
            # print('------ joint_post_log2: ', joint_post_log2) 
            return 999, 999

        
    def get_topo_charac(self, adj_pred_arr):
        # get complete estimated adj mat
            # copy upper triangle to lower triangle
        adj_pred_arr_sym_list = [copy_upper_to_lower(X) for X in adj_pred_arr]
       
        # average degree
        ave_deg_list = [np.mean(np.sum(X, axis=0)) for X in adj_pred_arr_sym_list]
        #ave_deg_list = [x/max(ave_deg_list) for x in ave_deg_list]
       
        # link density
        total_possib_link = self.n_node_total * (self.n_node_total - 1) / 2
        link_density_list = [np.count_nonzero(X==1)/total_possib_link for X in adj_pred_arr_sym_list]
        
        print('--- link_density: ', link_density_list)
        
        # edge overlap compared to the layer with most links
        idx_temp = link_density_list.index(max(link_density_list))
        adj_temp = adj_pred_arr_sym_list[idx_temp]
        # n_link_temp = np.count_nonzero(adj_temp==1)


        edge_overlap_ratio_list = [np.count_nonzero(X == adj_temp)/self.n_node_total**2 for X in adj_pred_arr_sym_list]

            

        return ave_deg_list, link_density_list, edge_overlap_ratio_list


    def get_metric_value_sub(self, adj_pred_arr, sgl_link_prob):
        ''' performance evaluation using multiple metrics for imbalanced data, e.g., geometric mean, MCC
        '''
        adj_pred_unobs_list = [[] for _ in range(self.n_layer)]
        for i_lyr in range(self.n_layer):
            # TODO: use mask
            for [i,j] in self.layer_possib_link_unobs[i_lyr]:
                adj_pred_unobs_list[i_lyr].append(adj_pred_arr[i_lyr][i,j])
        adj_pred_unobs = np.concatenate(adj_pred_unobs_list)
        
        precision, recall, _ = precision_recall_curve(self.adj_true_unobs, adj_pred_unobs)
        auc_pr = auc(recall, precision)
        
        if not ((adj_pred_unobs == 0) | (adj_pred_unobs == 2)).all():
            # round the probabilistic predictions by EM
            adj_pred_unobs = np.round(adj_pred_unobs)
        gmean  = geometric_mean_score(self.adj_true_unobs, adj_pred_unobs)      
        mcc = matthews_corrcoef(self.adj_true_unobs, adj_pred_unobs)
        # f1 = f1_score(self.adj_true_unobs, adj_pred_unobs)
        recall_val = recall_score(self.adj_true_unobs, adj_pred_unobs)
        precision_val = precision_score(self.adj_true_unobs, adj_pred_unobs)
        accuracy_val = accuracy_score(self.adj_true_unobs, adj_pred_unobs)
        # b_acc = balanced_accuracy_score(self.adj_true_unobs, adj_pred_unobs)  #balanced_accuracy_score
        # metric_value = [recall, precision, auc_pr, gmean, mcc] #, f1] 
        tn, fp, fn, tp = confusion_matrix(self.adj_true_unobs, adj_pred_unobs, labels=[0, 1]).ravel()
        
        # print('\n------ auc_pr, gmean, mcc, recall_val, precision_val: ', auc_pr, gmean, mcc, recall_val, precision_val)
        
        # conditional entropy log2
        cond_entropy_log2, IG_ratio = self.cal_cond_entropy(np.round(adj_pred_arr), sgl_link_prob)

        # topological characteristics
        ave_deg_list, link_density_list, edge_overlap_ratio_list = self.get_topo_charac(adj_pred_arr)
        
        return [recall, precision, auc_pr, gmean, mcc, recall_val, precision_val, accuracy_val,
                tn, fp, fn, tp, cond_entropy_log2, IG_ratio, ave_deg_list, link_density_list, edge_overlap_ratio_list]

    
    def get_metric_value(self):
        self.get_adj_true_unobs()
        self.metric_value_list = []
        for ix, adj_pred in enumerate(self.adj_pred_arr_list):
            self.metric_value_list.append(self.get_metric_value_sub(adj_pred, self.sgl_link_prob_list[ix])) 

        
    def print_result(self): 
        def round_list(any_list, n_digit=4):
            return [np.round(ele, n_digit) for ele in any_list]
        
        n_space = 2
        n_dot = 19
        # n_digit = 4
        print('\nDegree sequence')
        for idx in range(self.n_layer):   
            if idx > 0:
                print(' ' * n_space, '-'*n_dot*2)
            npprint(round_list(self.deg_seq_last_arr)[idx], n_space)  
        
        print('\n')
        for idx in range(self.n_layer):   
            if idx > 0:
                print(' ' * n_space, '-'*n_dot)
            npprint(self.deg_seq_last_arr_round[idx], n_space) 
            
        print('\nRecovered adj mat')
        for idx in range(self.n_layer):   
            if idx > 0:
                print(' ' * n_space, '-'*n_dot)
            npprint(self.adj_pred_arr_round[idx,:,:], n_space)  
       
        
    # def gen_sub_graph(self, sample_meth='random_unif'):
    #     ''' 
    #     input:
    #         sample_meth: 'random_unif' - uniformly random or 'random_walk'          
    #     TODO:
    #         the following two lines should be modified the observed links should be among observed nodes
    #         'if np.random.uniform(0, 1) < self.frac_obs_link:
    #              self.adj_pred_arr[i_curr][i,j] = self.adj_true_arr[i_curr][i,j]'
    #     '''
    #     pass                        

class Plots:
    # class variables
    colors = ['tab:{}'.format(x) for x in ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown','cyan','olive']]
    markers = ['o', 'v', '>', '*', 'x', '<','d', 'o', 'v', '>', '*', 's', 'd','x']
    lw = .9
    med_size = 7
    linestyles = plotfuncs.get_linestyles()          
    
    def plot_link_mae(mae_list_no_agg_adj, mae_list_with_agg_adj):
        plt.figure(figsize=(5, 4), dpi=400)
        plt.plot(range(len(mae_list_no_agg_adj)), mae_list_no_agg_adj, label='Without agg. adj')
        plt.plot(range(len(mae_list_with_agg_adj)), mae_list_with_agg_adj, label='With agg. adj')
        plt.xlabel("Iteration")
        plt.ylabel("Mean MAE")
        plt.legend(loc='best')
        plt.savefig('../output/{}_link_prob_mae_{}layers_{}nodes.pdf'.format(net_name, n_layer, n_node_total))
        plt.show() 

    def get_mean_prc(x_mean, x_list, y_list):
        '''get the mean of precision (y) values at mean of recall (x)
           when # of x and y points are unequal over different repetitions
        return: a list of y mean values associated with x mean values
        '''
        y_intp_list = []
        for i in range(len(x_list)):
            y_intp_list.append(np.interp(x_mean, x_list[i][::-1], y_list[i])[::-1])       
        # y_mean = np.mean(np.array(y_intp_list), axis=0).tolist()       
        return np.mean(np.array(y_intp_list), axis=0).tolist()
    
    def plot_each_metric_sub(frac_list, metric_mean_by_model, metric, model_list, n_layer, n_node_total):
        
        # for mtc in 
        # metric_value_by_frac = metric_mean_by_frac[2:]
        # first_score_metric = 2
        # metric_mean_by_frac_select = metric_mean_by_frac[first_score_metric:]
        # metric_select = ['Recall', 'Precision', 'AUC-PR', 'G-mean']  #, 'MCC'][first_score_metric:]
        # print('------ in plot_each_metric_sub {}: {}'.format(metric, metric_mean_by_model))
        plotfuncs.format_fig(1.1)
        plt.figure(figsize=(5, 4), dpi=500)
        if metric == 'LogH':
            model_list = model_list[:-1]
            # metric = r'$\log_2(\mathcal{H})$'
        for i in range(len(model_list)):
            plt.plot(frac_list, metric_mean_by_model[i], color=Plots.colors[i],
                     marker=Plots.markers[i], alpha=.85,ms=Plots.med_size,
                     lw=Plots.lw,linestyle = '--', label=model_list[i])                    
        plt.xlim([min(frac_list)-0.03, 1.03])        
        # plt.xlim(right=1.03)
        # plt.ylim([0, 1.03])
        # plt.xlim([-0.03, 1.03])
        # plt.ylim([0.0, 1.03])
        # plt.xlabel(r"$c$")
        plt.xlabel('Fraction of observed components')
        plt.ylabel(metric)
        plt.legend(loc="best", fontsize=11)
        # plt.xticks([0.2*i for i in range(5+1)])
        plt.savefig('../output/{}_{}layers_{}nodes_{}.pdf'.format(net_name, n_layer, n_node_total, metric))
        plt.show() 
        
    def plot_each_metric(frac_list, metric_mean_by_frac, n_layer, n_node_total, metric_list, model_list):        
        metric_to_plot = ['Recall_range', 'Precision_range', 'AUC-PR', 'G-mean',
                          'MCC', 'Recall', 'Precision', 'Accuracy',
                          'TN', 'FP', 'FN', 'TP', 'Log_H', 'IG_ratio']
        for i_mtc, mtc in enumerate(metric_list):
            if mtc in metric_to_plot:
                # print('\n------ metric_mean_by_model_frac[i_mtc]', np.array(metric_mean_by_model_frac[i_mtc-2]))
                print('\n')
                # if i_mtc <= last_metric_to_plot:
                Plots.plot_each_metric_sub(frac_list, metric_mean_by_frac[i_mtc],
                                           metric_list[i_mtc], model_list,
                                           n_layer, n_node_total) 
                # if metric_list[i_mtc] in ['MCC', 'Recall', 'Precision']:
                print('------ {}: {}'.format(metric_list[i_mtc], metric_mean_by_frac[i_mtc]))
                
                print('\n')
        
    def plot_prc(frac_list, metric_mean_by_frac, n_layer, n_node_total, model_list):
        ''' precision-recall curve for each fraction of observed nodes
        '''
        recall_list, precision_list = metric_mean_by_frac[0], metric_mean_by_frac[1]
        plotfuncs.format_fig(1.2)       
        n_frac = len(frac_list)
        for i_frac in range(n_frac):
            plt.figure(figsize=(5.5, 5.5*4/5), dpi=500)
            for i_mdl in range(len(model_list)):
                plt.plot(recall_list[i_mdl][i_frac], precision_list[i_mdl][i_frac],
                         color=Plots.colors[i_mdl], #marker='', #markers[i_idx], 
                         # ms=med_size, 
                         lw=Plots.lw, linestyle = Plots.linestyles[i_mdl][1],
                         alpha=.85,
                         # label="{:.2f} ({:0.2f})".format(frac_list[idx], auc_list[idx]))
                         label=model_list[i_mdl])                
            # plt.plot([0, 1], [0, 1], "k--", lw=lw)
            # baseline is random classifier. P/ (P+N)
            # https://stats.stackexchange.com/questions/251175/what-is-baseline-in-precision-recall-curve
            # plt.xlim([-0.015, 1.015])
            plt.xlim([min(frac_list)-0.03, 1.03])
            # plt.ylim([-0.015, 1.015])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            # plt.xticks(np.linspace(0, 1, num=6, endpoint=True))
            plt.legend(loc="lower right", fontsize=13) #, title=r'$c$') #title=r'$c$  (AUC)')
            # ax = plt.gca()
            # handles, labels = ax.get_legend_handles_labels()
            # ax.legend(reversed(handles), reversed(labels), title=r'$c$', loc='lower left', fontsize=14.5,)        
            plt.savefig('../output/{}_prc_frac{}_{}layers_{}nodes.pdf'.format(
                net_name, frac_list[i_frac],n_layer, n_node_total))
            plt.show()


    def plot_other(frac_list, metric_mean_by_frac, n_layer, n_node_total):
        # metric_value_by_frac = metric_mean_by_frac[2:]
        first_score_metric = 2
        metric_mean_by_frac_select = metric_mean_by_frac[first_score_metric:]
        metric_select = ['Recall_range', 'Precision_range',
                         'AUC-PR', 'G-mean', 'MCC', 'Recall', 'Precision', 'Accuracy']
        metric_select= metric_select[first_score_metric:]
        plotfuncs.format_fig(1.1)
        plt.figure(figsize=(5, 4), dpi=400)
        for i in range(len(metric_select)):
            plt.plot(frac_list, metric_mean_by_frac_select[i], color=Plots.colors[i],
                     marker=Plots.markers[i], alpha=.85, ms=Plots.med_size,
                     lw=Plots.lw, linestyle = '--', label=metric_select[i])
        # plt.xlim(right=1.03)
        # plt.ylim([0, 1.03])
        plt.xlim([min(frac_list)-0.03, 1.03])
        # plt.ylim([0.0, 1.03])
        plt.xlabel(r"$c$")
        plt.ylabel("Value of metric")
        plt.legend(loc="lower right", fontsize=13)
        # plt.xticks([0.2*i for i in range(5+1)])
        plt.savefig('../output/{}_imbl_metrics_{}layers_{}nodes.pdf'.format(net_name, n_layer, n_node_total))
        plt.show()                  

    
def load_data(path):
    link_df = pd.read_csv(path)
    relation_list = link_df['Relation'].unique().tolist()
    layer_link_list = []
    for idx, ele in enumerate(relation_list):
        link_temp = link_df.loc[link_df['Relation']== ele, ['From', 'To']].values.tolist()
        layer_link_list.append(link_temp)    
    return layer_link_list


def get_layer_node_list(layer_link_list, n_layer, n_node_total):
    node_set_agg = set(np.concatenate(layer_link_list).ravel())
    layer_node = [set(np.concatenate(layer_link_list[i])) for i in range(n_layer)]
    # layer_n_node = [len(ele) for ele in layer_node]
    layer_real_node = [list(ele) for ele in layer_node]
    # node in the aggregate net but not a layer
    layer_virt_node = [list(node_set_agg.difference(ele)) for ele in layer_node]
    return layer_real_node, layer_virt_node

# def cal_edge_overlap_rate

@numba.njit
def get_permuts_half_numba(vec: np.ndarray):
    k, size = 0, vec.size
    output = np.empty((size * (size - 1) // 2, 2))
    for i in range(size):
        for j in range(i+1, size):
            output[k,:] = [vec[i], vec[j]]
            k += 1
    return output
             
# there are always unobserved links when the network is large enough    
# def sample_node_obs(layer_link_list, layer_real_node, layer_virt_node, i_frac):    
#     real_node_obs = [np.random.choice(layer_real_node[i_lyr], n_real_node_obs[i_frac][i_lyr],
#                          replace=False).tolist() for i_lyr in range(n_layer)]                
#     # append virtual nodes: all nodes - nodes in each layer
#     real_virt_node_obs = [real_node_obs[i_lyr] + layer_virt_node[i_lyr] \
#                     for i_lyr in range(n_layer)]
    
#     # avoid trivial cases where all links are observed
#     is_empty = []
#     reconst_temp = Reconstruct(layer_link_list=layer_link_list,
#                                real_virt_node_obs=real_virt_node_obs, n_node_total=n_node_total)
#     layer_link_unobs_list = reconst_temp.layer_link_unobs_list
#     for i_lyr in range(reconst_temp.n_layer):
#         if reconst_temp.layer_link_unobs_list[i_lyr].size == 0:
#             is_empty.append(i_lyr)
#     if len(is_empty) == reconst_temp.n_layer:
#         print('--- No layers have unobserved links. Will resample observed nodes.')
#         return sample_node_obs(layer_link_list, layer_real_node, layer_virt_node, i_frac)
#     else:
#         return real_virt_node_obs, layer_link_unobs_list

# i_frac = 0
# for 2 layer 6 node toy net, real_virt_node_obs = [[0,1,2], [0,4,5]] leads to zero error
def single_run(i_frac):  #, layer_link_list, n_node_total):
    metric_value_rep_list = []
    for i_rep in range(n_rep):
        # real_virt_node_obs, layer_link_unobs_list = sample_node_obs(layer_link_list, layer_real_node,
        #                                                       layer_virt_node, i_frac)
        # t000 = time()
        # foo
        real_node_obs = [np.random.choice(layer_real_node[i_lyr], n_real_node_obs[i_frac][i_lyr],
                         replace=False).tolist() for i_lyr in range(n_layer)]
        [ele.sort() for ele in real_node_obs]            
        # append virtual nodes: all nodes - nodes in each layer
        real_virt_node_obs = [real_node_obs[i_lyr] + layer_virt_node[i_lyr] \
                              for i_lyr in range(n_layer)]
        [ele.sort() for ele in real_virt_node_obs] 
        reconst = Reconstruct(layer_link_list=layer_link_list, node_attr_df=node_attr_df,
                              real_virt_node_obs=real_virt_node_obs, real_node_obs=real_node_obs,
                              layer_real_node=layer_real_node,
                              # net_layer_list=net_layer_list,
                              # layer_link_unobs_list=layer_link_unobs_list,
                              n_node_total=n_node_total, itermax=itermax, err_tol=1e-2)    
        reconst.main() 
        # t100 = time()
        # print('=== {} mins on this rep in total'.format( round( (t100-t000)/60, 3) ) ) 
        metric_value_rep_list.append(reconst.metric_value_list)
        # if i_rep == n_rep - 1:
            # print('--- rep: {}'.format(i_rep))
    return metric_value_rep_list
# self = reconst
# reconst.print_result()

def paral_run():
    n_cpu = mp.cpu_count()
    if n_cpu <= 8:
        n_cpu = 5
    else:
        n_cpu = int(n_cpu*0.6)
    
    # print('=== # of CPUs used: ', n_cpu)
    with mp.Pool(n_cpu) as pool:
        results = pool.map(single_run, range(n_frac))
    return results

# results include metric_value_rep_list for each frac. 
# metric_value_rep_list include metric_value

# results[i_frac][i_rep][i_mdl][i_mtc]

def save_output(net_name, n_node_total, n_layer, frac_list, metric_mean_by_frac):
    '''save log2H at each observatation rate to a pandas pf
    '''
    log2H_loc = metric_list.index('Log_H')
    IG_ratio_loc = metric_list.index('IG_ratio')
    accuracy_metric_list = ['AUC-PR', 'G-mean', 'MCC']
    topo_charc_list = ['ave_deg', 'link_density', 'edge_overlap_ratio']
    # accuracy_loc = metric_list.index('Accuracy')
    # mcc_loc = metric_list.index('MCC')
    # auc_loc = metric_list.index('AUC_PR')
    
    EM_with_agg_adj_loc = 0
    log2H = metric_mean_by_frac[log2H_loc][EM_with_agg_adj_loc][:]
    IG_ratio = metric_mean_by_frac[IG_ratio_loc][EM_with_agg_adj_loc][:]
    # accuracy = metric_mean_by_frac[accuracy_loc][EM_with_agg_adj_loc][:]
    
    output_df = pd.DataFrame(columns=['net_name', 'n_node_total', 'n_layer',
                                      'obs_frac', 'log2H'] \
                                      + accuracy_metric_list + topo_charc_list)
    output_df['obs_frac'] = frac_list
    output_df['log2H'] = log2H 
    output_df['IG_ratio'] = IG_ratio
    
    for metric in accuracy_metric_list + topo_charc_list:
        output_df[metric] = metric_mean_by_frac[metric_list.index(metric)][EM_with_agg_adj_loc][:]
    output_df['net_name'] = net_name
    output_df['n_node_total'] = n_node_total
    output_df['n_layer'] = n_layer    
    
    path = '../output/log2H_by_obs_frac.csv'    
    file_exists = os.path.exists(path) 
    if file_exists:
        output_df.to_csv(path, mode='a', header=False, index=False)        
    else:                
        output_df.to_csv(path, header=True, index=False)      


def run_plot(frac_list, metric_mean_by_frac, n_layer, n_node_total, metric_list, model_list):
    #Plots
    Plots.plot_each_metric(frac_list, metric_mean_by_frac, n_layer, n_node_total, metric_list, model_list)
    Plots.plot_prc(frac_list, metric_mean_by_frac, n_layer, n_node_total, model_list)


def get_metric_mean(results):
    metric_value_by_frac = [ [ [[] for _ in frac_list] for _ in model_list] for _ in metric_list]
    for i_mtc in range(n_metric):
        for i_frac in range(n_frac):
            for i_rep in range(n_rep):
                for i_mdl in range(n_model):
                    metric_value_by_frac[i_mtc][i_mdl][i_frac].append(results[i_frac][i_rep][i_mdl][i_mtc])
                    # if i_mtc >=2 and i_rep==1:
                    #     print('---', i_mtc, i_mdl, i_frac, metric_value_by_frac[i_mtc][i_mdl][i_frac])
    # calculate the mean
    metric_mean_by_frac = [ [[ [] for _ in frac_list] for _ in model_list] for _ in metric_list]   
    recall_mean = np.linspace(0, 1, 40)
    for i_mtc, mtc in enumerate(metric_list): # enumerate(metric_list):
        for i_mdl in range(n_model):
            if mtc == 'Recall_range':
                for i_frac in range(n_frac):
                    metric_mean_by_frac[i_mtc][i_mdl][i_frac] = recall_mean
            elif mtc == 'Precision_range':
                for i_frac in range(n_frac):
                    recall_list = metric_value_by_frac[i_mtc-1][i_mdl][i_frac]
                    prec_list = metric_value_by_frac[i_mtc][i_mdl][i_frac]
                    # print('\n--- recall_list', recall_list[0])
                    # print('\n--- prec_list', prec_list[0])
                    prec_mean = Plots.get_mean_prc(recall_mean, recall_list, prec_list)
                    # print('\n--- prec_mean', prec_mean)
                    metric_mean_by_frac[i_mtc][i_mdl][i_frac] = prec_mean
            elif mtc in ['AUC-PR', 'G-mean', 'MCC', 'Recall', 'Precision', 'Accuracy',
               'TN', 'FP', 'FN', 'TP', 'Log_H', 'IG_ratio']: 
                print(mtc)
                for i_frac in range(n_frac):
                    # print(np.array(metric_value_by_frac[i_mtc][i_mdl][i_frac]))
                    metric_mean_by_frac[i_mtc][i_mdl][i_frac] = np.nanmean(
                        np.array(metric_value_by_frac[i_mtc][i_mdl][i_frac]))
                    # print('---',i_mtc, i_mdl, i_frac, metric_mean_by_frac[i_mtc][i_mdl][i_frac])
            else: # mtc in ['ave_deg', 'link_density', 'edge_overlap_ratio']: 
                print(mtc)
                print(metric_value_by_frac[i_mtc][i_mdl][i_frac])
                for i_frac in range(n_frac):  
                    metric_mean_by_frac[i_mtc][i_mdl][i_frac] = np.mean(
                        np.array(metric_value_by_frac[i_mtc][i_mdl][i_frac]), axis=0)
    return metric_mean_by_frac      


def run_main():
    #results = paral_run()
    # for ele in metric_list:
    #     exec('{}_list = [[] for item in range(len(frac_list))]'.format(ele))
    # get mean
    metric_mean_by_frac = get_metric_mean(paral_run())
    # metric_value_by_frac = [auc_list, prec_list, recall_list, acc_list]
    # print('\nmetric_value_by_frac: ', metric_value_by_frac)
    #Plots
    run_plot(frac_list, metric_mean_by_frac, n_layer, n_node_total, metric_list, model_list)
    # save log2H
    save_output(net_name, n_node_total, n_layer, frac_list, metric_mean_by_frac)

# import data
# net_name = 'toy'
# n_node_total, n_layer = 6, 2


# net_name = 'power'
# n_node_total, n_layer = 500, 2
# n_node_total, n_layer = 500, 3
# n_node_total, n_layer = 500, 4
# n_node_total, n_layer = 500, 5
# n_node_total, n_layer = 500, 6


net_name = 'dup'
n_node_total = 400
n_layer = 2
# n_layer = 3
# n_layer = 4
# n_layer = 5
# n_layer = 6

# dup_frac = 0.0
# dup_frac = 0.05
# dup_frac = 0.2
# dup_frac = 0.4
# dup_frac = 0.6
# dup_frac = 0.8
dup_frac = 0.95
net_name = net_name+str(dup_frac)
file_name = '{}_{}layers_{}nodes'.format(net_name, n_layer, n_node_total)

# net_name = 'dup'
# n_node_total, n_layer = 500, 2
# n_node_total, n_layer = 500, 3
# n_node_total, n_layer = 500, 4
# n_node_total, n_layer = 500, 5
# n_node_total, n_layer = 500, 6

# net_name = 'rand'
# n_node_total, n_layer = 500, 2
# n_node_total, n_layer = 500, 3
# n_node_total, n_layer = 500, 4
# n_node_total, n_layer = 500, 5
# n_node_total, n_layer = 500, 6

# net_name = 'drug'
# n_node_total, n_layer = 2114, 2
# n_node_total, n_layer = 2196, 4
# n_node_total, n_layer = 2139, 3
# load each layer (a nx class object)
# with open('../data/drug_net_layer_list.pkl', 'rb') as f:
#     net_layer_list = load(f)

# net_name = 'mafia'
# n_node_total, n_layer = 143, 2

# net_name = 'london_transport'
# n_node_total = 356
# n_layer = 3
# n_node_total = 318
# n_layer = 2

# net_name = 'embassybomb1'
# n_node_total, n_layer = 22, 2

# net_name = 'elegan'
# n_node_total = 279
# n_layer = 3
# n_node_total = 273
# n_layer = 2
  
# file_name = '{}_net_{}layers_{}nodes'.format(net_name, n_layer, n_node_total)
layer_link_list = load_data('../data/{}.csv'.format(file_name))



# find overlap edges
layer_link_tuple = [tuple(item) for sublist in layer_link_list for item in sublist]
from collections import Counter
overlap_link = [ele for ele, cnt in Counter(layer_link_tuple).items() if cnt >= n_layer]


if net_name == 'drug':
    node_attr_df = pd.read_csv('../data/drug_net_attr_{}layers_{}nodes.csv'. \
                               format(n_layer, n_node_total))
    node_attr_df = node_attr_df[['Node_ID', 'Gender', 'Drug_Activity', 'Recode_Level', 'Drug_Type',
                                 'Group_Membership_Type', 'Group_Membership_Code']]    
    node_attr_dict = node_attr_df.set_index('Node_ID').to_dict('index')
    node_attr_df.drop(['Node_ID'], axis=1)
else:
    node_attr_df, node_attr_dict = None, None
layer_real_node, layer_virt_node = get_layer_node_list(layer_link_list, n_layer, n_node_total)


# layer_list_name = '{}_net_layer_list_{}layers_{}nodes'.format(net_name, n_layer, n_node_total)

# frac_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
frac_list = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95] 
# frac_list = [0.4, 0.9] 
# frac_list = [0.4, 0.7, 0.9]
n_real_node = [len(layer_real_node[i]) for i in range(n_layer)]
n_real_node_obs = [[int(frac*n_real_node[i]) for i in range(n_layer)] for frac in frac_list]     

metric_list = ['Recall_range', 'Precision_range', 'AUC-PR', 'G-mean', 'MCC', 'Recall', 'Precision', 'Accuracy',
               'TN', 'FP', 'FN', 'TP', 'Log_H', 'IG_ratio', 'ave_deg', 'link_density', 'edge_overlap_ratio']
# model_list = ['DegEM'] + ['Jaccard', 'Resource Allocation', 'Adamic Adar', 'Prefer. Attachment',
#                           'Eskin', 'Random Model', 'Random Walk']#, 'NN']   'CN']
model_list = ['EM with agg. topol.', 'EM without agg. topol.', 'Random model']
n_metric = len(metric_list)
n_model = len(model_list)
n_frac = len(frac_list)
n_rep = 10
itermax = 5


# parellel processing
if __name__ == '__main__': 
    import matplotlib
    matplotlib.use('Agg')
    t00 = time()
    run_main()
    print('Total elapsed time: {} mins'.format( round( (time()-t00)/60, 4) ) )     
    
    
# cd c:\code\illicit_net_resil\src
# python multi_net.py
    
    
    # precision: tp / (tp + fp)
    # recall: tp / (tp + fn) = tp / P  # recall > precision since fn > fp
     
    # for i_fd in range(n_fold):   
    #     for i_frac in range(len(frac_list)):
    #         print('--- Fraction: {}'.format(frac_list[i_frac]))
    #         real_node_obs = [np.random.choice(drug_net.layer_real_node[i_lyr],
    #                                               n_real_node_obs[i_frac][i_lyr],
    #                                               replace=False).tolist()\
    #                              for i_lyr in range(drug_net.n_layer)]                
    #         # append virtual nodes: all nodes - nodes in each layer
    #         real_virt_node_obs = [real_node_obs[i_lyr] + drug_net.layer_virt_node[i_lyr] \
    #                         for i_lyr in range(drug_net.n_layer)]

    #         reconst = Reconstruct(layer_link_list=drug_net.layer_link_list,
    #                               real_virt_node_obs=real_virt_node_obs, n_node_total=drug_net.n_node_total,
    #                               itermax=int(10), err_tol=1e-5)        
    #         for ele in metric_list:
    #             exec('{}_list.append(reconst.{})'.format(ele,ele)) 
    #         # # show results    
    #         # reconst.print_result()
    # metric_value_by_frac = [auc_list, prec_list, recall_list, acc_list]
    # #Plots
    # Plots.plot_roc(frac_list, fpr_list, tpr_list, auc_list)
    # Plots.plot_other(frac_list, metric_value_by_frac)
# # self = reconst      
# # import time
# t0 = time()
# main_drug_net()
# t2 = time()
# t_diff = t2-t0
# print("\n=== %s mins ===" % (t_diff/60))




# def sgl_run(idx):  #, layer_link_list, n_node_total):
#     real_virt_node_obs = [np.random.choice(n_real_node[i],
#                                   n_node_obs_list[i][idx],
#                                   replace=False).tolist()\
#                 for i in range(len(layer_df_list))]
#     reconst = Reconstruct(layer_link_list=layer_link_list,
#                           real_virt_node_obs=real_virt_node_obs, n_node_total=n_node_total,
#                           itermax=int(1e3), err_tol=1e-6) 
#     # acc_list.append(reconst.acc)
#     # metric_value = []
#     # for ele in metric_list:
#     #     metric_value.append(exec('reconst.{}'.format(ele)))
#     # metric_value = [exec('reconst.{}'.format(ele)) for ele in metric_list]  
#     return reconst.metric_value

# def get_result():
#     with mp.Pool(mp.cpu_count()-2) as pool:
#         results = pool.map(sgl_run, list(range(len(frac_list))))
#     return results

# def run_plot():
#     results = get_result()
#     fpr_list = []
#     tpr_list = []
#     auc_list = []
#     prec_list = []
#     recall_list = []
#     acc_list = []
#     for i_frac in range(len(frac_list)):
#         print('--- i frac: ', i_frac)
#         for i_mtc, mtc in enumerate(metric_list):
#             exec('{}_list.append({})'.format(mtc, results[i_frac][i_mtc].tolist()))

#     metric_value_by_frac = [auc_list, prec_list, recall_list, acc_list]
#     #Plots
#     Plots.plot_roc(frac_list, fpr_list, tpr_list, auc_list)
#     Plots.plot_other(frac_list, metric_value_by_frac)


            
# def main_toy(): 
    
    # # import data
    # path = '../data/toy_net/layer_links_3_layer.csv'
    # layer_df_list = [pd.read_csv(path, sheet_name='layer_{}'.format(i)) for i in [1,2,3]]
    # layer_link_list = [ele.to_numpy() for ele in layer_df_list]
    
    # # initilize
    # node_id_list = [set(np.concatenate(ele)) for ele in layer_link_list]
    # n_node_total = max(max(node_id_list)) + 1  
    # n_real_node = [len(ele) for ele in node_id_list]
    # frac_list = [round(0.2*i,1) for i in range(1,10)]
    # n_node_obs_list = [[int(i*n_real_node[j]) for i in frac_list] \
    #                     for j in range(len(layer_link_list))]   
    # n_fold = 1
    # # fpr_list = []
    # # tpr_list = []
    # # auc_list = []
    # # prec_list = []
    # # recall_list = []
    # # acc_list = []
    # # fpr_list, tpr_list = [], []
    # # auc_list, acc_list, prec_list, recall_list = [], [], [], []
    # metric_list
    # for ele in metric_list:
    #     exec('{}_list = []'.format(ele))
    # import multiprocessing as mp
    # def run_sgl_frac(i_frac):
    #     # print('--- Fraction: {}'.format(frac_list[i_frac]))
    #     # real_virt_node_obs = [[0,1,2], [0,4,5]]  # this comb leads to no error
    #     real_virt_node_obs = [np.random.choice(n_real_node[i],
    #                                       n_node_obs_list[i][idx],
    #                                       replace=False).tolist()\
    #                     for i in range(len(layer_df_list))]

    #     reconst = Reconstruct(layer_link_list=layer_link_list,
    #                   real_virt_node_obs=real_virt_node_obs, n_node_total=n_node_total,
    #                   itermax=int(5e3), err_tol=1e-6) 
    #     # reconst.print_result()
    #     # for ele in metric_list:
    #     #     exec('{}_list.append(reconst.{})'.format(ele,ele))
    #     return None
    # for _ in range(n_fold):         
    #     pool = mp.Pool(mp.cpu_count()-3)        
    #     results = [pool.apply(run_sgl_frac, args=(i_frac)) for i_frac in range(len(frac_list))]
    #     pool.close()   



# path = '../data/toy_net/layer_links_3_layer.csv'
# layer_df_list = [pd.read_csv(path, sheet_name='layer_{}'.format(i)) for i in [1,2,3]]
# layer_link_list = [ele.to_numpy() for ele in layer_df_list]

# # initilize
# node_id_list = [set(np.concatenate(ele)) for ele in layer_link_list]
# n_node_total = max(max(node_id_list)) + 1  
# n_real_node = [len(ele) for ele in node_id_list]
# frac_list = [round(0.2*i,1) for i in range(3, 6)]
# n_real_node_obs = [[int(frac*n) for n in multi_net.layer_n_node] for frac in frac_list ]     
# n_fold = 1
# metric_list

#     # return multi_net, frac_list, n_real_node_obs, metric_list

# # i_frac = 4
# def single_run(i_frac):  #, layer_link_list, n_node_total):
#     real_node_obs = [np.random.choice(multi_net.layer_real_node[i_lyr],
#                                           n_real_node_obs[i_frac][i_lyr],
#                                           replace=False).tolist()\
#                          for i_lyr in range(multi_net.n_layer)]                
#     # append virtual nodes: all nodes - nodes in each layer
#     real_virt_node_obs = [real_node_obs[i_lyr] + multi_net.layer_virt_node[i_lyr] \
#                     for i_lyr in range(multi_net.n_layer)]

#     reconst = Reconstruct(layer_link_list=multi_net.layer_link_list,
#                           real_virt_node_obs=real_virt_node_obs, n_node_total=multi_net.n_node_total,
#                           itermax=int(5), err_tol=1e-6)    
#     # acc_list.append(reconst.acc)
#     # metric_value = []
#     # for ele in metric_list:
#     #     metric_value.append(exec('reconst.{}'.format(ele)))
#     # metric_value = [exec('reconst.{}'.format(ele)) for ele in metric_list]  
#     # print('reconst.metric_value', reconst.metric_value)
#     return reconst.metric_value
# # self = reconst

# def run_plot():
#     results = get_result()
#     fpr_list = []
#     tpr_list = []
#     auc_list = []
#     prec_list = []
#     recall_list = []
#     acc_list = []
#     for i_frac in range(len(frac_list)):
#         for i_mtc, mtc in enumerate(metric_list):
#             exec('{}_list.append({})'.format(mtc, results[i_frac][i_mtc].tolist()))

#     metric_value_by_frac = [auc_list, prec_list, recall_list, acc_list]
#     print('\nmetric_value_by_frac: ', metric_value_by_frac)
#     #Plots
#     Plots.plot_roc(frac_list, fpr_list, tpr_list, auc_list)
#     Plots.plot_other(frac_list, metric_value_by_frac)

        
    # for i_fd in range(n_fold):
   
    #     for idx in range(len(frac_list)):
    #         # real_virt_node_obs = [[0,1,2], [0,4,5]]  # this comb leads to no error
    #         real_virt_node_obs = [np.random.choice(n_real_node[i],
    #                                           n_node_obs_list[i][idx],
    #                                           replace=False).tolist()\
    #                         for i in range(len(layer_df_list))]

    #         reconst = Reconstruct(layer_link_list=layer_link_list,
    #                       real_virt_node_obs=real_virt_node_obs, n_node_total=n_node_total,
    #                       itermax=int(5e3), err_tol=1e-6) 
    #         for ele in metric_list:
    #             exec('{}_list.append(reconst.{})'.format(ele,ele))          
    #         # # show results    
    #         reconst.print_result()
    # metric_value_by_frac = [auc_list, prec_list, recall_list, acc_list]
    # #Plots
    # Plots.plot_roc(frac_list, fpr_list, tpr_list, auc_list)
    # Plots.plot_other(frac_list, metric_value_by_frac)
     
    
# main_toy() 

# self = reconst

