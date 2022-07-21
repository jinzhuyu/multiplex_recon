# -*- coding: utf-8 -*-

# import os
# os.chdir('c:/code/illicit_net_resil/src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import cm
import networkx as nx

import multiprocessing as mp 
from functools import reduce
from itertools import permutations, product
from copy import deepcopy
from pickle import load

from sklearn.metrics import auc
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score, accuracy_score
# from sklearn.metrics.cluster import fowlkes_mallows_score # geometric mean (G-mean)
from imblearn.metrics import geometric_mean_score


# from prg import prg
from time import time, sleep
import numba

from my_utils import *



# TODO: how many iterations are required?

# import sys
# sys.path.insert(0, './xxxx')
# from xxxx import *
# TODO: move Q[i,j] = A[i,j] for observed links outside for loop
    # : when the adj[i,j] is observed, how to save computational cost???
    # : avoid indexing those virtual observed nodes and lnks
    # : the current complexity is N_iter*(N_all^2 + N_obs^2)
    # : includethe list of observed links as an input
    
    #: use other metric for tolorance
    #: use numpy array instead of nested list in cal_link_prob deg and PON_list

# TODO: 1. learn the aggregate topology between every two layers first?
      # 2. given aggregate topology, find a way to update probability of a link when the prob of link in a layer is below 0.5 
          # as number of layers increases using other link prediciton method?
          # (Adamic adar and preferential attachment(but covert net's hubs are small)?)

class Reconstruct:
    def __init__(self, layer_link_list, node_attri_df, PON_idx_list=None, #net_layer_list=None, #layer_link_unobs_list=None,
                 n_node=None, itermax=100, eps=1e-6, 
                 simil_index_list=['jaccard_coefficient', 'preferential_attachment',
                                   'common_neighbor_centrality', 'adamic_adar_index'],
                 **kwargs):
        '''     
        Parameters
        ----------
        layer_link_list: a list containing 2d link array for each layer
        PON_idx1, PON_idx2 : node set containing the index of nodes in the subgraphs.
        n_node : number of nodes in each layer. nodes are shared over layers
        eps: error tolerance at convergence

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
        

        # if layer_link_unobs_list == None:
        self.get_unobs_link_list()
        # else:
        # self.get_layer_link_list()
        # self.get_layer_node_list()
        self.get_adj_true_arr()
        self.get_agg_adj()
        
        self.lyr_pair_list = list(permutations(list(range(self.n_layer)), r=2))
        self.get_obs_node_mask()
        
        # start_time2 = time()
        # self.predict_adj()
        # print("  --- {} mins on predicting adj".format( round((time() - start_time2)/60, 3) ) ) 
   
        # start_time3 = time()         
        # self.eval_perform()
        # print('--- {} mins on evaluating performance'.format( round( (time() - start_time3)/60, 3) ) )    
        self.run_all_models()
        self.get_metric_value()
    
    # functions for generating the ground truth of a multiplex network      
    # def get_subgraph_list(self):
    #     '''get each layer as a subgraph 
    #     '''
    #     self.sub_graph_list = []
    #     for idx in range(len(self.layer_id_list)):
    #         G_sub = nx.Graph()
    #         edges_this_layer = [(u,v) for (u,v,e) in self.G.edges(data=True)\
    #                           if e['label']==self.layer_id_list[idx]]
    #         G_sub.add_edges_from(edges_this_layer)
    #         self.sub_graph_list.append(G_sub)
        
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
    #     self.n_node = len(self.node_set_agg) 

    #     node_set_layer = [set(np.concatenate(self.layer_link_list[i]))
    #                       for i in range(len(self.layer_id_list))]
    #     self.layer_n_node = [len(ele) for ele in node_set_layer]
    #     self.real_node_list = [list(ele) for ele in node_set_layer]
    #     # node in the aggregate net but not a layer
    #     self.virt_node_list = [list(self.node_set_agg.difference(ele)) for ele in node_set_layer] 
              
    def get_adj_true_arr(self):
        def get_adj_true(link_arr):
            if not isinstance(link_arr, list):
                link_list = link_arr.tolist()
            else:
                link_list = link_arr
            A = np.zeros([self.n_node, self.n_node])
            link_arr = np.array(link_list)
            A[link_arr[:,0], link_arr[:,1]] = 1
            return A        
        self.adj_true_arr = np.zeros((self.n_layer, self.n_node, self.n_node))
        for i_lyr in range(self.n_layer):              
            self.adj_true_arr[i_lyr] = get_adj_true(self.layer_link_list[i_lyr])
            tri_up_idx = np.triu_indices(self.n_node, k=1)
            tri_low_idx = tri_up_idx[::-1]
            self.adj_true_arr[i_lyr][tri_low_idx] = \
                self.adj_true_arr[i_lyr][tri_up_idx]

    
    def get_agg_adj(self):
        '''get the aggregate network using the OR agggregation
        '''
        adj_true_diff = np.ones([self.n_layer, self.n_node, self.n_node]) - self.adj_true_arr
        self.agg_adj = np.ones([self.n_node, self.n_node]) - np.prod(adj_true_diff, axis=0)

        self.agg_adj_3d = np.repeat(self.agg_adj[:, :, np.newaxis], self.n_layer, axis=2) 
        self.agg_adj_3d = np.moveaxis(self.agg_adj_3d, 2, 0)


    def get_obs_node_mask(self):
        self.obs_node_mask_list = []
        for i_lyr in range(self.n_layer):
            obs_idx = np.array(list(permutations(self.PON_idx_list[i_lyr], r=2))).T
            if len(obs_idx) != 0:
                mask = (obs_idx[0], obs_idx[1])
            else:
                mask = []
            self.obs_node_mask_list.append(mask)

    def get_unobs_link_list(self):
        ''' Due to symmetry, only unobserved links in the upper half is selected 
        '''
        self.layer_link_unobs_list = []
        # permuts = list(permutations(range(self.n_node), r=2))
        # permuts_half = [ele for ele in permuts if ele[1] > ele[0]]
        for i_lyr in range(self.n_layer):
            node_obs = self.PON_idx_list[i_lyr]
            node_unobs = [ele for ele in range(self.n_node) if ele not in node_obs]
            # both nodes are unobserved
            link_unobs_1 = get_permuts_half_numba(np.array(node_unobs)).astype(int).tolist()
            # ubobserved links have at least one unobserved node (if vertex-induced, use links among observed nodes orig)
            # one node unobserved
            link_prod_temp = list(product(node_obs, node_unobs))
            link_unobs_2 = [[ele[0], ele[1]] for ele in link_prod_temp if ele[0] < ele[1]]
            # layer_link_unobs = [ele for ele in link_posbl if \
            #                     (ele[0] in layer_node_unobs or ele[1] in layer_node_unobs)]
            self.layer_link_unobs_list.append(np.array(link_unobs_1 + link_unobs_2))  
        self.n_link_unobs = np.array([ len(sub) for sub in self.layer_link_unobs_list ])
        # print(self.layer_link_unobs_list)
    
    def get_n_link_obs(self):
        ''' Due to symmetry, only unobserved links in the upper half is selected 
        '''
        self.n_link_obs = []
        # permuts = list(permutations(range(self.n_node), r=2))
        # permuts_half = [ele for ele in permuts if ele[1] > ele[0]]
        for i_lyr in range(self.n_layer):
            node_obs = self.PON_idx_list[i_lyr]
            link_obs = get_permuts_half_numba(np.array(node_obs)).astype(int).tolist()
            link_obs = np.array(link_obs).T
            n_link_obs_temp = np.sum(self.agg_adj[(link_obs[0], link_obs[1])] == 1)
            self.n_link_obs.append(n_link_obs_temp)
    # functions used in learn layer adj        
    # def avoid_prob_overflow(self):
    #     ''' avoid probability overflow in configuration model
    #         prob overflow can be avoided automatically if degree gusses are integers 
    #     '''
    #     self.adj_pred_arr[self.adj_pred_arr<0] = 0 
    #     self.adj_pred_arr[self.adj_pred_arr>1] = 1

    
    def cal_link_prob_deg(self):
        ''' calculate link probability between two nodes using their degrees (configuration model)
            used as prior link probability
            entries where the respective aggregate adj is 1 are updated
        '''        
        self.sgl_link_prob_3d = np.zeros((self.n_layer, self.n_node, self.n_node))
        for i in range(self.n_layer):
            self.sgl_link_prob_3d[i,:,:] = self.deg_seq_arr[i, :, None]*self.deg_seq_arr[i,:].T\
                                           / (self.deg_sum_arr[i] -1)                                       
  
        agg_link_prob = 1 - np.prod(1 - self.sgl_link_prob_3d, axis=0)        
        agg_link_prob_3d = np.repeat(agg_link_prob[:, :, np.newaxis], self.n_layer, axis=2)
        agg_link_prob_3d = np.moveaxis(agg_link_prob_3d, 2, 0)
        
        agg_link_prob_3d[agg_link_prob_3d==0] = np.nan  # avoid Runtime error: divided by 0
        self.adj_pred_arr = self.agg_adj_3d * self.sgl_link_prob_3d / agg_link_prob_3d
        self.adj_pred_arr = np.nan_to_num(self.adj_pred_arr) 
        
        self.adj_pred_arr[self.adj_pred_arr<0] = 0 
        self.adj_pred_arr[self.adj_pred_arr>1] = 1
        
    
    def cal_link_prob_PON(self):
        '''update link probability using partial observed nodes in each layer
        '''
        # links among observed nodes are observed
        for i_lyr in range(self.n_layer):
            mask = self.obs_node_mask_list[i_lyr]
            if mask:  # not empty
                self.adj_pred_arr[i_lyr][mask] = self.adj_true_arr[i_lyr][mask]
        
        # ttt0 = time()
        agg_link_prob_list = []
        for i_curr in range(self.n_layer):
            agg_link_prob_temp = 1 - np.prod(1 - np.delete(self.sgl_link_prob_3d, i_curr, axis=0), axis=0)
            agg_link_prob_temp[agg_link_prob_temp==0] = np.nan  # avoid Runtime error: divided by 0
            agg_link_prob_list.append(agg_link_prob_temp)
            
        for lyr_pair in self.lyr_pair_list:
            i_curr, i_othr = lyr_pair[0], lyr_pair[1]
            # if link is present in the aggregate network
                # if link is present in current layer and link is not between observed nodes
            mask_1 = self.adj_pred_arr[i_curr, :, :] == 1 & \
                     np.logical_not(np.isin(self.adj_pred_arr[i_othr, :, :], [0, 1]))
            self.adj_pred_arr[i_othr, mask_1] = self.agg_adj[mask_1] * self.sgl_link_prob_3d[i_othr, mask_1]
            
                # if link is not present in current layer and link is not between observed nodes                               
            agg_link_prob = agg_link_prob_list[i_curr]
            # agg_link_prob[i,j] = 0 means that sgl_link_prob_3d[i_othr, i, j] over other lyrs are all zeros
            mask_0 = self.adj_pred_arr[i_curr, :, :] == 0 & \
                     np.logical_not(np.isin(self.adj_pred_arr[i_othr, :, :], [0, 1])) 
            self.adj_pred_arr[i_othr, mask_0] = self.agg_adj[mask_0] * self.sgl_link_prob_3d[i_othr, mask_0]/ \
                                                agg_link_prob[mask_0]
            self.adj_pred_arr [i_othr, mask_0] = np.nan_to_num(self.adj_pred_arr[i_othr, mask_0])
        # print('--- Time on updating adj_pred_arr [i_othr]: {} s'.format(time()-ttt0))
        self.adj_pred_arr[self.adj_pred_arr<0] = 0 
        self.adj_pred_arr[self.adj_pred_arr>1] = 1
    
    # def cal_link_prob_MAA(self):
    #     # get average degree in each layer: <k>_alpha
    #         # self.degree_mean_arr  = 2*n_link / self.n_node
    #     # get degree of each node  in each layer: k_w^alpha
        
    #     # get number of unobserved links in each layer
        
    #     self.adj_pred_arr
    #     for i_lyr in range(self.n_layer):
    #         link_unobs = self.layer_link_unobs_list[i_lyr]
    #         for w in nx.common_neighbors(G, u, v):
    #             MAA = sum(1/np.sqrt( np.log(G_lyr[i_lyr0].degree(w))*np.log(G_lyr[i_lyr1].degree(w))) )
    # def get_JC_mask_list(self):  
    #     '''JC score for all unobserved links
    #     '''
    #     self.JC_mask_list = []
    #     for i_lyr in range(self.n_layer):
    #         jc_score = list(nx.jaccard_coefficient(self.net_layer_list[i_lyr], list(net_layer_list[i_lyr].edges)))
    #         jc_score_sort = sorted(jc_score, key = lambda x: x[2], reverse=True)
    #         link_select = np.array([ (ele[0], ele[1]) for ele in jc_score_sort[:self.n_link_unobs[i_lyr]] ])
    #         mask = (link_select[0], link_select[1])
    #         self.JC_mask_list.append(mask)
    
        
    #     # not in 0 and 1
    # def get_jc_score(self, G, u, v):
    #     '''
    #     ref.: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_prediction.jaccard_coefficient.html
    #     '''
    #     union_size = len(set(G[u]) | set(G[v]))
    #     if union_size == 0:
    #         return 0
    #     else:
    #         return len(list(nx.common_neighbors(G, u, v))) / union_size  
         
    def cal_link_prob_jc(self):
        ''' adamic_adar_index, preferential_attachment, common_neighbor_centrality
            ref.: https://networkx.org/documentation/stable/reference/algorithms/link_prediction.html
        '''
        # JC_mask_list = []
        for i_lyr in range(self.n_layer):
            G_temp = nx.from_numpy_matrix(np.ceil(self.adj_pred_arr[i_lyr]))
            # TODO: use link probability as weight
            # link_unobs = self.layer_link_unobs_list[i_lyr] 
            link_unobs_left = np.array( np.where( (self.adj_pred_arr[i_lyr] >0) & (self.adj_pred_arr[i_lyr] <1) ) )
            # n_link_unobs_left = link_unobs_left.shape[1]
            link_unobs_left_half_arr = link_unobs_left[:, link_unobs_left[0,:] < link_unobs_left[1,:]]
            link_unobs_left_half_list = list(zip(link_unobs_left_half_arr[0], link_unobs_left_half_arr[1]))
            
            # list(G_temp.edges) only contain edges in the upper half
            # jc_score = [(u, v, self.get_jc_score(G_temp, u, v)) for [u,v] in link_unobs_left_half_list]
            jc_score = list(nx.jaccard_coefficient(G_temp, link_unobs_left_half_list))
            # jc_score_sort = sorted(jc_score, key = lambda x: x[2], reverse=True)
            # jc_score_select = [ele for ele in jc_score if ele[2] >= 0.25]
            
            # link_unobs = self.layer_link_unobs_list[i_lyr] 
            # n_link_unobs_left = int(np.sum( (self.adj_pred_arr[i_lyr] >0) & (self.adj_pred_arr[i_lyr] <1) ) / 2)
            # mask_1 = np.isin(np.triu(adj_pred_temp, k=1), [0, 1])           
            link_select = [(ele[0], ele[1]) for ele in jc_score if ele[2] >= 0.333]
            if len(link_select) >=1:
                link_select = np.array(link_select).T
                try:
                    self.adj_pred_arr[i_lyr, (link_select[0], link_select[1])] = 1
                except:
                    print('------ link_select', link_select)
                    raise ValueError('no predicted links selected')
                        
            # JC_mask_list.append(mask)
            # self.adj_pred_arr[i_lyr, self.JC_mask_list[i_lyr]] = 1
         
    def predict_adj_EM(self):     
        #initialize the network model parameters
        self.deg_seq_arr = np.random.uniform(1, self.n_node+1, size=(self.n_layer, self.n_node))
        self.deg_seq_last_arr = np.zeros((self.n_layer, self.n_node))
        self.adj_pred_arr_last = np.zeros((self.n_layer, self.n_node, self.n_node))
        for iter in range(self.itermax):
            # if (iter+1) % 10 == 0: 
            #     print('  === iter: {}'.format(iter+1))                                
            
            self.deg_sum_arr = np.sum(self.deg_seq_arr, axis=1) #[np.sum(ele) for ele in self.deg_seq_list]
    
            #calculate link prob by configuration model
            self.cal_link_prob_deg()

            # update link prob using partial node sets and all links among observed nodes
            # tt0 = time()
            # self.cal_link_prob_PON()  
            # print('--- Time on cal_link_prob_PON: {} seconds'.format(round(time() - tt0, 2) ))  
            
            # self.cal_link_prob_jc()
            
            #update network model parameters
            self.deg_seq_arr = np.sum(self.adj_pred_arr, axis=1)

            # check convergence of degree sequence
            # mae_deg = np.sum(np.abs(self.deg_seq_last_arr - self.deg_seq_arr), axis=1) #/ self.n_link_unobs
            mae_link_prob = np.abs(self.adj_pred_arr_last - self.adj_pred_arr)
            #TODO: may need to use recall or precision as metrics
            # if np.all(mae < self.eps) :
            if np.all(mae_link_prob < self.eps):
                # print('\nConverges at iter: {}'.format(iter))
                break
            # else:
            #     if iter == self.itermax-1:
            #         print('\nNOT converged at the last iteration. MAE: {}\n'.\
            #               format(np.sum(mae_link_prob)))            
            
            # self.deg_seq_last_arr = self.deg_seq_arr
            self.adj_pred_arr_last = self.adj_pred_arr
            # self.adj_pred_arr_round = np.round(self.adj_pred_arr, 0)
            # self.deg_seq_last_arr_round = np.round(self.deg_seq_last_arr)   
        return self.adj_pred_arr_last

    # similarity measure for categorical data
        # ref.: https://epubs.siam.org/doi/epdf/10.1137/1.9781611972788.22
    def _apply_prediction(self, func, ebunch):
        '''ebunch: list of tuple containing a pair of nodes. E.g., ebunch = [(1,2), (2,3)] 
        '''
        return ((u, v, func(u, v)) for [u, v] in ebunch)
    
    # def jaccard(self, ebunch):
    #     def predict(u, v):
    #         u_attri, v_attri = node_attri_df.iloc[u, 1:], node_attri_df.iloc[v, 1:]
    #         intersec_size = sum(u_attri==v_attri)
    #         union_size = 2 * u_attri.size - intersec_size
    #         if union_size == 0:
    #             return 0
    #         else:
    #             return  intersec_size / union_size
    #     return self._apply_prediction(predict, ebunch)

    def overlap(self, ebunch):
        def predict(u, v):
            u_attri, v_attri = self.node_attri_df.iloc[u, 1:], self.node_attri_df.iloc[v, 1:]
            intersec_size = sum(u_attri==v_attri)
            return  intersec_size / u_attri.size
        return self._apply_prediction(predict, ebunch)

    def eskin(self, ebunch):
        n_unique = self.node_attri_df.nunique()[1:].to_numpy()
        sim_score = n_unique**2 / (n_unique**2 + 2)
        def predict(u, v):
            sim_score[self.node_attri_df.iloc[u, 1:]==self.node_attri_df.iloc[v, 1:]] = 1
            return  np.sum(sim_score) / (self.node_attri_df.shape[1]-1)
        return self._apply_prediction(predict, ebunch)

    def IOF(self, ebunch):
        freq_list = []
        for col in self.node_attri_df.columns[1:]:
            freq_list.append(self.node_attri_df[col].value_counts().to_dict())              
        def predict(u, v):
            u_attri, v_attri = self.node_attri_df.iloc[u, 1:], self.node_attri_df.iloc[v, 1:]
            sim_score = np.zeros(self.node_attri_df.shape[1]-1)
            n_attri = self.node_attri_df.shape[1]-1
            for k in range(n_attri):
                sim_score[k] = 1 / (1 + np.log(freq_list[k][u_attri[k]]) * \
                                    np.log(freq_list[k][v_attri[k]]) )
            sim_score[u_attri==v_attri] = 1
            return  np.sum(sim_score) / n_attri
        return self._apply_prediction(predict, ebunch)

    def OF(self, ebunch):
        freq_list = []
        for col in self.node_attri_df.columns[1:]:
            freq_list.append(self.node_attri_df[col].value_counts().to_dict())              
        def predict(u, v):
            u_attri, v_attri = self.node_attri_df.iloc[u, 1:], self.node_attri_df.iloc[v, 1:]
            sim_score = np.zeros(self.node_attri_df.shape[1]-1)
            n_attri = self.node_attri_df.shape[1]-1
            n_data = self.node_attri_df.shape[0]
            for k in range(n_attri):
                sim_score[k] = 1 / (1 + np.log(n_data/freq_list[k][u_attri[k]]) * \
                                    np.log(n_data/freq_list[k][v_attri[k]]) )
            sim_score[u_attri==v_attri] = 1
            return  np.sum(sim_score) / n_attri
        return self._apply_prediction(predict, ebunch)

    def Goodall4(self, ebunch): 
        freq_list = []
        for col in self.node_attri_df.columns[1:]:
            freq_list.append(self.node_attri_df[col].value_counts().to_dict())          
        def predict(u, v):
            u_attri, v_attri = self.node_attri_df.iloc[u, 1:], self.node_attri_df.iloc[v, 1:]
            sim_score = np.zeros(self.node_attri_df.shape[1]-1)
            n_attri = self.node_attri_df.shape[1]-1
            n_data = self.node_attri_df.shape[0]
            for k in range(n_attri):
                if u_attri[k] == v_attri[k]:
                    sim_score[k] = freq_list[k][u_attri[k]]*(freq_list[k][u_attri[k]] -1)/ \
                                   ( n_data*(n_data-1) )
            return  np.sum(sim_score) / n_attri
        return self._apply_prediction(predict, ebunch)
    
    def pred_adj_simil(self):
        '''predict links using similarity index 
        '''
        # layer_link_unobs_list = self.layer_link_unobs_list
        # for i_lyr in range(self.n_layer):
        #     # Convert it into a 1D array and find the indices in the 1D array
        #     index_value = self.layer_adj_arr[i_lyr, :,:]
        #     idx_1d = index_value.flatten().argsort()[-n_link_unobs:]   
        #     # convert the idx_1d back into indices arrays for each dimension
        #     x_idx, y_idx = np.unravel_index(idx_1d, index_value.shape)      
        #     # change the value of adj matrix accordingly
        #     self.layer_adj_arr[i_lyr, (x_idx, y_idx)] = 1
        
        # get partially observed adj
        # links among observed nodes are observed
        adj_pred_arr = np.zeros((self.n_layer, self.n_node, self.n_node))
        for i_lyr in range(self.n_layer):
            mask = self.obs_node_mask_list[i_lyr]
            if mask:  # not empty
                adj_pred_arr[i_lyr][mask] = self.adj_true_arr[i_lyr][mask] 
        self.get_n_link_obs()
        adj_pred_arr_simil = []
        # self.simil_index_list = ['jaccard_coefficient', 'preferential_attachment',
                                 # 'common_neighbor_centrality', 'adamic_adar_index']
        # for simil_index in self.simil_index_list:
        #     adj_pred_arr_temp = adj_pred_arr
        #     for i_lyr in range(self.n_layer):
        #         G_temp = nx.from_numpy_matrix(np.ceil(adj_pred_arr[i_lyr]))
        #         # TODO: use link probability as weight
        #         # link_unobs = self.layer_link_unobs_list[i_lyr] 
        #         n_link_left = int(np.sum(self.agg_adj==1) / 2) - self.n_link_obs[i_lyr]   # obtain from aggregate network topology: where aggregate adj ==1 - observed links among nodes 
                
        #         exec('score = list(nx.{}(G_temp, self.layer_link_unobs_list[i_lyr]))'.format(simil_index), globals() )
        #         # score = list(nx.adamic_adar_index(G_temp, self.layer_link_unobs_list[i_lyr]))
        #         # print(score)
        #         score_select = sorted(score, key = lambda x: x[2], reverse=True)[:n_link_left]         
        #         idx_link_select = [(ele[0], ele[1]) for ele in score_select]
        #         if len(idx_link_select) >= 1:
        #             idx_link_select = np.array(idx_link_select).T
        #             adj_pred_arr_temp[i_lyr, (idx_link_select[0], idx_link_select[1])] = 1
        #     adj_pred_arr_simil.append(adj_pred_arr_temp)
        print('--- Jaccard')
        adj_pred_arr_temp = deepcopy(adj_pred_arr)
        for i_lyr in range(self.n_layer):
            G_temp = nx.from_numpy_matrix(np.ceil(adj_pred_arr[i_lyr]))
            n_link_left = int(np.sum(self.agg_adj==1) / 2) - self.n_link_obs[i_lyr]   # obtain from aggregate network topology: where aggregate adj ==1 - observed links among nodes 
            
            score = list(nx.jaccard_coefficient(G_temp, self.layer_link_unobs_list[i_lyr]))
            score_select = sorted(score, key = lambda x: x[2], reverse=True)[:n_link_left]         
            idx_link_select = [(ele[0], ele[1]) for ele in score_select]
            if len(idx_link_select) >= 1:
                # print('--- Using jaccard: ', len(idx_link_select))
                idx_link_select = np.array(idx_link_select).T
                adj_pred_arr_temp[i_lyr, (idx_link_select[0], idx_link_select[1])] = 1
            else:
                print('------- No new links generated using jaccard')
        adj_pred_arr_simil.append(adj_pred_arr_temp)
        
        print('--- Include node attributes')
        print('--- Eskin')
        adj_pred_arr_temp = deepcopy(adj_pred_arr)
        for i_lyr in range(self.n_layer):
            # G_temp = nx.from_numpy_matrix(np.ceil(adj_pred_arr[i_lyr]))
            n_link_left = int(np.sum(self.agg_adj==1) / 2) - self.n_link_obs[i_lyr]   # obtain from aggregate network topology: where aggregate adj ==1 - observed links among nodes 
            
            score = list(self.eskin(self.layer_link_unobs_list[i_lyr]))
            score_select = sorted(score, key = lambda x: x[2], reverse=True)[:n_link_left]         
            idx_link_select = [(ele[0], ele[1]) for ele in score_select]
            if len(idx_link_select) >= 1:
                # print('---Using eskin: ', len(idx_link_select))
                idx_link_select = np.array(idx_link_select).T
                adj_pred_arr_temp[i_lyr, (idx_link_select[0], idx_link_select[1])] = 1
            else:
                print('------- No new links generated using eskin')
        adj_pred_arr_simil.append(adj_pred_arr_temp)        
        # adj_pred_arr_temp = adj_pred_arr
        # for i_lyr in range(self.n_layer):
        #     G_temp = nx.from_numpy_matrix(np.ceil(adj_pred_arr[i_lyr]))
        #     n_link_left = int(np.sum(self.agg_adj==1) / 2) - self.n_link_obs[i_lyr]   # obtain from aggregate network topology: where aggregate adj ==1 - observed links among nodes 
            
        #     score = list(nx.preferential_attachment(G_temp, self.layer_link_unobs_list[i_lyr]))
        #     score_select = sorted(score, key = lambda x: x[2], reverse=True)[:n_link_left]         
        #     idx_link_select = []  #[(ele[0], ele[1]) for ele in score_select]
        #     if len(idx_link_select) >= 1:
        #         print(len(idx_link_select))
        #         idx_link_select = np.array(idx_link_select).T
        #         adj_pred_arr_temp[i_lyr, (idx_link_select[0], idx_link_select[1])] = 1
        #     else:
        #         print('------- No new links by preferential_attachment')
        # adj_pred_arr_simil.append(adj_pred_arr_temp)
        
        
        # adj_pred_arr_temp = np.zeros((self.n_layer, self.n_node, self.n_node))
        # for i_lyr in range(self.n_layer):
        #     G_temp = nx.from_numpy_matrix(np.ceil(adj_pred_arr[i_lyr]))
        #     n_link_left = int(np.sum(self.agg_adj==1) / 2) - self.n_link_obs[i_lyr]   # obtain from aggregate network topology: where aggregate adj ==1 - observed links among nodes 
            
        #     score = list(nx.common_neighbor_centrality(G_temp, self.layer_link_unobs_list[i_lyr]))
        #     score_select = sorted(score, key = lambda x: x[2], reverse=True)[:n_link_left]         
        #     idx_link_select = [(ele[0], ele[1]) for ele in score_select]
        #     if len(idx_link_select) >= 1:
        #         print(len(idx_link_select))
        #         idx_link_select = np.array(idx_link_select).T
        #         adj_pred_arr_temp[i_lyr, (idx_link_select[0], idx_link_select[1])] = 1
        #     else:
        #         print('------- No new links by common_neighbor_centrality')
        # adj_pred_arr_simil.append(adj_pred_arr_temp)
        # # print('(adj_pred_arr_simil[2] == adj_pred_arr_simil[1]).all()', (adj_pred_arr_simil[2] == adj_pred_arr_simil[1]).all())
        # # print('(adj_pred_arr_simil[2] == adj_pred_arr_simil[0]).all()', (adj_pred_arr_simil[2] == adj_pred_arr_simil[0]).all())
        
        # adj_pred_arr_temp = deepcopy(adj_pred_arr)
        # for i_lyr in range(self.n_layer):
        #     G_temp = nx.from_numpy_matrix(np.ceil(adj_pred_arr[i_lyr]))
        #     n_link_left = int(np.sum(self.agg_adj==1) / 2) - self.n_link_obs[i_lyr]   # obtain from aggregate network topology: where aggregate adj ==1 - observed links among nodes 
            
        #     score = list(nx.adamic_adar_index(G_temp, self.layer_link_unobs_list[i_lyr]))
        #     score_select = sorted(score, key = lambda x: x[2], reverse=True)[:n_link_left]         
        #     idx_link_select = [(ele[0], ele[1]) for ele in score_select]
        #     if len(idx_link_select) >= 1:
        #         print(len(idx_link_select))
        #         idx_link_select = np.array(idx_link_select).T
        #         adj_pred_arr_temp[i_lyr, (idx_link_select[0], idx_link_select[1])] = 1
        #     else:
        #         print('------- No new links by adamic_adar_index')
        # adj_pred_arr_simil.append(adj_pred_arr_temp)
        
        # null model
        adj_pred_arr_temp = np.random.randint(0,2, (self.n_layer, self.n_node, self.n_node))
        for i_lyr in range(self.n_layer):
            mask = self.obs_node_mask_list[i_lyr]
            if mask:  # not empty
                adj_pred_arr_temp[i_lyr][mask] = self.adj_true_arr[i_lyr][mask] 
        adj_pred_arr_simil.append(adj_pred_arr_temp)        
        
        return adj_pred_arr_simil
    
    def run_all_models(self):
        print('--- EM')
        adj_pred_arr_EM = self.predict_adj_EM()
        adj_pred_arr_simil = self.pred_adj_simil()
        self.adj_pred_arr_list = [adj_pred_arr_EM] + adj_pred_arr_simil

                
    def get_adj_true_unobs(self):
        adj_true_unobs_list = [[] for _ in range(self.n_layer)]
        for i_lyr in range(self.n_layer):
            for [i,j] in self.layer_link_unobs_list[i_lyr]:
                adj_true_unobs_list[i_lyr].append(self.adj_true_arr[i_lyr][i,j])
        self.adj_true_unobs = np.concatenate(adj_true_unobs_list)
            
    def get_metric_value_sub(self, adj_pred_arr):
        ''' performance evaluation using multiple metrics for imbalanced data, e.g., geometric mean, MCC
        '''
        adj_pred_unobs_list = [[] for _ in range(self.n_layer)]
        for i_lyr in range(self.n_layer):
            for [i,j] in self.layer_link_unobs_list[i_lyr]:
                adj_pred_unobs_list[i_lyr].append(adj_pred_arr[i_lyr][i,j])
        adj_pred_unobs = np.concatenate(adj_pred_unobs_list)
        
        precision, recall, _ = precision_recall_curve(self.adj_true_unobs, adj_pred_unobs)
        auc_pr = auc(recall, precision)
        
        if not ((adj_pred_unobs == 0) | (adj_pred_unobs == 2)).all():
            adj_pred_unobs = np.round(adj_pred_unobs)
        gmean  = geometric_mean_score(self.adj_true_unobs, adj_pred_unobs)      
        mcc = matthews_corrcoef(self.adj_true_unobs, adj_pred_unobs)
        # f1 = f1_score(self.adj_true_unobs, adj_pred_unobs)
        recall_val = recall_score(self.adj_true_unobs, adj_pred_unobs)
        precision_val = precision_score(self.adj_true_unobs, adj_pred_unobs)
        # b_acc = balanced_accuracy_score(self.adj_true_unobs, adj_pred_unobs)  #balanced_accuracy_score
        # metric_value = [recall, precision, auc_pr, gmean, mcc] #, f1] 
        return [recall, precision, auc_pr, gmean, mcc, recall_val, precision_val]
    
    def get_metric_value(self):
        self.get_adj_true_unobs()
        self.metric_value_list = []
        for adj_pred in self.adj_pred_arr_list:
            self.metric_value_list.append(self.get_metric_value_sub(adj_pred))
        
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
    
    def plot_each_metric_sub(frac_list, metric_mean_by_model, metric, model_list, n_layer, n_node):
        
        # metric_list = ['AUC-PR', 'G-mean', 'MCC']
        
        # for mtc in 
        # linestyles = plotfuncs.linestyles()
        # metric_value_by_frac = metric_mean_by_frac[2:]
        # metric_list = ['AUC', 'Precision', 'Recall','Accuracy']
        # first_score_metric = 2
        # metric_mean_by_frac_select = metric_mean_by_frac[first_score_metric:]
        # metric_select = ['Recall', 'Precision', 'AUC-PR', 'G-mean']  #, 'MCC'][first_score_metric:]
        plotfuncs.format_fig(1.1)
        lw = .9
        med_size = 7
        colors = ['tab:{}'.format(x) for x in ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown']]
        markers = ['o', 'v', '>', '*', 's', 'd']
        plt.figure(figsize=(5, 4), dpi=400)
        for i in range(len(model_list)):
            plt.plot(frac_list, metric_mean_by_model[i], color=colors[i], marker=markers[i], alpha=.85,
                     ms=med_size, lw=lw,linestyle = '--', label=model_list[i])
                
        plt.xlim(right=1.03)
        plt.ylim([0, 1.03])
        # plt.xlim([-0.03, 1.03])
        # plt.ylim([0.0, 1.03])
        plt.xlabel(r"$c$")
        plt.ylabel(metric)
        plt.legend(loc="lower right", fontsize=13)
        plt.xticks([0.2*i for i in range(5+1)])
        plt.savefig('../output/{}layers_{}nodes_{}.pdf'.format(n_layer, n_node, metric))
        plt.show() 
        
# metric_mean_by_frac[i_mtc][i_mdl][i_frac]

    def plot_each_metric(frac_list, metric_mean_by_frac, n_layer, n_node):        
        metric_list = ['Recall', 'Precision', 'AUC-PR', 'G-mean', 'MCC', 'Recall', 'Precision']
        first_metric_select = 2
        model_list = ['EM'] + ['JC', 'AA', 'RM'] #, 'PA', 'CN', 'AA']
        for i_mtc in range(first_metric_select, len(metric_list)):
            # print('\n------ metric_mean_by_model_frac[i_mtc]', np.array(metric_mean_by_model_frac[i_mtc-2]))
            Plots.plot_each_metric_sub(frac_list, metric_mean_by_frac[i_mtc],
                                       metric_list[i_mtc], model_list, n_layer, n_node) 
            # if metric_list[i_mtc] in ['MCC', 'Recall', 'Precision']:
            #     print('------', metric_list[i_mtc], metric_mean_by_frac[i_mtc])
        
    def plot_roc(frac_list, metric_mean_by_frac, n_layer, n_node):
        recall_list, precision_list, auc_list = metric_mean_by_frac[0], metric_mean_by_frac[1], metric_mean_by_frac[2]
        # linestyles = plotfuncs.linestyles()
        plotfuncs.format_fig(1.2)
        lw = .9
        med_size = 7        
        n_frac = len(frac_list)
        # if n_frac <= 10:
        #     colors = cm.get_cmap('tab10').colors
        # else:
        #     colors = cm.get_cmap('tab20').colors  
        colors = ['tab:{}'.format(x) for x in ['red', 'blue', 'green', 'orange', 'purple']]
        markers = ['o', 'v', 's', '>', '*', 'x', '<', 'o', 'x', 'd', '*', 's']
        n_select = 5
        if n_frac <= n_select:
            selected_idx = [ele for ele in range(n_frac)]
        else:
            intvl = 2
            selected_idx = [intvl*ele for ele in range(n_select) if intvl*ele < n_frac]
        plt.figure(figsize=(5.5, 5.5*4/5), dpi=400)
        for i_idx, idx in enumerate(selected_idx):
            plt.plot(recall_list[idx], precision_list[idx], color=colors[i_idx], #marker='', #markers[i_idx], 
                     # ms=med_size, 
                     lw=lw,linestyle = linestyles[idx][1], alpha=.85,
                     # label="{:.2f} ({:0.2f})".format(frac_list[idx], auc_list[idx]))
                     label="{:.2f}".format(frac_list[idx]))
                
        # plt.plot([0, 1], [0, 1], "k--", lw=lw)
        # baseline is random classifier. P/ (P+N)
        # https://stats.stackexchange.com/questions/251175/what-is-baseline-in-precision-recall-curve
        plt.xlim([-0.015, 1.015])
        plt.ylim([-0.015, 1.015])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xticks(np.linspace(0, 1, num=6, endpoint=True))
        # plt.legend(loc="lower right", fontsize=14.5, title=r'$c$') #title=r'$c$  (AUC)')
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), title=r'$c$', loc='lower left', fontsize=14.5,)        

        plt.savefig('../output/prc_{}layers_{}nodes.pdf'.format(n_layer, n_node))
        plt.show()

    def plot_other(frac_list, metric_mean_by_frac, n_layer, n_node):
        # linestyles = plotfuncs.linestyles()
        # metric_value_by_frac = metric_mean_by_frac[2:]
        # metric_list = ['AUC', 'Precision', 'Recall','Accuracy']
        first_score_metric = 2
        metric_mean_by_frac_select = metric_mean_by_frac[first_score_metric:]
        metric_select = ['Recall', 'Precision', 'AUC-PR', 'G-mean', 'MCC', 'Recall', 'Precision']
        metric_select= metric_select[first_score_metric:]
        plotfuncs.format_fig(1.1)
        lw = .9
        med_size = 7
        colors = ['tab:{}'.format(x) for x in ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown']]
        markers = ['o', 'v', '>', '*', 's', 'd']
        plt.figure(figsize=(5, 4), dpi=400)
        for i in range(len(metric_select)):
            plt.plot(frac_list, metric_mean_by_frac_select[i], color=colors[i], marker=markers[i], alpha=.85,
                     ms=med_size, lw=lw,linestyle = '--', label=metric_select[i])
                
        plt.xlim(right=1.03)
        plt.ylim([0, 1.03])
        # plt.xlim([-0.03, 1.03])
        # plt.ylim([0.0, 1.03])
        plt.xlabel(r"$c$")
        plt.ylabel("Value of metric")
        plt.legend(loc="lower right", fontsize=13)
        plt.xticks([0.2*i for i in range(5+1)])
        plt.savefig('../output/imbl_metrics_{}layers_{}nodes.pdf'.format(n_layer, n_node))
        plt.show()                  
    

def load_data(path):
    link_df = pd.read_excel(path)
    relation_list = link_df['Relation'].unique().tolist()
    layer_link_list = []
    for idx, ele in enumerate(relation_list):
        link_temp = link_df.loc[link_df['Relation']== ele, ['From', 'To']].values.tolist()
        layer_link_list.append(link_temp)    
    return layer_link_list

def get_layer_node_list(layer_link_list, n_layer, n_node):
    node_set_agg = set(np.concatenate(layer_link_list).ravel())
    node_set_layer = [set(np.concatenate(layer_link_list[i])) for i in range(n_layer)]
    # layer_n_node = [len(ele) for ele in node_set_layer]
    real_node_list = [list(ele) for ele in node_set_layer]
    # node in the aggregate net but not a layer
    virt_node_list = [list(node_set_agg.difference(ele)) for ele in node_set_layer]
    return real_node_list, virt_node_list


@numba.njit
def get_permuts_half_numba(vec: np.ndarray):
    k, size = 0, vec.size
    output = np.empty((size * (size - 1) // 2, 2))
    for i in range(size):
        for j in range(i+1, size):
            output[k,:] = [i,j]
            k += 1
    return output
             
# there are always unobserved links when the network is large enough    
# def sample_node_obs(layer_link_list, real_node_list, virt_node_list, i_frac):    
#     PON_idx_list_orig = [np.random.choice(real_node_list[i_lyr], n_node_obs[i_frac][i_lyr],
#                          replace=False).tolist() for i_lyr in range(n_layer)]                
#     # append virtual nodes: all nodes - nodes in each layer
#     PON_idx_list = [PON_idx_list_orig[i_lyr] + virt_node_list[i_lyr] \
#                     for i_lyr in range(n_layer)]
    
#     # avoid trivial cases where all links are observed
#     is_empty = []
#     reconst_temp = Reconstruct(layer_link_list=layer_link_list,
#                                PON_idx_list=PON_idx_list, n_node=n_node)
#     layer_link_unobs_list = reconst_temp.layer_link_unobs_list
#     for i_lyr in range(reconst_temp.n_layer):
#         if reconst_temp.layer_link_unobs_list[i_lyr].size == 0:
#             is_empty.append(i_lyr)
#     if len(is_empty) == reconst_temp.n_layer:
#         print('--- No layers have unobserved links. Will resample observed nodes.')
#         return sample_node_obs(layer_link_list, real_node_list, virt_node_list, i_frac)
#     else:
#         return PON_idx_list, layer_link_unobs_list

# i_frac = 1
# for 2 layer 6 node toy net, PON_idx_list = [[0,1,2], [0,4,5]] leads to zero error
def single_run(i_frac):  #, layer_link_list, n_node):
    metric_value_rep_list = []
    for i_rep in range(n_rep):
        # PON_idx_list, layer_link_unobs_list = sample_node_obs(layer_link_list, real_node_list,
        #                                                       virt_node_list, i_frac)
        # t000 = time()
        PON_idx_list_orig = [np.random.choice(real_node_list[i_lyr], n_node_obs[i_frac][i_lyr],
                             replace=False).tolist() for i_lyr in range(n_layer)]                
        # append virtual nodes: all nodes - nodes in each layer
        PON_idx_list = [PON_idx_list_orig[i_lyr] + virt_node_list[i_lyr] \
                        for i_lyr in range(n_layer)]
        reconst = Reconstruct(layer_link_list=layer_link_list, node_attri_df=node_attri_df,
                              PON_idx_list=PON_idx_list,
                              # net_layer_list=net_layer_list,
                              # layer_link_unobs_list=layer_link_unobs_list,
                              n_node=n_node, itermax=int(1), eps=1e-2)    
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
    # drug_net, frac_list, n_node_obs, metric_list = import_data()
    n_cpu = mp.cpu_count()
    if n_cpu == 8:
        n_cpu -= 3
    else:
        n_cpu = int(n_cpu*0.7)
    
    # print('=== No. of CPUs used: ', n_cpu)
    # n_core = 1
    with mp.Pool(n_cpu) as pool:
        results = pool.map(single_run, range(n_frac))
    return results

# results include metric_value_rep_list for each frac. 
# metric_value_rep_list include metric_value

# results[i_frac][i_rep][i_mdl][i_mtc]

def run_plot():
    results = paral_run()
    # print('---n_frac', len(results))
    # print('---n_rep', len(results[0]))
    # print('---n_model', len(results[0][0]))
    # print('---n_metric', len(results[0][0][0]))
    metric_value_by_frac = [ [ [[] for _ in frac_list] for _ in model_list] for _ in metric_list]
    # for ele in metric_list:
    #     exec('{}_list = [[] for item in range(len(frac_list))]'.format(ele))
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
    for i_mtc in range(n_metric): # enumerate(metric_list):
        for i_mdl in range(n_model):
            if i_mtc == 0:
                for i_frac in range(n_frac):
                    metric_mean_by_frac[i_mtc][i_mdl][i_frac] = recall_mean
            elif i_mtc == 1:
                for i_frac in range(n_frac):
                    recall_list = metric_value_by_frac[i_mtc-1][i_mdl][i_frac]
                    prec_list = metric_value_by_frac[i_mtc][i_mdl][i_frac]
                    # print('\n--- recall_list', recall_list[0])
                    # print('\n--- prec_list', prec_list[0])
                    prec_mean = Plots.get_mean_prc(recall_mean, recall_list, prec_list)
                    # print('\n--- prec_mean', prec_mean)
                    metric_mean_by_frac[i_mtc][i_mdl][i_frac] = prec_mean
            else:                
                for i_frac in range(n_frac):
                    # print(metric_value_by_frac[i_mtc][i_frac])
                    metric_mean_by_frac[i_mtc][i_mdl][i_frac] = np.nanmean(np.array(metric_value_by_frac[i_mtc][i_mdl][i_frac]))
                    # print('---',i_mtc, i_mdl, i_frac, metric_mean_by_frac[i_mtc][i_mdl][i_frac])
    # metric_value_by_frac = [auc_list, prec_list, recall_list, acc_list]
    # print('\nmetric_value_by_frac: ', metric_value_by_frac)
    #Plots
    # Plots.plot_roc(frac_list, metric_mean_by_frac, n_layer, n_node) 
    # Plots.plot_other(frac_list, metric_mean_by_frac, n_layer, n_node)
    Plots.plot_each_metric(frac_list, metric_mean_by_frac, n_layer, n_node)

# # import data
# net_type = 'toy'
# n_node, n_layer = 6, 2

# net_type = 'rand'
# n_node, n_layer = 50, 2

net_type = 'drug'
# n_node, n_layer = 2114, 2
n_node, n_layer = 2196, 4
# n_node, n_layer = 2139, 3
# load each layer (a nx class object)
# with open('../data/drug_net_layer_list.pkl', 'rb') as f:
#     net_layer_list = load(f)
    
net_name = '{}_net_{}layers_{}nodes'.format(net_type, n_layer, n_node)
layer_link_list = load_data('../data/{}.xlsx'.format(net_name))
if net_type == 'drug':
    node_attri_df = pd.read_excel('../data/drug_net_attri_{}layers_{}nodes.xlsx'. \
                                  format(n_layer, n_node))
    node_attr_dict = node_attri_df.set_index('Node_ID').to_dict('index')
else:
    node_attr_dict = None
real_node_list, virt_node_list = get_layer_node_list(layer_link_list, n_layer, n_node)

# layer_list_name = '{}_net_layer_list_{}layers_{}nodes'.format(net_type, n_layer, n_node)
# path_layer_list = '../data/{}.pkl'.format(layer_list_name) 
# with open(path_layer_list, 'rb') as f:
#     net_layer_list = load(f)


# frac_list = [0.8, 0.95]
# frac_list = [0, 0.9, 0.95] 
# frac_list = [round(0.1*i, 2) for i in range(0, 10)] + [0.95]
# frac_list = [0, 0.1] + [round(0.2*i,1) for i in range(1, 5)] + [0.9] # [0.9, 0.95]
# frac_list = [round(0.2*i,1) for i in range(5)] + [0.9]
frac_list = [0.2, 0.4, 0.6, 0.8]
n_node_list = [len(real_node_list[i]) for i in range(n_layer)]
n_node_obs = [[int(frac*n_node_list[i]) for i in range(n_layer)] for frac in frac_list]     

# metric_list = ['fpr', 'tpr', 'auc', 'prec', 'recall','acc']
metric_list = ['Recall', 'Precision', 'AUC-PR', 'G-mean', 'MCC', 'Recall', 'Precision']
model_list = ['EM'] + ['JC', 'Eskin (with attributes)', 'RM']  #, 'PA', 'CN', 'AA']
n_metric = len(metric_list)
n_model = len(model_list)
n_frac = len(frac_list)
n_rep = 1
# TODO: use other similarity-based prediction methods

# parellel processing

if __name__ == '__main__': 

    import matplotlib
    matplotlib.use('Agg')
    t00 = time()
    run_plot()
    print('Total elapsed time: {} mins'.format( round( (time()-t00)/60, 4) ) ) 

     
    # for i_fd in range(n_fold):   
    #     for i_frac in range(len(frac_list)):
    #         print('--- Fraction: {}'.format(frac_list[i_frac]))
    #         PON_idx_list_orig = [np.random.choice(drug_net.real_node_list[i_lyr],
    #                                               n_node_obs[i_frac][i_lyr],
    #                                               replace=False).tolist()\
    #                              for i_lyr in range(drug_net.n_layer)]                
    #         # append virtual nodes: all nodes - nodes in each layer
    #         PON_idx_list = [PON_idx_list_orig[i_lyr] + drug_net.virt_node_list[i_lyr] \
    #                         for i_lyr in range(drug_net.n_layer)]

    #         reconst = Reconstruct(layer_link_list=drug_net.layer_link_list,
    #                               PON_idx_list=PON_idx_list, n_node=drug_net.n_node,
    #                               itermax=int(10), eps=1e-5)        
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




# def sgl_run(idx):  #, layer_link_list, n_node):
#     PON_idx_list = [np.random.choice(n_node_list[i],
#                                   n_node_obs_list[i][idx],
#                                   replace=False).tolist()\
#                 for i in range(len(layer_df_list))]
#     reconst = Reconstruct(layer_link_list=layer_link_list,
#                           PON_idx_list=PON_idx_list, n_node=n_node,
#                           itermax=int(1e3), eps=1e-6) 
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

# if __name__ == '__main__': 
#     run_plot()
 
            
# def main_toy(): 
    
    # # import data
    # path = '../data/toy_net/layer_links_3_layer.xlsx'
    # layer_df_list = [pd.read_excel(path, sheet_name='layer_{}'.format(i)) for i in [1,2,3]]
    # layer_link_list = [ele.to_numpy() for ele in layer_df_list]
    
    # # initilize
    # node_id_list = [set(np.concatenate(ele)) for ele in layer_link_list]
    # n_node = max(max(node_id_list)) + 1  
    # n_node_list = [len(ele) for ele in node_id_list]
    # frac_list = [round(0.2*i,1) for i in range(1,10)]
    # n_node_obs_list = [[int(i*n_node_list[j]) for i in frac_list] \
    #                     for j in range(len(layer_link_list))]   
    # n_fold = 1
    # # fpr_list = []
    # # tpr_list = []
    # # auc_list = []
    # # prec_list = []
    # # recall_list = []
    # # acc_list = []
    # # metric_list = ['fpr', 'tpr', 'auc', 'prec', 'recall','acc']
    # # fpr_list, tpr_list = [], []
    # # auc_list, acc_list, prec_list, recall_list = [], [], [], []
    # metric_list = ['fpr', 'tpr', 'auc', 'prec', 'recall','acc']
    # for ele in metric_list:
    #     exec('{}_list = []'.format(ele))
    # import multiprocessing as mp
    # def run_sgl_frac(i_frac):
    #     # print('--- Fraction: {}'.format(frac_list[i_frac]))
    #     # PON_idx_list = [[0,1,2], [0,4,5]]  # this comb leads to no error
    #     PON_idx_list = [np.random.choice(n_node_list[i],
    #                                       n_node_obs_list[i][idx],
    #                                       replace=False).tolist()\
    #                     for i in range(len(layer_df_list))]

    #     reconst = Reconstruct(layer_link_list=layer_link_list,
    #                   PON_idx_list=PON_idx_list, n_node=n_node,
    #                   itermax=int(5e3), eps=1e-6) 
    #     # reconst.print_result()
    #     # for ele in metric_list:
    #     #     exec('{}_list.append(reconst.{})'.format(ele,ele))
    #     return None
    # for _ in range(n_fold):         
    #     pool = mp.Pool(mp.cpu_count()-3)        
    #     results = [pool.apply(run_sgl_frac, args=(i_frac)) for i_frac in range(len(frac_list))]
    #     pool.close()   



# path = '../data/toy_net/layer_links_3_layer.xlsx'
# layer_df_list = [pd.read_excel(path, sheet_name='layer_{}'.format(i)) for i in [1,2,3]]
# layer_link_list = [ele.to_numpy() for ele in layer_df_list]

# # initilize
# node_id_list = [set(np.concatenate(ele)) for ele in layer_link_list]
# n_node = max(max(node_id_list)) + 1  
# n_node_list = [len(ele) for ele in node_id_list]
# frac_list = [round(0.2*i,1) for i in range(3, 6)]
# n_node_obs = [[int(frac*n) for n in multi_net.layer_n_node] for frac in frac_list ]     
# n_fold = 1
# metric_list = ['fpr', 'tpr', 'auc', 'prec', 'recall','acc']

#     # return multi_net, frac_list, n_node_obs, metric_list

# # i_frac = 4
# def single_run(i_frac):  #, layer_link_list, n_node):
#     PON_idx_list_orig = [np.random.choice(multi_net.real_node_list[i_lyr],
#                                           n_node_obs[i_frac][i_lyr],
#                                           replace=False).tolist()\
#                          for i_lyr in range(multi_net.n_layer)]                
#     # append virtual nodes: all nodes - nodes in each layer
#     PON_idx_list = [PON_idx_list_orig[i_lyr] + multi_net.virt_node_list[i_lyr] \
#                     for i_lyr in range(multi_net.n_layer)]

#     reconst = Reconstruct(layer_link_list=multi_net.layer_link_list,
#                           PON_idx_list=PON_idx_list, n_node=multi_net.n_node,
#                           itermax=int(5), eps=1e-6)    
#     # acc_list.append(reconst.acc)
#     # metric_value = []
#     # for ele in metric_list:
#     #     metric_value.append(exec('reconst.{}'.format(ele)))
#     # metric_value = [exec('reconst.{}'.format(ele)) for ele in metric_list]  
#     # print('reconst.metric_value', reconst.metric_value)
#     return reconst.metric_value
# # self = reconst

# def get_result():
#     # multi_net, frac_list, n_node_obs, metric_list = import_data()
#     with mp.Pool(mp.cpu_count()-3) as pool:
#         results = pool.map(single_run, range(len(frac_list)))
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
#         for i_mtc, mtc in enumerate(metric_list):
#             exec('{}_list.append({})'.format(mtc, results[i_frac][i_mtc].tolist()))

#     metric_value_by_frac = [auc_list, prec_list, recall_list, acc_list]
#     print('\nmetric_value_by_frac: ', metric_value_by_frac)
#     #Plots
#     Plots.plot_roc(frac_list, fpr_list, tpr_list, auc_list)
#     Plots.plot_other(frac_list, metric_value_by_frac)

        
    # for i_fd in range(n_fold):
   
    #     for idx in range(len(frac_list)):
    #         # PON_idx_list = [[0,1,2], [0,4,5]]  # this comb leads to no error
    #         PON_idx_list = [np.random.choice(n_node_list[i],
    #                                           n_node_obs_list[i][idx],
    #                                           replace=False).tolist()\
    #                         for i in range(len(layer_df_list))]

    #         reconst = Reconstruct(layer_link_list=layer_link_list,
    #                       PON_idx_list=PON_idx_list, n_node=n_node,
    #                       itermax=int(5e3), eps=1e-6) 
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

