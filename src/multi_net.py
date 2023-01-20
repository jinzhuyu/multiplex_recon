# -*- coding: utf-8 -*-

import numpy as np  # 1.19.2
import pandas as pd  # 1.4.4
import matplotlib
import multiprocessing as mp
import os 
# from functools import reduce
from itertools import permutations, product
# from copy import deepcopy

import time
import numba  # 0.55.1
# print(numba.__version__)
# conda install -c numba numba

# metrics
from sklearn.metrics import auc, confusion_matrix  # sklearn 1.1.1
from sklearn.metrics import precision_score, recall_score, precision_recall_curve 
from sklearn.metrics import matthews_corrcoef, accuracy_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import f1_score #balanced_accuracy_score 
# from sklearn.metrics.cluster import fowlkes_mallows_score # geometric mean (G-mean)
from imblearn.metrics import geometric_mean_score  # imlearn 0.7.0
# conda install -c conda-forge imbalanced-learn
from scipy import stats
from sklearn.neighbors import KernelDensity

from my_utils import plotfuncs, npprint, copy_upper_to_lower, hel_dist
from my_utils import load_data, save_object   

class Reconstruct:
    def __init__(self, layer_link_list, node_attr_df=None, real_virt_node_obs=None, #net_layer_list=None, #layer_link_unobs_list=None,
                 real_node_obs=None, layer_real_node=None, n_node_total=None, itermax=20, err_tol=1e-5, 
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
        self.get_mtc_val()
    
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
            
        #     layer_possib_link_unobs = [ele for ele in layer_possib_links\
        #                                if ele not in layer_possib_links_obs]
        #     self.layer_possib_link_unobs.append(layer_possib_link_unobs)            
        # self.n_possib_link_unobs = np.array([len(sub) for sub in self.layer_possib_link_unobs]) 


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
        # n_link_estimate_by_layer = (self.n_link_obs*n_node_true_by_layer/ n_node_obs_by_layer).astype(int)
        self.n_link_total_by_layer = np.array([len(x) for x in self.layer_link_list])

    
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
        for i_lyr in range(self.n_layer):
            mask = self.obs_link_mask_list[i_lyr]
            self.agg_adj[mask] = 1 - np.prod(1 - self.adj_true_arr, axis=0)[mask]     
        self.agg_adj[self.agg_adj<0] = 0 
        self.agg_adj[self.agg_adj>1] = 1

    
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
            self.adj_pred_arr[i_othr, mask_0] = np.nan_to_num(self.adj_pred_arr[i_othr, mask_0])
                                    
        self.adj_pred_arr[self.adj_pred_arr<0] = 0 
        self.adj_pred_arr[self.adj_pred_arr>1] = 1


    def cal_mean_MAE(self, adj_pred_arr_last):
        '''mean absolute differences between the predicted adj at two consercutive iters 
        '''
        adj_MAE = []
        for i_lyr in range(self.n_layer):
            mask = self.link_unobs_mask[i_lyr]
            MAE = np.sum(np.abs(adj_pred_arr_last[i_lyr][mask] - self.adj_pred_arr[i_lyr][mask]))/len(mask[0])
            adj_MAE.append(MAE)
        return np.mean(adj_MAE)                 

    
    def predict_adj_EM(self, is_agg_topol_known=False, is_update_agg_topol=True):
        '''predict the adj mat of each layer using EM or EMA
        EM: is_update_agg_topol=False
        EMA: is_update_agg_topol=True. The aggregation step is integrated into EM where
             the aggregate topology is updated at every iteration.
        '''
        if is_agg_topol_known:
            # use true aggregate adj
            self.get_agg_adj() 
            is_update_agg_topol = False
        else:
            # need to estimate the aggregate adj
            self.get_agg_adj_obs()
        #initialize the network model parameters
        self.deg_seq_arr = np.random.uniform(1, self.n_node_total+1, size=(self.n_layer, self.n_node_total))
        adj_pred_arr_last = np.random.uniform(0, 1, (self.n_layer, self.n_node_total, self.n_node_total))
        # adj_MAE = []
        self.adj_MAE_diff = []
        t_start = time.time()
        for i_iter in range(self.itermax):
            # get variable required by the EM algorithm                                                
            self.deg_sum_arr = np.sum(self.deg_seq_arr, axis=1)   
            #calculate link prob by configuration model
            self.cal_link_prob_deg() 
            # update link prob using partial node sets and all links among observed nodes
            self.cal_link_prob_PON()  
            #update network model parameter: degree sequence
            self.deg_seq_arr = np.sum(self.adj_pred_arr, axis=1)
            # update aggregate adj
            if is_update_agg_topol:
                self.update_agg_adj()
            # check convergence according to changes in MAE of predicted adj mats
            self.adj_MAE_diff.append(self.cal_mean_MAE(adj_pred_arr_last)) 
            # update the latest adj pred after cal MAE
            adj_pred_arr_last = self.adj_pred_arr 
            if i_iter == 0:
                continue
            else:
                if self.adj_MAE_diff[-1] <= self.err_tol:
                    print('\nConverges at iter: {}'.format(i_iter))
                    break
                else:
                    if i_iter == self.itermax - 1:
                        print('\nNOT converged at the last iteration')                    
        run_time = time.time() - t_start
        return adj_pred_arr_last, self.sgl_link_prob_3d, self.deg_seq_arr, self.adj_MAE_diff, run_time
 
    
    def get_link_unobs_mask(self):
        self.link_unobs_mask = []
        for i_lyr in range(self.n_layer):
            self.link_unobs_mask.append((self.layer_possib_link_unobs[i_lyr][:, 0],
                                         self.layer_possib_link_unobs[i_lyr][:, 1]))
            

    def random_model(self):
        '''randomly select no of links left among the unobserved links
        '''
        adj_pred_arr = np.zeros((self.n_layer, self.n_node_total, self.n_node_total))
        adj_pred_arr = self.get_adj_obs(adj_pred_arr)
        for i_lyr in range(self.n_layer):
            link_unobs_mask = self.link_unobs_mask[i_lyr]
            top_idx = np.random.choice(len(link_unobs_mask[0]), self.n_link_left[i_lyr]) 
            mask_select = (link_unobs_mask[0][top_idx], link_unobs_mask[1][top_idx])
            adj_pred_arr[i_lyr][mask_select] = 1
        return adj_pred_arr, np.sum(adj_pred_arr, axis=1)

           
    def run_all_models(self):  
        print('\n--- EMA')
        adj_pred_arr_EMA, sgl_link_prob_3d_EMA, deg_seq_EMA, MAE_list_EMA, run_time_EMA = \
            self.predict_adj_EM(is_agg_topol_known=False, is_update_agg_topol=True)
        print('\n--- EM')
        adj_pred_arr_EM, sgl_link_prob_3d_EM, deg_seq_EM, MAE_list_EM, run_time_EM = \
            self.predict_adj_EM(is_agg_topol_known=False, is_update_agg_topol=False)
        print('\n--- RA')
        adj_pred_arr_RM, deg_seq_RM =  self.random_model()

        self.adj_pred_arr_by_mdl = [adj_pred_arr_EMA] + [adj_pred_arr_EM] + [adj_pred_arr_RM]
        self.sgl_link_prob_by_mdl = [sgl_link_prob_3d_EMA] + [sgl_link_prob_3d_EM] + [None]             
        self.deg_seq_by_mdl = [deg_seq_EMA.tolist()] + [deg_seq_EM.tolist()] + [deg_seq_RM.tolist()]
        self.MAE_by_mdl = [MAE_list_EMA] + [MAE_list_EM] + [[np.nan for i in range(len(MAE_list_EM))]] 
        self.run_time_by_mdl = [run_time_EMA] + [run_time_EM] + [0]

        print('--- Done running all models\n')

                
    def get_adj_true_unobs(self):
        adj_true_unobs_list = [[] for _ in range(self.n_layer)]
        for i_lyr in range(self.n_layer):
            # TODO: use mask
            # print('------ layer: ', i_lyr)
            for [i,j] in self.layer_possib_link_unobs[i_lyr]:
                adj_true_unobs_list[i_lyr].append(self.adj_true_arr[i_lyr][i,j])
        self.adj_true_unobs = np.concatenate(adj_true_unobs_list)


    def cal_cond_entropy(self, Z, P=None):
        ''' calculate conditional entropy: H(Z|X, Theta)
        Z is the predicted adj mat for all layers, [layer, adj_mat_row, adj_mat_col]
            Z = agg_adj*P/agg_link_prob
        P is the estimated link probability with the same dimension with Z
        e.g., cond_entropy_wt_agg_adj = self.cal_cond_entropy(Z=adj_pred_arr_EMA, P=sgl_link_prob_3d_EMA)
        '''        
        if P is not None:
            # print('\n--- Conditional entropy of EM')
            # joint_post. P(Z, X|Theta) = prod over i <j * prod over layer
                # since the estimated Z contains X. It is reduced to P(Z|Theta)
            Z = np.round(Z)
            prod_over_layer = np.prod(np.multiply(P**Z, (1-P)**(1-Z)), axis=0)
            prod_over_layer_triu = prod_over_layer[np.triu_indices(self.n_node_total, k=1)]
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
                print('------ - joint_post_log2: ', - joint_post_log2)
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
             
        # link density
        total_possib_link = self.n_node_total * (self.n_node_total - 1) / 2
        link_density_list = [np.count_nonzero(X==1)/total_possib_link for X in adj_pred_arr_sym_list]       
        
        # edge overlap compared to the layer with most links
        idx_temp = link_density_list.index(max(link_density_list))
        adj_temp = adj_pred_arr_sym_list[idx_temp]

        edge_overlap_ratio_list = \
            [np.count_nonzero(X == adj_temp)/self.n_node_total**2 for X in adj_pred_arr_sym_list]

        return link_density_list, edge_overlap_ratio_list


    def cal_deg_distri_dist(self, deg_seq_arr_pred, deg_seq_arr_true):
        '''KS distance and Hellinger distancance of degree distributions
           input deg_seq are a list and should be transformed to a np.ndarray
        '''
        deg_distri_KS = []
        deg_distri_HL = []
        for i in range(self.n_layer):
            deg_seq_pred = deg_seq_arr_pred[i]
            deg_seq_true = deg_seq_arr_true[i]
            
            min_deg = min(min(deg_seq_pred), min(deg_seq_true))
            max_deg = max(max(deg_seq_pred), max(deg_seq_true))
            x = np.arange(min_deg, max_deg+1)[:, np.newaxis]       
            
            kd_pred = KernelDensity(kernel='gaussian').fit(np.array(deg_seq_pred)[:, np.newaxis])
            kd_vals_pred = np.exp(kd_pred .score_samples(x))       
            kd_true = KernelDensity(kernel='gaussian').fit(np.array(deg_seq_true)[:, np.newaxis])
            kd_vals_true = np.exp(kd_true.score_samples(x))
     
            deg_distri_KS.append(stats.ks_2samp(kd_vals_pred, kd_vals_true)[0])  # [1] for the p value
            deg_distri_HL.append(hel_dist(kd_vals_pred, kd_vals_true))
        
        return np.mean(deg_distri_KS), np.mean(deg_distri_HL)


    def get_mtc_val_sub(self, adj_pred_arr, sgl_link_prob, deg_seq_arr_pred, run_time, MAE):
        ''' performance evaluation using multiple metrics for imbalanced data, e.g., geometric mean and MCC
        '''
        adj_pred_unobs_list = [[] for _ in range(self.n_layer)]
        for i_lyr in range(self.n_layer):
            # TODO: use mask
            for [i,j] in self.layer_possib_link_unobs[i_lyr]:
                adj_pred_unobs_list[i_lyr].append(adj_pred_arr[i_lyr][i,j])
            # link_unobs_mask = self.link_unobs_mask[i_lyr]
            # adj_pred_unobs_list[i_lyr].append(adj_pred_arr[i_lyr][link_unobs_mask])
        adj_pred_unobs = np.concatenate(adj_pred_unobs_list)
        
        precision, recall, _ = precision_recall_curve(self.adj_true_unobs, adj_pred_unobs)
        auc_pr = auc(recall, precision)       
        
        if not ((adj_pred_unobs == 0) | (adj_pred_unobs == 2)).all():
            # round the probabilistic predictions by EM
            adj_pred_unobs = np.round(adj_pred_unobs)
        # accuracy metrics
        gmean  = geometric_mean_score(self.adj_true_unobs, adj_pred_unobs)      
        mcc = matthews_corrcoef(self.adj_true_unobs, adj_pred_unobs)
        recall_val = recall_score(self.adj_true_unobs, adj_pred_unobs)
        precision_val = precision_score(self.adj_true_unobs, adj_pred_unobs)
        accuracy_val = accuracy_score(self.adj_true_unobs, adj_pred_unobs)
        tn, fp, fn, tp = confusion_matrix(self.adj_true_unobs, adj_pred_unobs, labels=[0, 1]).ravel()            
        # conditional entropy log2
        cond_entropy_log2, IG_ratio = self.cal_cond_entropy(np.round(adj_pred_arr), sgl_link_prob)
        # topological characteristics
        link_density_list, edge_overlap_ratio_list = self.get_topo_charac(adj_pred_arr)
        # distance between deg distributions
        deg_seq_arr_true = np.sum(self.adj_true_arr, axis=1)
        deg_distri_KS, deg_distri_HL = self.cal_deg_distri_dist(deg_seq_arr_pred, deg_seq_arr_true)
                
        return [recall, precision, auc_pr, gmean, mcc, recall_val, precision_val, accuracy_val,
                tn, fp, fn, tp, cond_entropy_log2, IG_ratio, link_density_list,
                edge_overlap_ratio_list, deg_distri_KS, deg_distri_HL, run_time, MAE]

    
    def get_mtc_val(self):
        self.get_adj_true_unobs()
        self.mtc_val_list = []
        for i_mdl, adj_pred in enumerate(self.adj_pred_arr_by_mdl):
            mtc_val_temp = self.get_mtc_val_sub(adj_pred, 
                                                self.sgl_link_prob_by_mdl[i_mdl],
                                                self.deg_seq_by_mdl[i_mdl],
                                                self.run_time_by_mdl[i_mdl],
                                                self.MAE_by_mdl[i_mdl]) 
            self.mtc_val_list.append(mtc_val_temp)

        
    def print_deg_and_adj(self): 
        def round_list(any_list, n_digit=4):
            return [np.round(ele, n_digit) for ele in any_list]
        
        n_space = 2
        n_dot = 19
        # n_digit = 4
        print('\nDegree sequence')
        for idx in range(self.n_layer):   
            if idx > 0:
                print(' ' * n_space, '-'*n_dot*2)
            npprint(round_list(self.deg_seq_arr)[idx], n_space)  
        
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
       
            
               


def get_layer_node_list(layer_link_list, n_layer, n_node_total):
    node_set_agg = set(np.concatenate(layer_link_list).ravel())
    layer_node = [set(np.concatenate(layer_link_list[i])) for i in range(n_layer)]
    # layer_n_node = [len(ele) for ele in layer_node]
    layer_real_node = [list(ele) for ele in layer_node]
    # node in the aggregate net but not a layer
    layer_virt_node = [list(node_set_agg.difference(ele)) for ele in layer_node]
    return layer_real_node, layer_virt_node


@numba.njit
def get_permuts_half_numba(vec: np.ndarray):
    k, size = 0, vec.size
    output = np.empty((size * (size - 1) // 2, 2))
    for i in range(size):
        for j in range(i+1, size):
            output[k,:] = [vec[i], vec[j]]
            k += 1
    return output
             
# not run: there are unobserved links when the network is large enough    
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

def get_mean_prc(x_mean, x_list, y_list):
    '''get the mean of precision (y) values at mean of recall (x)
       when # of x and y points are unequal over different repetitions
    return: a list of y mean values associated with x mean values
    '''
    y_intp_list = []
    for i in range(len(x_list)):
        y_intp_list.append(np.interp(x_mean, x_list[i][::-1], y_list[i])[::-1])             
    return np.mean(np.array(y_intp_list), axis=0).tolist()

def single_run(i_frac):
    mtc_val_rep_list = []
    for i_rep in range(n_rep):
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
                              n_node_total=n_node_total, itermax=itermax, err_tol=1e-5)    
        reconst.main() 
        mtc_val_rep_list.append(reconst.mtc_val_list)
    return mtc_val_rep_list


def paral_run():
    n_cpu = mp.cpu_count()
    if n_cpu == 8:
        n_cpu = 4
    if n_cpu > 8:
        n_cpu = int(n_cpu*0.6)
    with mp.Pool(n_cpu) as pool:
        return pool.map(single_run, range(n_frac))


def save_output(net_name, n_node_total, n_layer, frac_list, mtc_mean_by_mdl_frac):
    '''save log2H at each observatation rate to a pandas pf
    '''
    ix_log2H = mtc_list.index('Log_H')
    ix_IG_ratio = mtc_list.index('IG_ratio')
    accuracy_mtc_list = ['AUC-PR', 'G-mean', 'MCC']
    topo_charc_list = ['link_density', 'edge_overlap_ratio']
    
    ix_EMA = 0
    log2H = mtc_mean_by_mdl_frac[ix_log2H][ix_EMA][:]
    IG_ratio = mtc_mean_by_mdl_frac[ix_IG_ratio][ix_EMA][:]
    # accuracy = mtc_mean_by_mdl_frac[accuracy_loc][ix_EMA][:]
    
    output_df = pd.DataFrame(columns=['net_name', 'n_node_total', 'n_layer',
                                      'obs_frac', 'log2H'] +
                                      accuracy_mtc_list + topo_charc_list)
    output_df['obs_frac'] = frac_list
    output_df['log2H'] = log2H 
    output_df['IG_ratio'] = IG_ratio
    
    for mtc in accuracy_mtc_list + topo_charc_list:
        output_df[mtc] = mtc_mean_by_mdl_frac[mtc_list.index(mtc)][ix_EMA][:]
    output_df['net_name'] = net_name
    output_df['n_node_total'] = n_node_total
    output_df['n_layer'] = n_layer    
    
    path = '../output/log2H_by_obs_frac.csv'    
    file_exists = os.path.exists(path) 
    if file_exists:
        output_df.to_csv(path, mode='a', header=False, index=False)        
    else:                
        output_df.to_csv(path, header=True, index=False)      


def extract_mtc_val(results, is_save=True):
    mtc_val_by_mdl_frac = [ [ [[] for _ in frac_list] for _ in model_list] for _ in mtc_list]
    for i_mtc in range(n_mtc):
        for i_frac in range(n_frac):
            for i_rep in range(n_rep):
                for i_mdl in range(n_model):
                    mtc_val_by_mdl_frac[i_mtc][i_mdl][i_frac].append(
                        results[i_frac][i_rep][i_mdl][i_mtc])     
    if is_save:
       path = '../output/raw_result/{}_{}layers_{}nodes_mtc_val_by_mdl_frac.pickle'.format(
              net_name, n_layer, n_node_total)
       save_object(mtc_val_by_mdl_frac, path)
    # calculate the mean
    mtc_mean_by_mdl_frac = [ [[ [] for _ in frac_list] for _ in model_list] for _ in mtc_list]
    mtc_std_by_mdl_frac = [ [[ [] for _ in frac_list] for _ in model_list] for _ in mtc_list]
    recall_mean = np.linspace(0, 1, 50)
    for i_mtc, mtc in enumerate(mtc_list):
        for i_mdl in range(n_model):
            if mtc == 'Recall_range':
                for i_frac in range(n_frac):
                    mtc_mean_by_mdl_frac[i_mtc][i_mdl][i_frac] = recall_mean
            elif mtc == 'Precision_range':
                for i_frac in range(n_frac):
                    recall_list = mtc_val_by_mdl_frac[i_mtc-1][i_mdl][i_frac]
                    prec_list = mtc_val_by_mdl_frac[i_mtc][i_mdl][i_frac]
                    prec_mean = get_mean_prc(recall_mean, recall_list, prec_list)
                    mtc_mean_by_mdl_frac[i_mtc][i_mdl][i_frac] = prec_mean
                    # # run if mean auc pr is required
                    # calculate the mean auc pr
                    # mtc_mean_by_mdl_frac[i_mtc+1][i_mdl][i_frac] = auc(recall_mean, prec_mean) 
            elif mtc =='AUC-PR':
                pass
            elif mtc in ['G-mean', 'MCC', 'Recall', 'Precision', 'Accuracy',
                         'TN', 'FP', 'FN', 'TP', 'Log_H', 'IG_ratio',
                         'KS distance', 'Hellinger distance',
                         'Run time (s)']: 
                for i_frac in range(n_frac):
                    mtc_mean_by_mdl_frac[i_mtc][i_mdl][i_frac] = np.nanmean(
                        np.array(mtc_val_by_mdl_frac[i_mtc][i_mdl][i_frac]))
            elif mtc in ['link_density', 'edge_overlap_ratio']: 
                for i_frac in range(n_frac):  
                    mtc_mean_by_mdl_frac[i_mtc][i_mdl][i_frac] = np.mean(
                        np.array(mtc_val_by_mdl_frac[i_mtc][i_mdl][i_frac]), axis=0)
            elif mtc == 'MAE':
                if i_mdl != model_list.index('RM'): # no MAE for RM model; skipped
                    for i_frac in range(n_frac):
                        # make sublists equal in length then cal mean and std
                        max_len_temp = max([len(x) for x in mtc_val_by_mdl_frac[i_mtc][i_mdl][i_frac]])
                        for i_rep in range(n_rep):
                            if len(mtc_val_by_mdl_frac[i_mtc][i_mdl][i_frac][i_rep]) < max_len_temp:
                                val_temp = mtc_val_by_mdl_frac[i_mtc][i_mdl][i_frac][i_rep]
                                len_diff = max_len_temp - len(val_temp)
                                mtc_val_by_mdl_frac[i_mtc][i_mdl][i_frac][i_rep] += \
                                    [val_temp[-1] for _ in range(len_diff)]
                        mtc_mean_by_mdl_frac[i_mtc][i_mdl][i_frac] = np.mean(
                            np.array(mtc_val_by_mdl_frac[i_mtc][i_mdl][i_frac]), axis=0)
                        mtc_std_by_mdl_frac[i_mtc][i_mdl][i_frac] = np.std(
                            np.array(mtc_val_by_mdl_frac[i_mtc][i_mdl][i_frac]), axis=0)
            else:  # mean and standard deviation of MAE at each iter
                pass              
    if is_save:
       path = '../output/raw_result/{}_{}layers_{}nodes_metric_mean_by_mdl_frac.pickle'.format(
              net_name, n_layer, n_node_total)
       save_object(mtc_mean_by_mdl_frac, path)
       
       path = '../output/raw_result/{}_{}layers_{}nodes_metric_std_by_mdl_frac.pickle'.format(
              net_name, n_layer, n_node_total)
       save_object(mtc_std_by_mdl_frac, path)
    return mtc_mean_by_mdl_frac, mtc_std_by_mdl_frac     


def run_main():
    # get mean
    mtc_mean_by_mdl_frac, mtc_std_by_mdl_frac = extract_mtc_val(paral_run())
    #save log2H
    save_output(net_name, n_node_total, n_layer, frac_list, mtc_mean_by_mdl_frac)


net_name = 'drug'
n_node_total, n_layer = 2114, 2
# n_node_total, n_layer = 2196, 4
# n_node_total, n_layer = 2139, 3


# net_name = 'mafia'
# n_node_total, n_layer = 143, 2

# net_name = 'london_transport'
# n_node_total, n_layer = 356, 3
# n_node_total, n_layer = 318, 2

# net_name = 'embassybomb1'
# n_node_total, n_layer = 22, 2

# net_name = 'elegan'
# n_node_total, n_layer = 279, 3
# n_node_total, n_layer = 273, 2

## toy / demo network is used to test the EM part of EMA
## for this toy network, real_virt_node_obs = [[0,1,2], [0,4,5]] leads to zero error when EM converges
# net_name = 'toy'
# n_node_total, n_layer = 6, 2

  
file_name = '{}_net_{}layers_{}nodes'.format(net_name, n_layer, n_node_total)
layer_link_list, relation_list = load_data('../data/{}.csv'.format(file_name))


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


frac_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
n_real_node = [len(layer_real_node[i]) for i in range(n_layer)]
n_real_node_obs = [[int(frac*n_real_node[i]) for i in range(n_layer)] for frac in frac_list]     
mtc_list = ['Recall_range', 'Precision_range', 'AUC-PR', 'G-mean', 'MCC', 
            'Recall', 'Precision', 'Accuracy', 'TN', 'FP', 'FN', 'TP', 
            'Log_H', 'IG_ratio', 'link_density', 'edge_overlap_ratio',
            'KS distance', 'Hellinger distance', 'Run time (s)', 'MAE']
model_list = ['EMA', 'EM', 'RM']
n_mtc = len(mtc_list)
n_model = len(model_list)
n_frac = len(frac_list)
n_rep = 50
itermax = 40


# parellel processing
if __name__ == '__main__':  
    t00 = time.time()
    run_main()
    print('Net: ', net_name)
    print('n layers: ', n_layer)
    print('Total elapsed time: {} mins'.format( round( (time.time()-t00)/60, 4) ) )     
    
