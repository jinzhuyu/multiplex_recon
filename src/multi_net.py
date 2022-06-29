# -*- coding: utf-8 -*-



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
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, accuracy_score
from time import time

from my_utils import *

# import sys
# sys.path.insert(0, './xxxx')
# from xxxx import *
# TODO: move Q[i,j] = A[i,j] for observed links outside for loop
    # : when the adj[i,j] is observed, how to save computational cost???
    # : install VS buildtools to use Cython
    # : avoid indexing those virtual observed nodes and lnks
    # : the current complexity is N_iter*(N_all^2 + N_obs^2)
    # : includethe list of observed links as an input

     

class Reconstruct:
    def __init__(self, layer_link_list, PON_idx_list=None, layer_link_unobs_list=None,
                 n_node=None, itermax=1000, eps=1e-5, **kwargs):
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
        
        self.n_layer = len(self.layer_link_list)

        if layer_link_unobs_list == None:
            self.get_unobs_link_list()    
        else:
            # self.get_layer_link_list()
            # self.get_layer_node_list()
            self.get_adj_true_list()
            self.get_agg_adj()
            # start_time2 = time()
            self.predict_adj()
            # print("--- {} mins on predicting adj".format( round( (time() - start_time2)/60, 4) ) ) 
       
            # start_time3 = time()         
            self.eval_perform()
            # print('--- {} mins on evaluating performance'.format( round( (time() - start_time3)/60, 4) ) )    
   
    
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
              
    def get_adj_true_list(self):
        def get_adj_true(link_arr):
            if not isinstance(link_arr, list):
                link_list = link_arr.tolist()
            else:
                link_list = link_arr
            A = np.zeros([self.n_node, self.n_node])
            rows, cols = [ele[0] for ele in link_list], [ele[1] for ele in link_list]
            A[rows, cols] = 1     
            return A        
        self.adj_true_list = []        
        for i_lyr in range(self.n_layer):
            self.adj_true_list.append(get_adj_true(self.layer_link_list[i_lyr]))
        # use symmetry
        for mat in self.adj_true_list:
            exec('mat[np.tril_indices(mat.shape[0], k=-1)] = mat[np.triu_indices(mat.shape[0], k=1)]')
    
    def get_agg_adj(self):
        '''get the aggregate network using the OR agggregation
        '''
        J_N = np.ones([self.n_node, self.n_node])
        adj_true_neg_list = [J_N - ele for ele in self.adj_true_list]
        self.agg_adj = J_N - reduce(np.multiply, adj_true_neg_list)

    def get_unobs_link_list(self):
        self.layer_link_unobs_list = []
        # permuts = list(permutations(range(self.n_node), r=2))
        # permuts_half = [ele for ele in permuts if ele[1] > ele[0]]
        for i_lyr in range(self.n_layer):
            layer_node_obs = self.PON_idx_list[i_lyr]
            layer_node_unobs = [ele for ele in range(self.n_node) if ele not in layer_node_obs]
            # both nodes are unobserved
            layer_link_unobs_1 = get_permuts_half(layer_node_unobs)
            # ubobserved links have at least one unobserved node (if vertex-induced, use links among observed nodes orig)
            # one node unobserved
            link_prod_temp = list(product(layer_node_obs, layer_node_unobs))
            layer_link_unobs_2 = [[ele[0], ele[1]] for ele in link_prod_temp if ele[0] < ele[1]] 
            # layer_link_unobs = [ele for ele in link_posbl if \
            #                     (ele[0] in layer_node_unobs or ele[1] in layer_node_unobs)]
            self.layer_link_unobs_list.append(np.array(layer_link_unobs_1 + layer_link_unobs_2))  
        # print(self.layer_link_unobs_list)



    # functions used in learn layer adj        
    def avoid_prob_overflow(self, adj_pred_list):
        ''' avoid probability overflow in configuration model
            prob overflow can be avoided automatically if degree gusses are integers 
        '''
        for Q in adj_pred_list:
            Q[Q<0] = 0 
            Q[Q>1] = 1 
        return adj_pred_list
    
    def cal_link_prob_deg(self):
        ''' calculate link probability between two nodes using their degrees (configuration model)
            used as prior link probability
        '''
        adj_pred_list = self.adj_pred_list
        n_node = self.n_node
        tri_up_idx = np.triu_indices(n_node, k=1)
        rows, cols = tri_up_idx[0].tolist(), tri_up_idx[1].tolist()
        #TODO: only calculate  single_prob_rev_list once for all pairs of nodes?       
        for i_idx in range(len(rows)):
            i, j = rows[i_idx], cols[i_idx]
            single_prob_rev_list = [1-self.deg_seq_list[ele][i]*self.deg_seq_list[ele][j]/\
                                    (self.deg_sum_list[ele]-1) for ele in range(self.n_layer)]
            agg_link_prob = 1 - np.prod(single_prob_rev_list)
            # agg_link_prob = agg_link_prob_list[i_idx] 
            for idx, Q in enumerate(adj_pred_list):
                if agg_link_prob == 0:
                    Q[i,j] = 0
                else:
                    # single link prob using degree of two nodes: page 27 in SI
                    Q[i,j] = self.agg_adj[i,j]*(1 - single_prob_rev_list[idx])/agg_link_prob
                    # Q[i,j] = self.agg_adj[i,j]*single_prob_list[idx]/agg_link_prob
                Q[j,i] = Q[i,j]
        # use symmetry
        # [exec('mat[np.tril_indices(mat.shape[0], k=-1)] = mat[np.triu_indices(mat.shape[0], k=1)]') \
        #  for mat in adj_pred_list]
        self.adj_pred_list = self.avoid_prob_overflow(adj_pred_list)      
    
    def cal_link_prob_PON(self):
        '''update link probability using partial observed nodes in each layer
        '''
        # links among observed nodes are observed
        adj_pred_list = self.adj_pred_list
        for i_curr, curr_node_list in enumerate(self.PON_idx_list):
            permuts = list(permutations(curr_node_list, r=2))
            permuts_half = [ele for ele in permuts if ele[1] > ele[0]]
            rows, cols = [ele[0] for ele in permuts_half], [ele[1] for ele in permuts_half]                  
                
            for i_idx in range(len(permuts_half)):
                i, j = rows[i_idx], cols[i_idx]
                # # TODO: suppose only a portion of links among observed nodes are observed
                adj_pred_list[i_curr][i,j] = self.adj_true_list[i_curr][i,j]
                adj_pred_list[i_curr][j,i] = adj_pred_list[i_curr][i,j]
    
                # OR-aggregate mechanism: page 25 in SI
                if self.agg_adj[i,j] == 1:
                    othr_lyr_idx = [ele for ele in range(self.n_layer) if ele != i_curr]
                    # prior link probability
                    sgl_link_prob_list = [self.deg_seq_list[ele][i]*self.deg_seq_list[ele][j] \
                                          /(self.deg_sum_list[ele] - 1) \
                                          for ele in range(self.n_layer)]
                    # determine the actual predicted link [i,j] probability in other layers
                    if adj_pred_list[i_curr][i,j] == 1:
                        for i_othr in othr_lyr_idx:
                            if adj_pred_list[i_othr][i,j] not in [0, 1]: # !!! question: why not =0 and not =1:
                                adj_pred_list[i_othr][i,j] = sgl_link_prob_list[i_othr]
                                adj_pred_list[i_othr][j,i] = sgl_link_prob_list[i_othr]
                    if adj_pred_list[i_curr][i,j] == 0:
                        # make at least one Q_ij = 1 to make A0_ij = 1
                        if len(othr_lyr_idx) == 1:
                            i_othr = othr_lyr_idx[0]
                            adj_pred_list[i_othr][i,j], adj_pred_list[i_othr][j,i] = 1, 1                            
                        else: 
                            # pi /1- prod_i(1-pi). If two layers only, this formula is always 1.
                            prod_temp = np.prod([1-sgl_link_prob_list[i_othr] for i_othr in othr_lyr_idx])
                            if prod_temp < 1:  # such that prod_temp != 1
                                for i_othr in othr_lyr_idx:
                                    adj_pred_list[i_othr][i,j] = sgl_link_prob_list[i_othr]/ (1 - prod_temp)
                                    adj_pred_list[i_othr][j,i] = adj_pred_list[i_othr][i,j]
                        #     max_prob = np.max(sgl_link_prob_list)
                        #     if max_prob != 0:
                        #         # normalize each single link prob by the max
                        #         # so that the max automatically becomes the chosen 1
                        #         for i_othr in othr_lyr_idx:
                        #             adj_pred_list[i_othr][i,j] = sgl_link_prob_list[i_othr]/max_prob
                        #             adj_pred_list[i_othr][j,i] = adj_pred_list[i_othr][i,j]
                        #     else: # TODO: randomly select one?
                        #         rand_idx = np.random.choice(othr_lyr_idx)
                        #         adj_pred_list[rand_idx][i,j], adj_pred_list[rand_idx][j,i] = 1, 1
                        # else: # two layers in total
                        #     i_othr = othr_lyr_idx[0]
                        #     adj_pred_list[i_othr][i,j], adj_pred_list[i_othr][j,i] = 1, 1
                # else:
                #     for i_lyr in range(self.n_layer):  #[ele for ele in range(self.n_layer) if ele != i_curr]
                #         adj_pred_list[i_lyr][i,j], adj_pred_list[i_lyr][j,i] = 0, 0
        # use symmetry
        # [exec('mat[np.tril_indices(mat.shape[0], k=-1)] = mat[np.triu_indices(mat.shape[0], k=1)]') \
        #  for mat in adj_pred_list]
        self.adj_pred_list = self.avoid_prob_overflow(adj_pred_list)
           
    def predict_adj(self):     
        #initialize the network model parameters
        self.deg_seq_list = [np.random.uniform(1, self.n_node+1, size=self.n_node)
                             for idx in range(self.n_layer)]
        self.deg_seq_last_list = [np.zeros(self.n_node) for idx in range(self.n_layer)] 
        # t0 = time()
        for iter in range(self.itermax):
            # if (iter+1) % 1000 == 0: 
            #     print('  === iter: {}'.format(iter+1))
            # init          
            self.adj_pred_list = [np.zeros([self.n_node, self.n_node]) for idx in range(self.n_layer)]
            self.deg_sum_list = [np.sum(ele) for ele in self.deg_seq_list]
    
            #calculate link prob by configuration model
            self.cal_link_prob_deg()

            # update link prob using partial node sets and all links among observed nodes
            self.cal_link_prob_PON()        
       
            #update network model parameters
            self.deg_seq_list  = [np.sum(ele, axis=0) for ele in self.adj_pred_list]
            # print('deg_seq_list', self.deg_seq_list)

            # check convergence of degree sequence
            cond = [np.sum(np.abs(self.deg_seq_last_list[ele]-self.deg_seq_list[ele])) < self.eps\
                    for ele in range(self.n_layer)]            
            if all(cond):
                print('\nConverges at iter: {}'.format(iter))
                break
            else:
                if iter + 1 == self.itermax:
                    err = [np.sum(np.abs(self.deg_seq_last_list[ele]-self.deg_seq_list[ele])) for ele in range(self.n_layer)]
                    print('\nNOT converged at the last iteration. Err: {}\n'.format(err))
            
            self.deg_seq_last_list = self.deg_seq_list
            # t1 = time()
            # t_diff = t1-t0
            # print("--- Time: {} mins after {} iters".format(round(t_diff/60, 2), iter+1))
            
    def eval_perform(self):
        ''' performance using accuracy, precision, recall, as well as roc curve and AUC
        '''
        n_digit = 0
        self.adj_pred_list_round = [np.round(ele, n_digit) for ele in self.adj_pred_list]
        self.deg_seq_last_list_round = [np.round(ele, n_digit) for ele in self.deg_seq_last_list]  
        
        adj_true_unobs_list = [[] for _ in range(self.n_layer)]
        adj_pred_unobs_list = [[] for _ in range(self.n_layer)]
        # adj_pred_unobs_round_list = [[] for _ in range(self.n_layer)]
        for idx in range(self.n_layer):
            for [i,j] in self.layer_link_unobs_list[idx]:
                adj_true_unobs_list[idx].append(self.adj_true_list[idx][i,j])
                adj_pred_unobs_list[idx].append(self.adj_pred_list[idx][i,j])
                # adj_pred_unobs_round_list[idx].append(self.adj_pred_list_round[idx][i,j]) 

        adj_true = np.concatenate(adj_true_unobs_list)
        adj_pred = np.concatenate(adj_pred_unobs_list)
        adj_pred_round = np.round(adj_pred)
        
        fpr, tpr, thresholds = roc_curve(adj_true, adj_pred)        
        auc_val = auc(fpr, tpr)
        prec = precision_score(adj_true, adj_pred_round, average='binary')
        if 1 not in adj_pred_round or 0 not in adj_true:
            fpr = [0 for _ in tpr]
            auc_val = 1  #np.nan
            prec = 1 #np.nan
        if 1 not in adj_true or 0 not in adj_pred_round:
            tpr = [1 for _ in fpr] 
            auc_val = 1 #np.nan
            prec = 1 #np.nan
        # if(np.isnan(prec).any()):
        # print('\n--- auc_val', auc_val, 'fpr', np.round(fpr, 2), 'tpr', np.round(tpr, 2))
        # print('--- adj_true', adj_true, 'adj_pred_round', adj_pred_round, '\n') 
            # print('\n--- adj_pred', adj_pred)
        recall = recall_score(adj_true, adj_pred_round, average='binary')
        acc = accuracy_score(adj_true, adj_pred_round)
        self.metric_value = [fpr, tpr, auc_val, prec, recall, acc]  

        # # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
        # pred_labels, true_labels = adj_pred_round, adj_true       
        # TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))        
        # # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        # TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))         
        # # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        # FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))         
        # # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        # FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))       
        # prec_cal = TP / (TP + FP)  #!!! why is FP = 0??? no ones in pred adj?
        # recall_cal = TP / (TP + FN)  # because FN is high if FP is low
        # acc_cal = (TP+TN) / (TP+TN + FP+FN)         
        # print('TP: {}, FP: {}, TN: {}, FN: {}'.format(TP,FP,TN,FN))        
        # print('prec_cal: {}, recall_cal: {}, acc_cal: {}'.format(prec_cal,recall_cal,acc_cal))
        
    
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
            npprint(self.adj_pred_list_round[idx], n_space)  
       
        
    def gen_sub_graph(self, sample_meth='random_unif'):
        ''' 
        input:
            sample_meth: 'random_unif' - uniformly random or 'random_walk'          
        TODO:
            the following two lines should be modified the observed links should be among observed nodes
            'if np.random.uniform(0, 1) < self.frac_obs_link:
                 adj_pred_list[i_curr][i,j] = self.adj_true_list[i_curr][i,j]'
        '''                        


class Plots:         
  
    def get_mean_roc(fpr_mean, fpr_list, tpr_list):
        tprs_intp_list = []
        for i in range(len(fpr_list)):
            tprs_intp_list.append(np.interp(fpr_mean, fpr_list[i], tpr_list[i]))       
        tpr_mean = np.mean(np.array(tprs_intp_list), axis=0).tolist()       
        return tpr_mean
        
    def plot_roc(frac_list, metric_mean_by_frac, n_layer, n_node):
        fpr_list, tpr_list, auc_list = metric_mean_by_frac[0], metric_mean_by_frac[1], metric_mean_by_frac[2]
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
        markers = ['o', 'v', 's', 'd', '*', 'x', 'v', 'o', 'x', 'd', '*', 's']
        n_select = 4
        if n_frac <= n_select:
            selected_idx = [ele for ele in range(n_frac)]
        else:
            intvl = 2
            selected_idx = [intvl*ele for ele in range(n_select) if intvl*ele < n_frac]
        plt.figure(figsize=(5.5, 5.5*4/5), dpi=400)
        for i_idx, idx in enumerate(selected_idx):
            plt.plot(fpr_list[idx], tpr_list[idx], color=colors[i_idx], marker=markers[i_idx], 
                     ms=med_size, lw=lw,linestyle = '--', alpha=.85,
                     # label="{:.2f} ({:0.2f})".format(frac_list[i], auc_list[i]))
                     label="{:.2f}".format(frac_list[idx]))
                
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([-0.015, 1.015])
        plt.ylim([-0.015, 1.015])
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.xticks([0.2*i for i in range(5+1)])
        # plt.legend(loc="lower right", fontsize=14.5, title=r'$c$') #title=r'$c$  (AUC)')
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), title=r'$c$', loc='lower right', fontsize=14.5,)        

        plt.savefig('../output/roc_{}layers_{}nodes_test.pdf'.format(n_layer, n_node))
        plt.show()

    def plot_other(frac_list, metric_mean_by_frac, n_layer, n_node):
        # linestyles = plotfuncs.linestyles()
        metric_value_by_frac = metric_mean_by_frac[2:]
        metric_list = ['AUC', 'Precision', 'Recall','Accuracy']
        plotfuncs.format_fig(1.15)
        lw = .9
        med_size = 7
        colors = ['tab:{}'.format(x) for x in ['red', 'blue', 'green', 'orange']]
        markers = ['o', 'v', 's', 'd']
        plt.figure(figsize=(5, 4), dpi=400)
        for i in range(len(metric_list)):
            plt.plot(frac_list, metric_value_by_frac[i], color=colors[i], marker=markers[i], alpha=.85,
                     ms=med_size, lw=lw,linestyle = '--', label=metric_list[i])
                
        plt.xlim(right=1.03)
        plt.ylim([0, 1.03])
        # plt.xlim([-0.03, 1.03])
        # plt.ylim([0.0, 1.03])
        plt.xlabel(r"$c$")
        plt.ylabel("Value of metric")
        plt.legend(loc="lower right", fontsize=13)
        plt.xticks([0.2*i for i in range(5+1)])
        plt.savefig('../output/metrics_{}layers_{}nodes_test.pdf'.format(n_layer, n_node))
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

# import data
# net_type = 'toy'
# n_node, n_layer = 6, 2

# net_type = 'rand'
# n_node, n_layer = 30, 2

net_type = 'drug'
n_node, n_layer = 2114, 2 # 2139, 3 # 2196, 4

net_name = '{}_net_{}layers_{}nodes'.format(net_type, n_layer, n_node)
path = '../data/{}.xlsx'.format(net_name)
layer_link_list = load_data(path)
real_node_list, virt_node_list = get_layer_node_list(layer_link_list, n_layer, n_node)

# frac_list = [0.8, 0.95]
# frac_list = [0, 0.9, 0.95] 
# frac_list = [round(0.1*i, 2) for i in range(0, 10)] + [0.95]
frac_list = [round(0.2*i,1) for i in range(0, 5)] + [0.9, 0.95]
n_node_list = [len(real_node_list[i]) for i in range(n_layer)]
n_node_obs = [[int(frac*n_node_list[i]) for i in range(n_layer)] for frac in frac_list]     
n_rep = 2
metric_list = ['fpr', 'tpr', 'auc', 'prec', 'recall','acc']

# parellel processing

def sample_node_obs(layer_link_list, real_node_list, virt_node_list, i_frac):
    
    PON_idx_list_orig = [np.random.choice(real_node_list[i_lyr], n_node_obs[i_frac][i_lyr],
                         replace=False).tolist() for i_lyr in range(n_layer)]                
    # append virtual nodes: all nodes - nodes in each layer
    PON_idx_list = [PON_idx_list_orig[i_lyr] + virt_node_list[i_lyr] \
                    for i_lyr in range(n_layer)]
    
    # avoid trivial cases where all links are observed
    is_empty = []
    reconst_temp = Reconstruct(layer_link_list=layer_link_list,
                               PON_idx_list=PON_idx_list, n_node=n_node)
    layer_link_unobs_list = reconst_temp.layer_link_unobs_list
    for i_lyr in range(reconst_temp.n_layer):
        if reconst_temp.layer_link_unobs_list[i_lyr].size == 0:
            is_empty.append(i_lyr)
    if len(is_empty) == reconst_temp.n_layer:
        print('--- No layers have unobserved links. Will resample observed nodes.')
        return sample_node_obs(drug_net, i_frac)
    else:
        return PON_idx_list, layer_link_unobs_list

# i_frac = 1
# for 2 layer 6 node toy net, PON_idx_list = [[0,1,2], [0,4,5]] leads to zero error
def single_run(i_frac):  #, layer_link_list, n_node):
    metric_value_rep_list = []
    for i_rep in range(n_rep):
        PON_idx_list, layer_link_unobs_list = sample_node_obs(layer_link_list, real_node_list,
                                                              virt_node_list, i_frac)
        t000 = time()    
        reconst = Reconstruct(layer_link_list=layer_link_list, PON_idx_list=PON_idx_list,
                              layer_link_unobs_list=layer_link_unobs_list, 
                              n_node=n_node, itermax=int(20), eps=1e-6)    
        t100 = time()
        print('=== {} mins on this rep in total'.format( round( (t100-t000)/60, 4) ) ) 
        metric_value_rep_list.append(reconst.metric_value)
        if i_rep == n_rep - 1:
            print('--- rep: {}'.format(i_rep))
    return metric_value_rep_list
# self = reconst

def paral_run():
    # drug_net, frac_list, n_node_obs, metric_list = import_data()
    n_core = mp.cpu_count()-3
    # n_core = 1
    with mp.Pool(n_core) as pool:
        results = pool.map(single_run, range(len(frac_list)))
    return results

# results include metric_value_rep_list for each frac. 
# metric_value_rep_list include metric_value
def run_plot():
    results = paral_run()
    # print(results)
    metric_value_by_frac = [[ [] for _ in range(len(frac_list))] for _ in metric_list]
    for ele in metric_list:
        exec('{}_list = [[] for item in range(len(frac_list))]'.format(ele))
    for i_mtc in range(len(metric_list)):
        for i_frac in range(len(frac_list)):
            for i_rep in range(n_rep):
                metric_value_by_frac[i_mtc][i_frac].append(results[i_frac][i_rep][i_mtc])
    # calculate the mean
    metric_mean_by_frac = [[ [] for _ in range(len(frac_list))] for _ in metric_list]
    
    fpr_mean = np.linspace(0, 1, 20)
    for i_mtc, mtc in enumerate(['fpr', 'tpr', 'auc', 'prec', 'recall','acc']):
        if mtc == 'fpr':
            for i_frac in range(len(frac_list)):
                metric_mean_by_frac[i_mtc][i_frac] = fpr_mean
        elif mtc == 'tpr':
            for i_frac in range(len(frac_list)):
                fpr_list = metric_value_by_frac[i_mtc-1][i_frac]
                tpr_list = metric_value_by_frac[i_mtc][i_frac]
                tpr_mean = Plots.get_mean_roc(fpr_mean, fpr_list, tpr_list)
                metric_mean_by_frac[i_mtc][i_frac] = tpr_mean
        else:                
            for i_frac in range(len(frac_list)):
                # print(metric_value_by_frac[i_mtc][i_frac])
                metric_mean_by_frac[i_mtc][i_frac] = np.nanmean(np.array(metric_value_by_frac[i_mtc][i_frac]))
    # metric_value_by_frac = [auc_list, prec_list, recall_list, acc_list]
    # print('\nmetric_value_by_frac: ', metric_value_by_frac)
    #Plots
    Plots.plot_roc(frac_list, metric_mean_by_frac, n_layer, n_node)
    Plots.plot_other(frac_list, metric_mean_by_frac, n_layer, n_node)

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

