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
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, accuracy_score
from time import time
import numba

from my_utils import *
from multi_net import *

 


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


# parellel processing
@numba.njit
def get_permuts_half_numba(vec: np.ndarray):
    k, size = 0, vec.size
    output = np.empty((size * (size - 1) // 2, 2))
    for i in range(size):
        for j in range(i+1, size):
            output[k,:] = [i,j]
            k += 1
    return output    

# # import data
# net_type = 'toy'
# n_node, n_layer = 6, 2

# net_type = 'rand'
# n_node, n_layer = 50, 2

net_type = 'drug'
n_node, n_layer = 2114, 2 # 2139, 3 # 2196, 4
# n_node, n_layer = 2196, 4
# n_node, n_layer = 2139, 3

net_name = '{}_net_{}layers_{}nodes'.format(net_type, n_layer, n_node)
path = '../data/{}.xlsx'.format(net_name)
layer_link_list = load_data(path)
real_node_list, virt_node_list = get_layer_node_list(layer_link_list, n_layer, n_node)

# frac_list = [0.8, 0.95]
# frac_list = [0, 0.9, 0.95] 
# frac_list = [round(0.1*i, 2) for i in range(0, 10)] + [0.95]
# frac_mean = [0.5, 0.7]
# allocate_ratio = [0.5, 1, 2]
frac_mean = [0.1, 0.5, 0.9]
allocate_ratio = [0.01, 0.1, 0.5, 1, 2, 5]
frac_2 = [[1/(1+x)*y for y in frac_mean] for x in allocate_ratio]
frac_1 = [[x/(1+x)*y for y in frac_mean] for x in allocate_ratio]
# frac_list = [frac_1, frac_2]
n_node_list_1 = len(real_node_list[0]) 
n_node_list_2 = len(real_node_list[1]) 
# n_node_list = [len(real_node_list[i]) for i in range(n_layer)]
n_node_obs_1 = [[int(frac*n_node_list_1) for frac in sub] \
                for sub in frac_1]
n_node_obs_2 = [[int(frac*n_node_list_2) for frac in sub] \
                for sub in frac_2]
# n_node_obs = [[int(frac*n_node_list[i]) for i in range(n_layer)] for frac in frac_list]     
n_rep = 10
metric_list = ['fpr', 'tpr', 'auc', 'prec', 'recall','acc']
prod_ratio_frac = list(product(list(range(len(allocate_ratio))),
                          list(range(len(frac_mean)))))
          
# def get_init_deg_seq(layer_link_list, PON_idx_list, virt_node_list):
#     ''' initialize degree sequence reduce the no. of iterations
#         note: the first iteration may do a similar job in estimating the degree sequence
#     '''
#     deg_seq_init = np.random.uniform(1, n_node+1, size=(n_layer, n_node))
#     layer_link_obs = []
#     for i_lyr in range(n_layer):
#         node_obs_temp = PON_idx_list[i_lyr]
#         links_obs_temp = [ele for ele in layer_link_list[i_lyr] \
#                           if (ele[0] in node_obs_temp and ele[1] in node_obs_temp)]
#         layer_link_obs.append(links_obs_temp)
#         # for observed real nodes, the initial degree will be the observed degree
#         for i_node in range(n_node):
#             deg_temp = len([ele for ele in links_obs_temp if i_node in ele])
#             deg_seq_init[i_lyr, i_node] = np.abs(np.random.normal(deg_temp, deg_temp))
        
#     return deg_seq_init

def plot_acc(frac_mean, allocate_ratio, acc_mean_by_ratio, n_layer, n_node):
    # linestyles = plotfuncs.linestyles()
    plotfuncs.format_fig(1.05)
    lw = .9
    med_size = 7
    colors = ['tab:{}'.format(x) for x in ['red', 'blue', 'green', 'orange', 'purple']]
    markers = ['o', 'v', 's', 'd', '*']
    plt.figure(figsize=(5, 4), dpi=400)
    # print('---acc_mean_by_ratio', acc_mean_by_ratio)
    for i in range(len(frac_mean)):
        plt.plot(allocate_ratio, np.array(acc_mean_by_ratio)[:,i],
                 color=colors[i], marker=markers[i], alpha=.85,
                 ms=med_size, lw=lw,linestyle = '--', label= frac_mean[i])        
    # plt.xlim(right=1.03)
    plt.ylim([0, 1.03])
    # plt.xlim([-0.03, 1.03])
    # plt.ylim([0.0, 1.03])
    plt.xlim([0.1/1.1, 10*1.1])
    plt.xscale('log', base=10)
    plt.xlabel("Allocation ratio")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right", fontsize=14, title='Observed fraction')
    # plt.xticks([0.2*i for i in range(5+1)])
    plt.savefig('../output/acc_vs_ratio_{}layers_{}nodes.pdf'.format(n_layer, n_node))
    plt.show() 

      
    
def sample_node_obs(layer_link_list, real_node_list, virt_node_list, i_frac, i_ratio): 
    n_node_obs = [n_node_obs_1[i_ratio][i_frac], n_node_obs_2[i_ratio][i_frac]]
    PON_idx_list_orig = [np.random.choice(real_node_list[i_lyr], n_node_obs[i_lyr],
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
        return sample_node_obs(layer_link_list, real_node_list, virt_node_list, i_frac)
    else:
        return PON_idx_list, layer_link_unobs_list

# i_frac = 1
# for 2 layer 6 node toy net, PON_idx_list = [[0,1,2], [0,4,5]] leads to zero error
def single_run(idx):  #, layer_link_list, n_node):
    i_ratio, i_frac = prod_ratio_frac[idx][0], prod_ratio_frac[idx][1]
    metric_value_rep_list = []
    for i_rep in range(n_rep):
        PON_idx_list, layer_link_unobs_list = sample_node_obs(layer_link_list, real_node_list,
                                                              virt_node_list, i_frac, i_ratio)
        # deg_seq_init = get_init_deg_seq(layer_link_list, PON_idx_list, virt_node_list) 
        t000 = time()    
        reconst = Reconstruct(layer_link_list=layer_link_list, PON_idx_list=PON_idx_list,
                              layer_link_unobs_list=layer_link_unobs_list, deg_seq_init=None,
                              n_node=n_node, itermax=int(200), eps=1e-6)    
        t100 = time()
        print('=== {} mins on this rep in total'.format( round( (t100-t000)/60, 4) ) ) 
        metric_value_rep_list.append(reconst.metric_value)
        if i_rep == n_rep - 1:
            print('--- rep: {}'.format(i_rep))
    return metric_value_rep_list
# self = reconst
# reconst.print_result()


def paral_run():
    # drug_net, frac_list, n_node_obs, metric_list = import_data()
    n_cpu = mp.cpu_count()
    if n_cpu == 8:
        n_cpu -= 3
    else:
        n_cpu = int(n_cpu*0.75)
    
    print('=== No. of CPUs selected: ', n_cpu)
    
    # n_core = 1
    with mp.Pool(n_cpu) as pool:
        results = pool.map(single_run, range(len(prod_ratio_frac)) )
    return results

# results include metric_value_rep_list for each frac. 
# metric_value_rep_list include metric_value
def run_plot():
    results = paral_run()
    # prod_ratio_frac = list(prod_ratio_frac)
    # print(results)
    acc_value_by_ratio = [[ [] for _ in frac_mean] for _ in allocate_ratio]
    # for ele in metric_list:
    #     exec('{}_list = [[] for item in range(len(frac_list))]'.format(ele))
    for idx in range(len(prod_ratio_frac)):
        i_ratio, i_frac = prod_ratio_frac[idx][0], prod_ratio_frac[idx][1]
        for i_rep in range(n_rep):
            acc_value_by_ratio[i_ratio][i_frac].append(results[idx][i_rep][-1])
    # calculate the mean
    acc_mean_by_ratio = [[ [] for _ in frac_mean] for _ in allocate_ratio]
    for i_ratio in range(len(allocate_ratio)):
        for i_frac in range(len(frac_mean)):
            # print(metric_value_by_frac[i_mtc][i_frac])
            acc_mean_by_ratio[i_ratio][i_frac] = np.nanmean(np.array(acc_value_by_ratio[i_ratio][i_frac]))
    # metric_value_by_frac = [auc_list, prec_list, recall_list, acc_list]
    # print('\nmetric_value_by_frac: ', metric_value_by_frac)
    # print('acc_value_by_ratio', acc_value_by_ratio)
    # print('acc_mean_by_ratio', acc_mean_by_ratio)
    #Plots
    plot_acc(frac_mean, allocate_ratio, acc_mean_by_ratio, n_layer, n_node)

if __name__ == '__main__': 

    import matplotlib
    matplotlib.use('Agg')
    t00 = time()
    run_plot()
    print('Total elapsed time: {} mins'.format( round( (time()-t00)/60, 4) ) ) 
