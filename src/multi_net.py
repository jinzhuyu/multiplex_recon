# -*- coding: utf-8 -*-

import os
os.chdir('c:/code/illicit_net_resil/src')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
# import pandas as pd
# from scipy.stats import poisson
import networkx as nx
import pickle

import sys
sys.path.insert(0, './')
from xxx import *


multi_net = MultiNet()
self = multi_net
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
        # len(node_list) # 2196
        # max(node_list) # 2198
        self.n_node = len(node_id)
        
        net = nx.Graph()

        layer_list = link_df['Type_relation'].unique()
        link_list = []    
        for index, row in link_df.iterrows():
            link_list.append([row['Actor_A'], row['Actor_B']]) 
        # plt.figure(figsize=(10, 20))
        fig.savefig("../output/mplex_net_nx.pdf")