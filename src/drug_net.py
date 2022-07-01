# -*- coding: utf-8 -*-
"""
Formate the drug trafficking network data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
# from matplotlib import cm
import networkx as nx
from copy import deepcopy

      

class DrugNet():
    
    def __init__(self, path, layer_selected_list, is_save_data=False, 
                 is_extract_char=False, is_plot=False, is_save_fig=False, **kwargs):
        '''     
        Parameters
            Path: path to the links of networks.
            layer_selected_list: list of layers to select

        Returns
            excel file containing start and end node ids of each link in the selected layers.
            node ids are continuous and start from 0
        
        '''    
        # super().__init__(**kwargs) # inherite parent class's method        
        vars = locals()
        self.__dict__.update(vars)
        del self.__dict__["self"] 
        
        self.n_layer = len(self.layer_selected_list)
        
        self.load_data()
        self.merge_layer()
        self.select_layers()
        self.correct_node_id()
        self.rename_col()
        if self.is_extract_char:
            self.gen_net()
            self.get_layer()
            self.extract_char()
        if self.is_save_data:
            self.save_df() 
        
        if self.is_plot:
            self.gen_net()
            self.plot_layer()            
    
    def load_data(self):
        self.link_df_orig = pd.read_excel(self.path, sheet_name='LINKS IND AND GROUP')
    
    def merge_layer(self):
        '''merge 'Associate/Friendship' into 'Legitimate'
           merge 'Negative/Enemies' into 'Formal Criminal Organization' 
        '''
        link_df = deepcopy(self.link_df_orig)        
        link_df.loc[link_df['Type_relation']=='Associate/Friendship', 'Type_relation'] = 'Legitimate'
        link_df.loc[link_df['Type_relation']=='Negative/Enemies', 'Type_relation'] = 'Formal Criminal Organization'
        # remove duplicates
        link_df_new = link_df.drop_duplicates(['Actor_A', 'Actor_B', 'Type_relation'], keep='last')
        self.link_df = link_df_new

    def select_layers(self): 
        '''selected layers
                4 layers: ['Co-Offenders', 'Kinship', 'Formal Criminal Organization', 'Legitimate']
                3 layers: ['Co-Offenders', 'Formal Criminal Organization', 'Legitimate']
                2 layers: ['Co-Offenders', 'Legitimate']
                layer_selected_list = ['Co-Offenders', 'Formal Criminal Organization', 'Legitimate']
        '''
        self.link_df_select = self.link_df.loc[self.link_df['Type_relation'].isin(self.layer_selected_list)]
    
    def correct_node_id(self):     
        df_temp = deepcopy(self.link_df_select)
        # let id start from 0
        node_id_min = min(list(df_temp.loc[:,['Actor_A', 'Actor_B']].min()))
        if node_id_min >= 1:
            df_temp[['Actor_A', 'Actor_B']] =  df_temp[['Actor_A', 'Actor_B']] - node_id_min 
        # find missing node ids
        node_id = pd.concat([df_temp['Actor_A'], df_temp['Actor_B']], ignore_index=True).unique().tolist()
        node_id_missed = [i for i in range(max(node_id)) if i not in node_id]
        # correct id error in df
        df_temp_new = df_temp
        for col in ['Actor_A', 'Actor_B']:
            node_list_temp = df_temp[col].tolist()
            counter = 0
            for this_id in node_id_missed:                          
                this_id -= counter
                node_list_temp = [ele - 1 if ele >= this_id else ele for ele in node_list_temp]                
                counter += 1
            df_temp_new[col] = pd.Series(node_list_temp, index=df_temp_new.index)
        node_id_new = pd.concat([df_temp_new['Actor_A'], df_temp_new['Actor_B']],
                                ignore_index=True).unique().tolist()          
        self.node_id = node_id_new
        self.n_node = len(node_id_new)
        self.link_df = df_temp_new

    def rename_col(self):
        self.link_df.rename(columns={'Actor_A': 'From', 'Actor_B': 'To', 'Type_relation': 'Relation'} , 
                            inplace=True)
    def save_df(self):
        self.link_df.to_excel('../data/drug_net_{}layers_{}nodes.xlsx'. \
                              format(self.n_layer, self.n_node), index=False)                 

    def gen_net(self):        
        G = nx.MultiGraph()
        self.layer_id_list = self.link_df['Relation'].unique() .tolist()  
        for _, row in self.link_df.iterrows():
            G.add_edge(row['From'], row['To'], label=row['Relation'])       
        self.G = G

    def get_layer(self):
        self.G_layer_list = []
        for i in range(self.n_layer):
            G_sub = nx.Graph()
            selected_edges = [(u,v) for (u,v,e) in self.G.edges(data=True) \
                              if e['label']==self.layer_selected_list[i]]
            G_sub.add_edges_from(selected_edges)
            self.G_layer_list.append(G_sub)
    
    def extract_char_sub(self, G):
        # charac_list = ['density', average']
        n_node = G.number_of_nodes()
        n_link = G.number_of_edges()
        
        density = nx.density(G)
        degree_mean = 2*n_node / float(n_link)
        cc_size = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        cc_mean = np.mean(cc_size)
        
        print('--------- No. of nodes: ', n_node)
        print('--------- No. of links: ', n_link)
        print('--------- Density: ', density)
        print('--------- Average degree: ', degree_mean)
        print('--------- Average size of connected component: ', cc_mean)
    
    def extract_char(self):
        print('--- Aggregate network')
        self.extract_char_sub(self.G)
        
        print('\n=== Each layer')
        for i in range(self.n_layer):
            print('\n------ {} layer'.format(i))
            self.extract_char_sub(self.G_layer_list[i])
                       
    def plot_layer(self): 
        # TODO: use color to represent layers or groups, or nodes in multiple layers
        # different node size
        font_size = 13
        fig, axes = plt.subplot_mosaic([['A', 'B'], ['C', 'D']], constrained_layout=True,
                                       figsize=(4*1.9, 4*2/3*1.9), dpi=300)        
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
        # i_lyr = 0
        ax_labels = [ele[0] for ele in list(axes.items())]
        for idx in range(len(ax_labels)):
            nx.draw(self.G_layer_list[idx], node_size=2)
            # print(layer_selected[counter], ': ', len(selected_edges))
            plt.sca(axes[ax_labels[idx]])
            nx.draw(self.G_layer_list[idx], node_size=2)
            # i_lyr += 1
        # save fig
        plt.tight_layout()
        if self.is_save_fig:
            # plt.savefig('../output/each_layer_net.pdf', dpi=800)
            plt.savefig('../output/drug_net_{}layers_{}nodes_each_layer.pdf'. \
                        format(self.n_layer, self.n_node), dpi=800)
        plt.show()
        
        plt.figure(figsize=(6,4), dpi=300)
        nx.draw(self.G, node_size=2)
        if self.is_save_fig:
            plt.savefig('../output/drug_net_{}layers_{}nodes_all_layer.pdf'. \
                        format(self.n_layer, self.n_node), dpi=800)
        plt.show()
        
def main():
    path='../data/drug_net_raw_data.xlsx'
    is_extract_char = True
    if is_extract_char:
        layer_selected_2d_list = [['Co-Offenders', 'Legitimate', 'Formal Criminal Organization', 'Kinship']]    
    else:
        layer_selected_2d_list = [['Co-Offenders', 'Kinship', 'Formal Criminal Organization', 'Legitimate'],
                                 ['Co-Offenders', 'Formal Criminal Organization', 'Legitimate'],
                                 ['Co-Offenders', 'Legitimate']]
    for layer_list in layer_selected_2d_list:
        print('\n=== {} layers ===', len(layer_list))
        drug_net = DrugNet(path=path, layer_selected_list=layer_list, is_extract_char=True)

if __name__ == '__main__':
    main()
