# -*- coding: utf-8 -*-
"""
Formate the drug trafficking network data
"""

# import os
# os.chdir('c:/code/illicit_net_resil/src')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
# from matplotlib import cm
import networkx as nx
from copy import deepcopy

      

class DrugNet():
    
    def __init__(self, path,layer_selected_list, is_plot=False, is_save_fig=False, **kwargs):
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

        self.load_data()
        self.merge_layer()
        self.select_layers()
        self.correct_node_id()
        self.rename_col()
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
        self.link_df = df_temp_new

    def rename_col(self):
        self.link_df.rename(columns={'Actor_A': 'From', 'Actor_B': 'To', 'Type_relation': 'Relation'} , 
                            inplace=True)
    def save_df(self):
        self.n_layer, self.n_node = len(self.layer_selected_list), len(self.node_id)
        self.link_df.to_excel('../data/drug_net_{}layers_{}nodes.xlsx'. \
                              format(self.n_layer, self.n_node), index=False)                 

    def gen_net(self):        
        G = nx.MultiGraph()
        self.layer_id_list = self.link_df['Relation'].unique() .tolist()  
        for _, row in self.link_df.iterrows():
            G.add_edge(row['From'], row['To'], label=row['Relation'])       
        self.G = G
                       
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
        counter = 0
        layer_selected = self.layer_selected
        ax_labels = [ele[0] for ele in list(axes.items())]
        for idx in range(len(ax_labels)):
            G_sub = nx.Graph()
            selected_edges = [(u,v) for (u,v,e) in self.G.edges(data=True) \
                              if e['label']==layer_selected[counter]]
            G_sub.add_edges_from(selected_edges)
            plt.sca(axes[ax_labels[idx]])
            nx.draw(G_sub, node_size=2)
            print(layer_selected[counter], ': ', len(selected_edges))
            counter += 1
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
    layer_selected_2d_list = [['Co-Offenders', 'Kinship', 'Formal Criminal Organization', 'Legitimate'],
                             ['Co-Offenders', 'Formal Criminal Organization', 'Legitimate'],
                             ['Co-Offenders', 'Legitimate']]
    for layer_list in layer_selected_2d_list:
        drug_net = DrugNet(path=path, layer_selected_list=layer_list)

if __name__ == '__main__':
    main()
