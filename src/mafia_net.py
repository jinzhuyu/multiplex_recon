# -*- coding: utf-8 -*-
"""
Formate the Sicilian Mafia network data
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import networkx as nx
from copy import deepcopy

from my_utils import get_net_charac_sub

class Net():
    
    def __init__(self, path, is_save_data=False, is_get_net_charac=False, 
                 is_plot=False, is_save_fig=False,
                 **kwargs):
        '''     
        Parameters
            Path: path to the links of networks.
            layer_selected_list: list of layers to select

        Returns
            csv file containing start and end node ids of each link in the selected layers.
            node ids are continuous and start from 0
        
        '''    
        # super().__init__(**kwargs) # inherite parent class's method        
        vars = locals()
        self.__dict__.update(vars)
        del self.__dict__["self"] 

    def main(self):        
        self.load_data()
        self.correct_node_id()
        self.put_small_id_to_start()
        self.remove_dupl()

        
        if self.is_get_net_charac:
            self.gen_net()
            self.get_layer()
            self.get_net_charac()
        
        if self.is_save_data:
            self.save_df() 
        
        if self.is_plot:
            self.gen_net()
            self.plot_layer()            
    
    def load_data(self):
        self.link_df_orig = pd.read_csv(self.path) 
        self.layer_id_list = self.link_df_orig['Relation'].unique().tolist()
        self.n_layer = len(self.layer_id_list)
  
    def correct_node_id(self):
        '''correct node id because some ids are skipped 
        '''
        link_df_temp = deepcopy(self.link_df_orig)
        # let id start from 0
        node_id_min = min(list(link_df_temp.loc[:,['From', 'To']].min()))
        if node_id_min >= 1:
            link_df_temp[['From', 'To']] =  link_df_temp[['From', 'To']] - node_id_min 
        # find missing node ids
        node_id = pd.concat([link_df_temp['From'], link_df_temp['To']], ignore_index=True).unique().tolist()
        node_id_missed = [i for i in range(max(node_id)) if i not in node_id]
        print('\n--- Skipped node id in the raw data: ', node_id_missed)
        # correct id error in df
        link_df_temp_new = link_df_temp
        for col in ['From', 'To']:
            node_list_temp = link_df_temp[col].tolist()
            counter = 0
            for this_id in node_id_missed:                          
                this_id -= counter
                node_list_temp = [ele - 1 if ele >= this_id else ele for ele in node_list_temp]                
                counter += 1
            link_df_temp_new[col] = pd.Series(node_list_temp, index=link_df_temp_new.index)
        node_id_new = pd.concat([link_df_temp_new['From'], link_df_temp_new['To']],
                                ignore_index=True).unique().tolist()          
        self.node_id = node_id_new
        self.n_node = len(node_id_new)
        self.link_df = link_df_temp_new
               
    def put_small_id_to_start(self):
        small_ids = self.link_df[['From', 'To']].min(axis=1) 
        large_ids = self.link_df[['From', 'To']].max(axis=1) 
        self.link_df['From'] = small_ids
        self.link_df['To'] = large_ids    
    
    def remove_dupl(self):
        self.link_df = self.link_df.drop_duplicates()
        
    def save_df(self):
        self.link_df.to_csv('../data/mafia_net_{}layers_{}nodes.csv'. \
                            format(self.n_layer, self.n_node), index=False)                 
            
    def gen_net(self):        
        G = nx.MultiGraph()
        for _, row in self.link_df.iterrows():
            G.add_edge(row['From'], row['To'], label=row['Relation'])       
        self.G = G

    def get_layer(self):
        self.G_layer_list = []
        for i in range(self.n_layer):
            G_sub = nx.Graph()
            selected_edges = [(u,v) for (u,v,e) in self.G.edges(data=True) \
                              if e['label']==self.layer_id_list[i]]
            G_sub.add_edges_from(selected_edges)
            self.G_layer_list.append(G_sub)
        
    def get_net_charac(self):
        print('\n--- Aggregate network')
        get_net_charac_sub(self.G)
        
        print('\n--- Each layer')
        for i in range(self.n_layer):
            print('\n------ {} layer'.format(self.layer_id_list[i]))
            get_net_charac_sub(self.G_layer_list[i])
                       
    def plot_layer(self): 
        # TODO: use color to represent layers or groups, or nodes in multiple layers
        # different node size
        font_size = 13
        fig, axes = plt.subplot_mosaic([['A', 'B']], figsize=(4*1.9, 4/3*1.9), dpi=400)        
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
            plt.savefig('../output/mafia_net_{}layers_{}nodes_each_layer.pdf'. \
                        format(self.n_layer, self.n_node), dpi=600)
        plt.show()
        
        plt.figure(figsize=(6,4), dpi=300)
        nx.draw(self.G, node_size=2)
        if self.is_save_fig:
            plt.savefig('../output/mafia_net_{}layers_{}nodes_all_layer.pdf'. \
                        format(self.n_layer, self.n_node), dpi=600)
        plt.show()
        
def main():
    path='../data/sicilian_mafia/edge_list_raw.csv'
    net = Net(path=path, is_save_data=True, is_get_net_charac=True)
    net.main()

if __name__ == '__main__':
    main()
