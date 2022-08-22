# -*- coding: utf-8 -*-
"""
Formate the Sicilian Mafia network data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import networkx as nx
from copy import deepcopy

class Net():
    
    def __init__(self, path, net_type, 
                 is_save_data=False, is_get_net_charac=False, 
                 is_plot=False, is_save_fig=False,
                 **kwargs):
        '''     
        Parameters
            Path: path to the links of networks.

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
        self.remove_dupl()

        
        if self.is_get_net_charac:
            self.gen_net()
            self.get_layer()
            self.get_net_charac()
        
        if self.is_save_data:
            self.save_df() 
        
        if self.is_plot:
            self.gen_net()         
    
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
        self.node_id_missed = node_id_missed
        self.n_node = len(node_id_new)
        self.link_df_new = link_df_temp_new
               
        # put_small_id_to_start. links are undirected.
        small_ids = self.link_df_new[['From', 'To']].min(axis=1) 
        large_ids = self.link_df_new[['From', 'To']].max(axis=1) 
        self.link_df_new['From'] = small_ids
        self.link_df_new['To'] = large_ids    
    
    def remove_dupl(self):
        self.link_df_new = self.link_df_new.drop_duplicates()
        
    def save_df(self):
        self.link_df_new.to_csv('../data/mafia_net_{}layers_{}nodes.csv'. \
                            format(self.n_layer, self.n_node), index=False)                 
            
    def gen_net(self):        
        self.G_agg = nx.MultiGraph()
        for _, row in self.link_df_new.iterrows():
            self.G_agg.add_edge(row['From'], row['To'], label=row['Relation'])       

    def get_layer(self):
        self.G_layer_list = []
        for i in range(self.n_layer):
            G_sub = nx.Graph()
            selected_edges = [(u,v) for (u,v,e) in self.G_agg.edges(data=True) \
                              if e['label']==self.layer_id_list[i]]
            G_sub.add_edges_from(selected_edges)
            self.G_layer_list.append(G_sub)

    def get_net_charac_sub(self, G):
        ''' get characteristics of each layer or the aggregate layer of a multilayer network
        '''
        n_node = G.number_of_nodes()
        n_link = G.number_of_edges()
        
        density = nx.density(G)
        degree_mean = 2*n_node / float(n_link)
        cc_size = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        cc_mean = np.mean(cc_size)
        
        n_digit = 2
        print('--------- No. of nodes: ', n_node)
        print('--------- No. of links: ', n_link)
        print('--------- Density/1000: ', round(density*1000, n_digit))
        print('--------- Average degree: ', round(degree_mean, n_digit))
        print('--------- Average size of connected component: ', round(cc_mean, n_digit))
        print('--------- Size of the greatest connected component: ', round(max(cc_size), n_digit))
        print('--------- CoV of connected components: ', round(np.std(cc_size)/cc_mean, n_digit))  
        
    def get_net_charac(self):
        print('\n--- Aggregate network')
        self.get_net_charac_sub(self.G_agg)
        
        print('\n--- Each layer')
        for i in range(self.n_layer):
            print('\n------ {} layer'.format(self.layer_id_list[i]))
            self.get_net_charac_sub(self.G_layer_list[i])
                       
    # def plot_layer(self): 
    #     # TODO: use color to repret layers or groups, or nodes in multiple layers
    #     # different node size
    #     font_size = 13
    #     fig, axes = plt.subplot_mosaic([['A', 'B']], figsize=(4*1.9, 4/3*1.9), dpi=400)        
    #     for label, ax in axes.items():
    #         # label physical distance to the left and up:
    #         trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    #         if label in ['B', 'D']:
    #             x_pos = 0.13
    #         else:
    #             x_pos = 0.13
    #         ax.text(x_pos, 0.93, label, transform=ax.transAxes + trans,
    #                 fontsize=font_size, va='bottom')
    #     fig.subplots_adjust(hspace=0.28)  
    #     fig.subplots_adjust(wspace=0.19)
    #     # i_lyr = 0
    #     ax_labels = [ele[0] for ele in list(axes.items())]
    #     for idx in range(len(ax_labels)):
    #         nx.draw(self.G_layer_list[idx], node_size=2)
    #         # print(layer_selected[counter], ': ', len(selected_edges))
    #         plt.sca(axes[ax_labels[idx]])
    #         nx.draw(self.G_layer_list[idx], node_size=2)
    #         # i_lyr += 1
    #     # save fig
    #     plt.tight_layout()
    #     if self.is_save_fig:
    #         # plt.savefig('../output/each_layer_net.pdf', dpi=800)
    #         plt.savefig('../output/{}_net_{}layers_{}nodes_each_layer.pdf'. \
    #                     format(self.net_type, self.n_layer, self.n_node), dpi=600)
    #     plt.show()
        
    #     plt.figure(figsize=(6,4), dpi=300)
    #     nx.draw(self.G, node_size=2)
    #     if self.is_save_fig:
    #         plt.savefig('../output/{}_net_{}layers_{}nodes_all_layer.pdf'. \
    #                     format(self.net_type, self.n_layer, self.n_node), dpi=600)
    #     plt.show()


# drug trafficking network
class DrugNet(Net):    
    def __init__(self, layer_id_list, **kwargs):
        '''     
        Parameters
            Path: path to the links of networks.
            layer_selected_list: list of layers to select

        Returns
            csv file containing start and end node ids of each link in the selected layers.
            node ids are continuous and start from 0
        
        '''    
        super().__init__(**kwargs) # inherite parent class's method        
        vars = locals()
        self.__dict__.update(vars)
        del self.__dict__["self"] 

    def main(self):        
        self.load_data()
        self.merge_layer()
        self.select_layers()        
        self.correct_node_id()

        self.select_attr()
        
        if self.is_get_net_charac:
            self.gen_net()
            self.get_layer()
            self.get_net_charac()
        
        if self.is_save_data:
            self.save_df() 
        
        if self.is_plot:
            self.gen_net()          
            
    def load_data(self):
        self.n_layer = len(self.layer_id_list)
        self.link_df_orig = pd.read_excel(self.path, sheet_name='LINKS IND AND GROUP')
        self.attr_df_orig = pd.read_excel(self.path, sheet_name='ACTOR ATTRIBUTES')
        # rename columns
        self.link_df_orig.rename(
            columns={'Actor_A': 'From', 'Actor_B': 'To', 'Type_relation': 'Relation'} , 
            inplace=True) 
        self.attr_df_orig.rename(columns={'Actor_ID': 'Node_ID'}, inplace=True) 
        
    def merge_layer(self):
        '''merge 'Associate/Friendship' into 'Legitimate'
           merge 'Negative/Enemies' into 'Formal Criminal Organization' 
        '''      
        self.link_df_orig.loc[self.link_df_orig['Relation']=='Associate/Friendship', 'Relation'] = 'Legitimate'
        self.link_df_orig.loc[self.link_df_orig['Relation']=='Negative/Enemies', 'Relation'] = 'Formal Criminal Organization'
        # remove duplicates
        self.link_df_orig = self.link_df_orig.drop_duplicates(['From', 'To', 'Relation'], keep='last')

    def select_layers(self): 
        '''selected layers from 4 layers
                4 layers: ['Co-Offenders', 'Kinship', 'Formal Criminal Organization', 'Legitimate']
                3 layers: ['Co-Offenders', 'Formal Criminal Organization', 'Legitimate']
                2 layers: ['Co-Offenders', 'Legitimate']
        '''
        self.link_df_orig = self.link_df_orig.loc[self.link_df_orig['Relation'].isin(self.layer_id_list)]        
    
    def correct_node_id(self):
        '''correct node id because some ids are skipped in the raw data
        '''
        super().correct_node_id()        
        self.attr_df = self.attr_df_orig[~self.attr_df_orig['Node_ID'].isin(self.node_id_missed)].reset_index() 
        self.attr_df['Actor_ID'] = pd.Series(list(range(len(self.attr_df. index))), index=self.attr_df.index)      
        
    def select_attr(self):
        self.attr_df = self.attr_df[['Node_ID', 'Gender', 'DOB_Year', 'Age_First_Analysis',
                                     'Drug_Activity', 'Recode_Level', 'Drug_Type',
                                     'Group_Membership_Type', 'Group_Membership_Code']]
        
    def impute_miss_data(self):
        # if an actor is not involved in drug activity
        self.attr_df.loc[self.attr_df['Drug_Activity']==0, ['Recode_Level', 'Drug_Type']] = 'NONE'
        for col in ['Gender', 'DOB_Year', 'Age_First_Analysis']:
            self.attr_df.loc[self.attr_df[col]=='.', col] = np.nan
        
        self.attr_df.loc[self.attr_df['Gender']=='Body', 'Gender'] = np.nan
        self.attr_df.loc[self.attr_df['Gender']=='Male', 'Gender'] = 1
        self.attr_df.loc[self.attr_df['Gender']=='Female', 'Gender'] = 0
        
        cols_to_factor = ['Recode_Level', 'Drug_Type', 'Group_Membership_Type', 'Drug_Activity']
        self.attr_df[cols_to_factor] = self.attr_df[cols_to_factor].apply(lambda x: pd.factorize(x)[0])
        
        self.attr_df['Group_Membership_Code'] = self.attr_df['Group_Membership_Code'].astype('float')
        
        cols_with_NA = ['Gender', 'DOB_Year', 'Age_First_Analysis'] #[['Gender', 'DOB_Year', 'Age_First_Analysis'], ['DOB_Year', 'Age_First_Analysis']]
        id_col = self.attr_df.loc[:, 'Node_ID']
        self.attr_df = self.attr_df.drop(['Node_ID'], axis=1)
        df_without_NA = self.attr_df.dropna()
        is_NA_idx = self.attr_df[cols_with_NA].isna().any(axis=1)
        df_with_NA_only = self.attr_df[is_NA_idx]
       
        # transform to float
        df_without_NA[cols_with_NA] = df_without_NA[cols_with_NA].astype('float')

        cols_no_NA = [ele for ele in df_without_NA.columns if ele not in cols_with_NA]
        cols_with_NA_new = [ele+'_New' for ele in cols_with_NA]
        X_train = df_without_NA.loc[:, cols_no_NA]
        X_test = df_with_NA_only.loc[:, cols_no_NA]    
        for idx, col in enumerate(cols_with_NA):
            print('--- Impute: ', col)
            y_train = df_without_NA[col]
                    
            # fit model on training data
            model = DecisionTreeRegressor(max_depth=8)
            model.fit(X_train, y_train)
            
            # make predictions for test data
            y_pred = model.predict(X_test)
            self.attr_df.loc[is_NA_idx, cols_with_NA_new[idx]] = y_pred
        
        for idx, col in enumerate(cols_with_NA):
            is_na_idx = self.attr_df[col].isna()
            self.attr_df.loc[is_na_idx, col] = self.attr_df.loc[is_na_idx, cols_with_NA_new[idx]].round(0)
           
        # remove cols_with_NA_new
        self.attr_df = self.attr_df.drop(cols_with_NA_new, axis=1)
        # add node id back
        self.attr_df['Node_ID'] = id_col
        cols = list(self.attr_df.columns)
        cols = [cols[-1]] + cols[:-1]
        self.attr_df = self.attr_df[cols]

def main_drug():
    from sklearn.tree import DecisionTreeRegressor
    path='../data/drug_net_raw_data.xlsx'
    net_type = 'drug'
    is_get_net_charac = True
    if is_get_net_charac:
        layer_selected = [['Co-Offenders', 'Legitimate', 'Formal Criminal Organization', 'Kinship']]    
    else:
        layer_selected = [['Co-Offenders', 'Kinship', 'Formal Criminal Organization', 'Legitimate'],
                          ['Co-Offenders', 'Formal Criminal Organization', 'Legitimate'],
                          ['Co-Offenders', 'Legitimate']]
    for layer_list in layer_selected:
        print('\n=== {} layers ==='.format(len(layer_list)))
        drug_net = DrugNet(path=path, net_type=net_type, layer_id_list=layer_list, is_save_data=True,
                           is_get_net_charac=is_get_net_charac)
        drug_net.main()

if __name__ == '__main__':
    main_drug()

# mafia network        
def main():
    path='../data/sicilian_mafia/edge_list_raw.csv'
    net_type = 'mafia'
    mafia = Net(path=path, net_type=net_type, is_save_data=True, is_get_net_charac=True)
    mafia.main()

# if __name__ == '__main__':
#     main()
