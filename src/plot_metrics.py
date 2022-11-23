# -*- coding: utf-8 -*-
"""
plot values of each metrics using EMA, EM, and random model
plot convergence of EMA and EM
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from string import ascii_uppercase
import numpy as np
from sklearn.metrics import auc

from my_utils import load_object, plotfuncs


class Plots:
    # class variables
    _colors = ['tab:{}'.format(x) for x in ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown','cyan','olive']]
    _markers = ['o', 'v', 's', '*', 'x', '<','d', 'o', 'v', '>', '*', 's', 'd','x']
    _LW = 1.8
    _MS = 11
    linestyles = plotfuncs.get_linestyles()       
    
    def __init__(self, net_name, n_layer, n_node_total,
                 frac_list, mtc_list, model_list, is_save_fig=True,
                 **kwargs):
        super().__init__(**kwargs) # inherite parent class's method if necessary       
        vars = locals()
        self.__dict__.update(vars)
        del self.__dict__["self"] 
    
    def main(self):
        self.load_data()
        self.plot_each_mtc()

    def load_data(self):       
        path_mean = '../output/raw_result/{}_{}layers_{}nodes_metric_mean_by_mdl_frac.pickle'.format(
                    self.net_name, self.n_layer, self.n_node_total)
        self.mtc_mean_by_mdl_frac = load_object(path_mean)
        self.mtc_std_by_mdl_frac = None 
        # std can be plotted if necessary
        # path_std = '../output/raw_result/{}_{}layers_{}nodes_metric_std_by_mdl_frac.pickle'.format(
        #             net_name, n_layer, n_node_total)
        # self.mtc_std_by_mdl_frac = load_object(path_std)       
        

    def plot_adj_MAE(self):
        ix_MAE = self.mtc_list.index('MAE')
        frac_select = [0.2, 0.4, 0.6, 0.8]
        ix_frac_select = [self.frac_list.index(x) for x in frac_select]
        
        n_col, n_row = 2, 2
        fig, axes = plt.subplots(n_row, n_col, figsize=(4.3*n_col, 4.1*n_row))  
        axes_flat = axes.flat      
        plt.subplots_adjust(wspace=0.1, hspace=0.43)
        plt.subplots_adjust(left=0, right=0.995, top=0.995, bottom=0)
        # plot curves
        font_size = Plots._MS+11
        max_MAE = 0
        y_scale = 1e4
        for n, ax in enumerate(axes_flat):
            mean_temp = self.mtc_mean_by_mdl_frac[ix_MAE][0][ix_frac_select[n]]
            # std_temp = self.mtc_std_by_mdl_frac[ix_MAE][0][ix_frac_select[n]]
            x_temp = range(1, len(mean_temp) + 1)
            y_mean_temp = [i*y_scale for i in mean_temp]
            # y_UB_temp = [i*y_scale for i in mean_temp + std_temp]
            # y_LB_temp = [i*y_scale for i in mean_temp - std_temp]
            # ax.fill_between(x_temp, y_UB_temp, y_LB_temp,
            #                 alpha=0.5, color=Plots._colors[0])
            p1 = ax.plot(x_temp, y_mean_temp, color=Plots._colors[0], linewidth=2.5)#,
                         #linestyle=Plots.linestyles[1])
            # p2 = ax.fill(np.NaN, np.NaN, color=Plots._colors[0], alpha=0.5)
            max_MAE = max(max_MAE, max(y_mean_temp))
            
            mean_temp = self.mtc_mean_by_mdl_frac[ix_MAE][1][ix_frac_select[n]]
            # std_temp = self.mtc_std_by_mdl_frac[ix_MAE][1][ix_frac_select[n]]
            x_temp = range(1, len(mean_temp) + 1)
            y_mean_temp = [i*y_scale for i in mean_temp]
            # y_UB_temp = [i*y_scale for i in mean_temp + std_temp]
            # y_LB_temp = [i*y_scale for i in mean_temp - std_temp]
            # ax.fill_between(x_temp, y_UB_temp, y_LB_temp,
            #                 alpha=0.5, color=Plots._colors[1])
            p3 = ax.plot(x_temp, y_mean_temp, color=Plots._colors[1], linewidth=2.5,
                         linestyle='dashed')
            # p4 = ax.fill(np.NaN, np.NaN, color=Plots._colors[1], alpha=0.5)
            max_MAE = max(max_MAE, max(y_mean_temp))
                       
            # ax.legend([(p2[0], p1[0]), (p4[0], p3[0]), ],
            #           ['EMA (mean$\pm$std)', 'EM  (mean$\pm$std)'], 
            #           loc='upper right', fontsize=font_size-2)
            ax.legend(['EMA', 'EM'], loc='upper right', fontsize=font_size)
            ax.tick_params(axis='both', labelsize=font_size)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # if n in [0, 1]:
            #     ax.set_xlabel([], color='w')
            #     ax.set_xticklabels([], color='w')
            # else:
            ax.set_xlabel('Iteration', size=font_size)
            if n in [1, 3]:
                ax.set_ylabel([], color='w')
                ax.set_yticklabels([], color='w')
            else:
                ax.set_ylabel(r'$\epsilon$ $(\times 10^{-4})$', size=font_size)
            if n in [0, 2]:
                txt_pos_x = -0.004
            else:
                txt_pos_x = -0.004
            txt_pos_y = 1.08
            ax.text(txt_pos_x, txt_pos_y, ascii_uppercase[n] + r'. $c$ = {}'.format(frac_select[n]),
                    transform=ax.transAxes, size=font_size+2)            
        # max_MAE = np.max(np.array(self.mtc_mean_by_mdl_frac[ix_MAE][0:2]))
        plt.setp(axes, ylim=(0, min(5, max_MAE*1.02)))
        #save fig
        if self.is_save_fig:
            file_name = '../output/{}_{}layers_{}nodes_MAE'.format(
                self.net_name, self.n_layer, self.n_node_total)
            plt.savefig(file_name +'.pdf', dpi=500)
        plt.show()
 

    def set_y_max(self, mtc, y_max=None):
        '''set the same max of y axis in G-mean and MCC for the same multiplex when the
           no. of layers are different for easier visual comparison
        '''
        if self.net_name == 'london_transport':
            if mtc == 'G-mean':
                y_max = 0.3
            if mtc == 'MCC':
                y_max = 0.25
        if self.net_name == 'elegan':
            if mtc == 'G-mean':
                y_max = 0.7
            if mtc == 'MCC':
                y_max = 0.65
        if self.net_name == 'drug':
            if mtc == 'G-mean':
                y_max = 0.5
            if mtc == 'MCC':
                y_max = 0.4
        if self.net_name == 'mafia':
            y_max = 0.4
        return y_max 
    
    
    def plot_acc_mtc(self, mtc_mean_by_mdl, mtc, y_max):
        ''' plot each metric for accuracy 
        '''
        plotfuncs.format_fig(1.2)
        plt.figure(figsize=(4.8*0.93, 4*0.93), dpi=400)
        if mtc == 'LogH':
            model_list = self.model_list[:-1]
        else:
            model_list = self.model_list
        for i in range(len(model_list)):
            plt.plot(self.frac_list, mtc_mean_by_mdl[i], color=Plots._colors[i],
                     marker=Plots._markers[i], alpha=.85, ms=Plots._MS,
                     lw=Plots._LW, linestyle = '--', label=self.model_list[i])                    
        plt.xlim([min(self.frac_list)-0.04, 0.94])
        if y_max is not None:
            plt.ylim(top=y_max)
        plt.xticks(np.arange(0.1, 1.1, 0.2))
        plt.xlabel(r'$c$', fontsize=Plots._MS+7)
        plt.ylabel(mtc, fontsize=Plots._MS+7)
        plt.legend(loc="best", fontsize=Plots._MS+3)
        if self.is_save_fig:
            file_name = '../output/{}_{}layers_{}nodes_{}'.format(
                self.net_name, self.n_layer, self.n_node_total, mtc)
            plt.savefig(file_name +'.pdf', dpi=500)
            if mtc in ['G-mean', 'MCC']:
                plt.savefig(file_name +'.png', dpi=500)
        plt.show() 

        
    def plot_each_mtc(self):        
        mtc_to_plot = ['G-mean', 'MCC', 'Recall', 'Precision', 'Accuracy',
                       'TN', 'FP', 'FN', 'TP', 'Log_H', 'IG_ratio', 
                       'KS distance', 'Hellinger distance','Run time (s)']
        for i_mtc, mtc in enumerate(self.mtc_list):
            if mtc in mtc_to_plot:
                print('\n')
                print('------ {}: {}'.format(self.mtc_list[i_mtc], self.mtc_mean_by_mdl_frac[i_mtc]))
                y_max = self.set_y_max(mtc)
                self.plot_acc_mtc(self.mtc_mean_by_mdl_frac[i_mtc], self.mtc_list[i_mtc], y_max) 
            if mtc == 'MAE':
                self.plot_adj_MAE()
                
                
    def plot_prc(self):
        ''' precision-recall curve for each fraction of observed nodes
        '''
        recall_list, precision_list = self.mtc_mean_by_mdl_frac[0], self.mtc_mean_by_mdl_frac[1]
        plotfuncs.format_fig(1.2)       
        n_frac = len(self.frac_list)
        for i_frac in range(n_frac):
            plt.figure(figsize=(5.5, 5.5*4/5), dpi=500)
            auc_list = []
            for i_mdl in range(len(self.model_list)):
                plt.plot(recall_list[i_mdl][i_frac], precision_list[i_mdl][i_frac],
                         color=Plots._colors[i_mdl], lw=Plots.lw, 
                         linestyle = Plots.linestyles[i_mdl][1], alpha=.85,
                         # label="{:.2f} ({:0.2f})".format(frac_list[idx], auc_list[idx]))
                         label=self.model_list[i_mdl]) 
                auc_list.append(auc(recall_list[i_mdl][i_frac], precision_list[i_mdl][i_frac]))
            plt.title('auc = {}'.format(auc_list))
            # plt.plot([0, 1], [0, 1], "k--", lw=lw)
            # baseline is random classifier. P/ (P+N)
            # https://stats.stackexchange.com/questions/251175/what-is-baseline-in-precision-recall-curve
            plt.xlim([min(self.frac_list)-0.03, 1.03])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            # plt.xticks(np.linspace(0, 1, num=6, endpoint=True))
            plt.legend(loc="lower right", fontsize=13) #, title=r'$c$') #title=r'$c$  (AUC)')
            # ax = plt.gca()
            # handles, labels = ax.get_legend_handles_labels()
            # ax.legend(reversed(handles), reversed(labels), title=r'$c$', loc='lower left', fontsize=14.5,)        
            plt.savefig('../output/{}_prc_frac{}_{}layers_{}nodes.pdf'.format(
                        self.net_name, self.frac_list[i_frac], self.n_layer, self.n_node_total))
            plt.savefig('../output/{}_prc_frac{}_{}layers_{}nodes.png'.format(
                        self.net_name, self.frac_list[i_frac], self.n_layer, self.n_node_total))
            plt.show() 


def main():
    from multi_net import frac_list, mtc_list, model_list

    # # london transport
    # net_name = 'london_transport'
    # n_node_total_list = [356, 318]
    # n_layer_list = [3, 2]
    # for i, _ in enumerate(n_node_total_list):
    #     plots = Plots(net_name, n_layer_list[i], n_node_total_list[i], frac_list, mtc_list, model_list)
    #     plots.main()

    # # elegans
    # net_name = 'elegan'
    # n_node_total_list = [279, 273]
    # n_layer_list = [3, 2]
    # for i, _ in enumerate(n_node_total_list):
    #     plots = Plots(net_name, n_layer_list[i], n_node_total_list[i], frac_list, mtc_list, model_list)
    #     plots.main()

    # drug trafficking
    net_name = 'drug'
    n_node_total_list = [2196, 2139, 2114]
    n_layer_list = [4, 3, 2]
    for i, _ in enumerate(n_node_total_list):
        plots = Plots(net_name, n_layer_list[i], n_node_total_list[i], frac_list, mtc_list, model_list)
        plots.main()

    # # mafia
    # net_name = 'mafia'
    # n_node_total_list = [143]
    # n_layer_list = [2]
    # for i, _ in enumerate(n_node_total_list):
    #     plots = Plots(net_name, n_layer_list[i], n_node_total_list[i], frac_list, mtc_list, model_list)
    #     plots.main()
if __name__ == "__main__":
    main()