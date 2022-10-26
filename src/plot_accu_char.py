# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:21:22 2022

@author: Jinzh
"""
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_utils import plotfuncs
import os
os.chdir('c:/code/illicit_net_resil/src')

def plot_gmean_obs(df, net_name):
    df[df['net_name']==net_name].groupby('n_layer')['G-mean'].plot(legend=True, figsize=(4.4, 3.3))
    plt.xlabel('Observed fraction')
    plt.ylabel('G-mean')
    plt.legend(title= 'No. of layers')
    plt.xlim([0,1])
    plt.show()

def main():
    path = '../output/log2H_by_obs_frac.csv'
    df = pd.read_csv(path)
    df.set_index('obs_frac', inplace=True)
    net_name_list = df['net_name'].unique().tolist() #['drug', 'power', 'rand']
    plotfuncs.format_fig(0.85)
    for net_name in net_name_list:
        plt.title(net_name + ' network')
        plot_gmean_obs(df, net_name)    

if __name__ == '__main__':
    main()




