# -*- coding: utf-8 -*-
"""

"""

import numpy as np
from matrix_completion import svt_solve
# conda install -c conda-forge cvxpy
# pip install matrix-completion
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt


def plot_roc(fpr, tpr, roc_auc):          
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
def NMF(m, n, cut_off=0.5, std=1):
    
    R = np.abs(np.random.normal(0, std, (m,n)).astype(float))
    R[R>1] = 1
    R = np.round(R)
    
    mask = np.random.rand(m, n)
    mask[mask<cut_off] = 0
    mask[mask>=cut_off] = 1    
    unobs_idx = np.where(mask==0)

    R_hat = svt_solve(R, mask, max_iterations=100)
    fpr, tpr, threshold = roc_curve(R[unobs_idx], R_hat[unobs_idx])
    roc_auc = roc_auc_score(R[unobs_idx], R_hat[unobs_idx])
    f1_score_val = f1_score(R[unobs_idx], np.round(R_hat[unobs_idx]))
    print("f1: ", f1_score_val)
    plot_roc(fpr, tpr, roc_auc)

NMF(m=100, n=90, cut_off=0.2, std=2)
