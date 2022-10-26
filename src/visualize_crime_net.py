# -*- coding: utf-8 -*-
"""

"""
import os
os.chdir('c:\code\illicit_net_resil\src')

import sys
sys.path.append('../multinet_vis')
from pymnet import *

import numpy as np

mplex = MultiplexNetwork(couplings="categorical")

n = 10
for i in np.random.randint(0,n, n):
    for j in np.random.randint(0,n,n):
        if i != j:
            mplex[i,'a'][j,'a'] = 1
for i in np.random.randint(0,n,n):
    for j in np.random.randint(0,n,n):
        if i != j:
            mplex[i,'b'][j,'b'] = 1

draw(mplex, show=True, layout='spring',
     nodeSizeRule={"rule":"degree","propscale":0.05},
     edgeColorRule={"rule":"weight","colormap":"jet","scaleby":0.1})

mplex.A['a'][[1,2],[1,3]]

# fig = draw(mplex, 
#            layout='spring',
#            nodeSizeRule={"rule":"degree","propscale":0.1})
# fig.savefig("../output/mplex.pdf")



# fig=draw(er(10,3*[0.3]),
#              layout="circular",
#              nodeColorDict={(0,0):"r",(1,0):"r",(0,1):"r"},
#              layerLabelRule={},
#              nodeLabelRule={},
#              nodeSizeRule={"rule":"degree","propscale":0.05},
#              edgeColorRule={"rule":"edgeweight","colormap":"jet","scaleby":0.1})


