# -*- coding: utf-8 -*-
"""

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling

# https://github.com/pyg-team/pytorch_geometric

# https://medium.com/stanford-cs224w/gnn-based-link-prediction-in-drug-drug-interaction-networks-c0e2136e4a72
# https://medium.com/stanford-cs224w/online-link-prediction-with-graph-neural-networks-46c1054f2aa4


'''
error in installing pytorch_geometric
'''
# (torch) c:\code\illicit_net_resil\src>conda install -c conda-forge pytorch_geometric
# Collecting package metadata (current_repodata.json): done
# Solving environment: failed with initial frozen solve. Retrying with flexible solve.
# Solving environment: failed with repodata from current_repodata.json, will retry with next repodata source.
# Collecting package metadata (repodata.json): done
# Solving environment: failed with initial frozen solve. Retrying with flexible solve.
# Solving environment: -
# Found conflicts! Looking for incompatible packages.
# This can take several minutes.  Press CTRL-C to abort.
# failed

# UnsatisfiableError: The following specifications were found to be incompatible with each other:

# Output in format: Requested package -> Available versions