# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: L-F-S

make edges list to create features for.
For big files to be split in multiple subfiles to run on different servers.
"""

import pickle
import os
import pandas as pd
from preproc_utils import load_training_data
from glob_vars import SPECIES, TRAIN_DATA, LBL_DIR, EDGES_DIR, NET_DIR, NET_FILE
    


with open(NET_DIR+NET_FILE, 'rb') as f:
    graph=pickle.load(f)
signed_TRAIN_DATA_labels, _ = load_training_data(LBL_DIR, TRAIN_DATA, SPECIES)
edges=list(set(graph.edges())-set(pd.concat(list(signed_TRAIN_DATA_labels.values())).index))
#%%  split in 6
splits=6
print(len(edges))
for x in range(splits):
    print(x+1)
    total = len(edges)
    if x==0:
        i = 0
    j = int(total/splits*(x+1))
    print(i,j)
    print(len(edges[i:j]))
    print('first', edges[i])
    print('last', edges[j-1])
    with open(EDGES_DIR+'net'+str(x+1)+'.edges', 'wb') as f:
        pickle.dump(edges[i:j],f)
    i=j
