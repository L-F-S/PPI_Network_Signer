# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: L-F-S
make edges list to create features for.
For big files to be split in multiple subfiles to run on different servers.
"""

import pickle
import collections
import itertools
import lxml.etree as etree
import os
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from random import seed, sample
from datetime import date
from preproc_utils import load_training_data
from glob_vars import SPECIES, TRAIN_DATA, PERT_MAP, HOME_DIR,  LBL_DIR,EDGES_DIR,\
    FT_DIR, SIGNAL_DIR, MOD_DIR, PRT_DIR, NET_DIR,  NET_FILE,\
    PROPAGATE_ALPHA, PROPAGATE_EPSILON ,PROPAGATE_ITERATIONS, DICT_DIR,\
    DICT_FILE, ANAT_VAL_DIR, N_JOBS
    

DIRECTED_DIR = HOME_DIR+"network_signing"+os.sep
MAIN_DATA_DIR = HOME_DIR+'Data'+os.sep+SPECIES+os.sep
OUTDIR=DIRECTED_DIR+'features_and_labels'+os.sep+SPECIES+os.sep

with open(NET_DIR+NET_FILE, 'rb') as f:
    graph=pickle.load(f)
signed_TRAIN_DATA_labels, _ = load_training_data(LBL_DIR, TRAIN_DATA, SPECIES)
edges=list(set(graph.edges())-set(pd.concat(list(signed_TRAIN_DATA_labels.values())).index))
#%%  dividi in 6
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
