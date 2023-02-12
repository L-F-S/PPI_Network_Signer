# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
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

SPECIES='H_sapiens'
# SPECIES = "S_cerevisiae"

HOME_DIR='G:'+os.sep+'Il Mio Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep
HOME_DIR='G:'+os.sep+'My Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep
DIRECTED_DIR = HOME_DIR+"network_signing"+os.sep
MAIN_DATA_DIR = HOME_DIR+'Data'+os.sep+SPECIES+os.sep
OUTDIR=DIRECTED_DIR+'features_and_labels'+os.sep+SPECIES+os.sep

with open( MAIN_DATA_DIR+'alias_2geneid.pkl', 'rb') as f:
    alias_2geneid = pickle.load(f)

network=pd.read_csv(MAIN_DATA_DIR +SPECIES+'.net',sep="\t", header=None)
graph = nx.from_pandas_edgelist(network.reset_index(), 0, 1, 2)
datasets = ['patkar_kegg','kpi', 'ubiq'] if SPECIES == 'S_cerevisiae' else ['kegg','PSP','depod','ubinet2']
signed_datasets_labels, _ = load_training_data(OUTDIR, datasets, SPECIES)
#%%
edges=list(set(graph.edges())-set(pd.concat(list(signed_datasets_labels.values())).index))
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
    with open(OUTDIR+'netedges'+str(i+1), 'wb') as f:
        pickle.dump(edges[i:j],f)
    i=j