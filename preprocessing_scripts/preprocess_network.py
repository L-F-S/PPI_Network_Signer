# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:38:44 2023

@author: L-F-S

to run before SIGNAL feature generation
Run once to add training label data to 
base network (if needed) and
convert network into pickled binary 
"""
import pickle
from preproc_utils import add_edges_from_labels, graph_from_dataframe,\
    load_training_data
import networkx as nx
from glob_vars import SPECIES, TRAIN_DATA, LBL_DIR, NET_DIR

###############################################################################
#   INPUTS
###############################################################################

print(TRAIN_DATA)
print(SPECIES)

###############################################################################
#     Load training sets:
###############################################################################
print("\n>>>>>>> loading signed edges datasets:")
signed_datasets_labels, signed_datasets_edge_weights = load_training_data(LBL_DIR, TRAIN_DATA, SPECIES)
###############################################################################
#     Load PPI network:
###############################################################################
print("\n>>>>>>> loading base network to propagte on:")
graph = graph_from_dataframe(NET_DIR, SPECIES)
#ADD edges from training labels which are not present:
#%%
print("\n>>>>>>> adding missing edges from signed data to base network:")
for dataname, signed_data in signed_datasets_edge_weights.items():
    graph = add_edges_from_labels(graph, signed_data, v=False)

#%%
datasetstr='_'.join(TRAIN_DATA)
with open(NET_DIR+SPECIES+'_'+datasetstr+'2.nx', 'wb') as f:
    pickle.dump(graph,f)
