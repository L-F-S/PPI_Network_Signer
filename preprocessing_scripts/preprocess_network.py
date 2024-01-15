# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:38:44 2023

@author: los4

created to run once to potentially add training label data to 
base network. to run before SIGNAL feature generation
"""
import os
import argparse
import pickle
import pandas as pd
import networkx
from time import time
from preproc_utils import add_edges_from_labels, read_network_from_file,graph_from_dataframe,\
    load_training_data
from score_edges import generate_similarity_matrix, create_the_features_different_knockouts_iterative
import networkx as nx

###############################################################################
#   INPUTS
###############################################################################
SPECIES ='S_cerevisiae'
datasets = ['patkar_kegg','kpi', 'ubiq'] if SPECIES == 'S_cerevisiae' else ['kegg','PSP','depod','ubinet2']

print(datasets)
print(SPECIES)

# # Servier directories:
# HOME_DIR  =  '/home/bnet/lorenzos/signed/signedv3/'
# INPUT_DIR=HOME_DIR+"input"+os.sep+SPECIES+os.sep
# OUTDIR=HOME_DIR+'output'+os.sep+SPECIES+os.sep+'validation_out'
# FEATURESDIR=OUTDIR
# BASENETDIR=INPUT_DIR

# Local directories
HOME_DIR  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# HOME_DIR  =  "G:" +os.sep+"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
SIGNAL_DIR = HOME_DIR+"network_signing"+os.sep
os.chdir(SIGNAL_DIR)
DATA_DIR = HOME_DIR+'Data'+os.sep+SPECIES+os.sep
INPUT_DIR=SIGNAL_DIR+'features_and_labels'+os.sep+SPECIES+os.sep
networkdir=SIGNAL_DIR+'input'+os.sep+SPECIES+os.sep+'network'+os.sep

BASENETDIR=DATA_DIR

# Propagation parameters:
PROPAGATE_ALPHA = 0.6
PROPAGATE_EPSILON = 1e-5
PROPAGATE_ITERATIONS = 100

###############################################################################
#     Load training sets:
###############################################################################
print("\n>>>>>>> loading signed edges datasets:")
signed_datasets_labels, signed_datasets_edge_weights = load_training_data(INPUT_DIR, datasets, SPECIES)
###############################################################################
#     Load PPI network:
###############################################################################
print("\n>>>>>>> loading base network to propagte on:")
graph = graph_from_dataframe(BASENETDIR, SPECIES)
#ADD edges from training labels which are not present:
#%%
print("\n>>>>>>> adding missing edges from signed data to base network:")
for dataname, signed_data in signed_datasets_edge_weights.items():
    graph = add_edges_from_labels(graph, signed_data, v=False)

#%%
datasetstr='_'.join(datasets)
with open(networkdir+SPECIES+'_'+datasetstr+'.nx', 'wb') as f:
    pickle.dump(graph,f)
