# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:42:52 2024

Generate SIGNAL scores
for base network, using yeast data, and all trainign data
WARNING: make sure input data in  glob_vars.py are the desired inputs

WARNING: SLOW! not recommended to run in iterative mode
"""

import os
import sys

from glob_vars import SPECIES, HOME_DIR, TRAIN_DATA, PERT_MAP, LBL_DIR, EDGES_DIR,\
    FT_DIR, SIGNAL_DIR, MOD_DIR, PRT_DIR, NET_DIR,  NET_FILE,\
    PROPAGATE_ALPHA, PROPAGATE_EPSILON ,PROPAGATE_ITERATIONS, DICT_DIR,\
    DICT_FILE, ANAT_VAL_DIR, N_JOBS
import pickle
import pandas as pd
import networkx as nx
import numpy as np
import subprocess

def load_net_edges(chunk):
    filename='net'+str(chunk)+'.edges'
    with open(EDGES_DIR+filename,'rb') as f:
        edges=pickle.load(f)
    return edges,filename
    

def generate_SIGNAL_features( edges_filename):
    # Define the command to execute the second script with arguments
    print('GENERATING SIGNAL FEATURES')
    command = ['python', HOME_DIR+'SIGNAL_ft_gen_PARALLEL.py',
                '-e', edges_filename, '-s', SPECIES, '-p', PERT_MAP]
    
    # Execute the command and capture the output
    output = subprocess.check_output(command, universal_newlines=True)
    
    # Print the output
    print(output) # output
    return

def apply_SIGNAL(features_filename):
    print('CALCULATING SIGNAL SCORE')
    command = ['python', HOME_DIR+'applySIGNAL.py',
               '-f', features_filename, '-s', SPECIES, '-p', PERT_MAP]
    output = subprocess.check_output(command, universal_newlines=True)
    return

def print_SIGNAL( SIGNALscores_filename):
    print('SIGNAL SCORE OF EDGES FROM ANCHOR TO TERMINAL:')
    data=pd.read_csv(SIGNAL_DIR+SIGNALscores_filename,sep=' ', header=None, index_col=([0,1]))
    print(data)

#%%   

# Load edges / generate edges
netname=6#int(sys.argv[1])
edges, edges_filename = load_net_edges(netname)
edges=edges[0:16]

# # GENERATE FEATURES FOR SIGNAL:
features_filename=edges_filename+'_'+PERT_MAP+'.ft'
if not os.path.isfile(FT_DIR+features_filename):
    generate_SIGNAL_features(edges_filename)

# Apply signal and predict SIGNAL score
SIGNAL_scores_filename=edges_filename+'_'+PERT_MAP+'.sgnl'
if not os.path.isfile(SIGNAL_DIR+SIGNAL_scores_filename):
    apply_SIGNAL(features_filename)
print('SIGNAL scores saved to:',SIGNAL_scores_filename)

# Print the results
print_SIGNAL(SIGNAL_scores_filename)

#%%
