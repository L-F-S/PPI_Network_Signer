# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:14:18 2023

@author: los4
"""

# -*- coding: utf-8 -*-
"""
Reformats ANAT network output

python 2SIGNALscorenetowrks.py filename

applies SIGNAL to edges of anchor-terminal subnetworks

V2 FIRST APPLY SIGNAL TO THE ANAT EDGES FILE AND AFTER ONLY AFTER FIND SUBPATHS!
"""

import pickle
import os
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import subprocess
import sys
import random
from preproc_utils import load_training_data, graph_from_dataframe
from score_edges import generate_similarity_matrix, create_the_features_different_knockouts_iterative
from time import time
from glob_vars import SPECIES, TRAIN_DATA, PERT_MAP, LBL_DIR,EDGES_DIR,\
    FT_DIR, SIGNAL_DIR, MOD_DIR, PRT_DIR, NET_DIR, HOME_DIR, NET_FILE

PERT_MAP +='parallelrefactored'

# LOAD inputs 
with open(NET_DIR+NET_FILE, 'rb') as f:
    graph=pickle.load(f)

def get_random_edges(graph, n):
    edges_filename='random'+str(n)+'.edges'
    random_edges=[list(graph.edges)[random.randint(0, len(graph.edges)-1)] for x in range(n)] 
    with open(EDGES_DIR+edges_filename, 'wb') as f:
        pickle.dump(random_edges,f)
    
    return edges_filename

def generate_SIGNAL_features( edges_filename):
    # Define the command to execute the second script with arguments
    print('GENERATING SIGNAL FEATURES')
    command = ['python', HOME_DIR+'network_signing'+os.sep+'SIGNAL_ft_gen_iterative.py',
                '-e', edges_filename, '-s', SPECIES, '-p', PERT_MAP,
                '-o', FT_DIR, '-ed', EDGES_DIR+os.sep]
    
    # Execute the command and capture the output
    output = subprocess.check_output(command, universal_newlines=True)
    
    # Print the output
    print(output) # output
    return

def generate_SIGNAL_features2(graph, edges_filename):
    # Define the command to execute the second script with arguments
    print('GENERATING SIGNAL FEATURES2')
    # LOAD SIGNAL FEATURES INPUTS
    
    #load network
    #fatto tuori
  
    
    # load KO dictionaries
    with open(PRT_DIR+'plus_targets_'+PERT_MAP+'.pkl', 'rb') as f:
        plus_targets_of_deletion = pickle.load(f)
    with open(PRT_DIR+'minus_targets_'+PERT_MAP+'.pkl', 'rb') as f:
        minus_targets_of_deletion = pickle.load(f)
    
    # Load edges
    with open(EDGES_DIR+edges_filename,'rb') as f:
        edges=pickle.load(f)
    
    # 'Generate similarity matrix.'
    genes = sorted(graph.nodes)
    raw_matrix = nx.to_scipy_sparse_matrix(graph, genes, format='csc').T
    matrix, raw_col_sums = generate_similarity_matrix(raw_matrix) #normalized similarity matrix
    num_genes     = len(genes)
    gene_indexes  = dict([(gene, index) for (index, gene) in enumerate(genes)]) #all genes present in matrix
    
    print('Generating features...')
    # Propagation parameters:
    PROPAGATE_ALPHA = 0.6
    PROPAGATE_EPSILON = 1e-5
    PROPAGATE_ITERATIONS = 100

        
    start=time()
    knockout_names, results = create_the_features_different_knockouts_iterative(raw_matrix, edges, gene_indexes, matrix, plus_targets_of_deletion,\
                          minus_targets_of_deletion,  num_genes, PROPAGATE_ALPHA, \
                              PROPAGATE_ITERATIONS,PROPAGATE_EPSILON )

    print("time passed", time()-start)
    data = pd.DataFrame().from_records(results, index=[0,1], columns = knockout_names[0])
    # data.to_csv(OUTDIR+name+'_'+perturbations_name+'.ft.csv')
    with open(FT_DIR+edges_filename+'_'+PERT_MAP+'.ft','wb') as f:
        pickle.dump(data, f)
        
    return

def apply_SIGNAL(features_filename):
    print('CALCULATING SIGNAL SCORE')
    command = ['python', HOME_DIR+os.sep+'applySIGNAL.py',
               '-f', features_filename, '-s', SPECIES, '-p', PERT_MAP]
   
    output = subprocess.check_output(command, universal_newlines=True)
    print(output)
    return

def print_SIGNAL( SIGNALscores_filename):
    print('SIGNAL SCORE OF EDGES FROM ANCHOR TO TERMINAL:')
    data=pd.read_csv(SIGNAL_DIR+SIGNALscores_filename,sep=' ', header=None, index_col=([0,1]))
    print(data)
#%%
def main():
    
    # Generate random edges file
    # edges_filename = get_random_edges(graph, n=9)
    # generate edges file from kpi trainind data (TEST: should get high numbers)
    # edges_filename = get_training_edges()
    
    # GENERATE FEATURES FOR ALL EDGES...
    # edges_filename='S_cerevisiae.edges'
    # with open(EDGES_DIR+edges_filename, 'wb') as f:
    #     pickle.dump(list(graph.edges),f)
    
    edges_filename='patkar_kegg' #training_edges
    # # GENERATE FEATURES FOR SIGNAL:

    features_filename=edges_filename+'_'+PERT_MAP+'.ft'
    if not os.path.isfile(FT_DIR+features_filename):
        generate_SIGNAL_features2(graph, edges_filename) #brutta qsta chiamata  agraph che cmq e variabile globale e o cmq andrebbe loaddata dentro come gli altri input. cda cambiare
    
    # Apply signal and predict SIGNAL score
    SIGNAL_scores_filename=edges_filename+'_'+PERT_MAP+'.sgnl'
    if not os.path.isfile(SIGNAL_DIR+SIGNAL_scores_filename):
        apply_SIGNAL(features_filename)
    print('SIGNAL scores saved to:',SIGNAL_scores_filename)
    
    # Print the results
    print_SIGNAL(SIGNAL_scores_filename)

if __name__ == '__main__':
    main()
