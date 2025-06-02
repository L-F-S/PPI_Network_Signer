# -*- coding: utf-8 -*-
"""
L-F-S

Generate SIGNAL scores
for ANAT validation  network,
using yeast data, and all trainign data
WARNING: make sure input data in  glob_vars.py are the desired inputs
"""

import os
import sys
DRIVE_PATH=  'G:'+os.sep+'My Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep
HOME_DIR  =  DRIVE_PATH+'network_signing'+os.sep
os.chdir(HOME_DIR)
sys.path.append(HOME_DIR)

from glob_vars import SPECIES, TRAIN_DATA, PERT_MAP, LBL_DIR,EDGES_DIR,\
    FT_DIR, SIGNAL_DIR, MOD_DIR, PRT_DIR, NET_DIR,  NET_FILE,\
    PROPAGATE_ALPHA, PROPAGATE_EPSILON ,PROPAGATE_ITERATIONS, DICT_DIR,\
    DICT_FILE, ANAT_VAL_DIR, N_JOBS
import pickle
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import subprocess
import random
from preproc_utils import load_training_data, graph_from_dataframe
from score_edges import generate_similarity_matrix, create_the_features_different_knockouts_iterative,\
    create_the_features_different_knockouts
from joblib import Parallel, delayed
from time import time


def generate_edges_from_anat(FILENAME,alias_2geneid,  write=True):
    ANAT_net_file=ANAT_VAL_DIR+FILENAME
    columns = ['ID1', 'ID2']
    # Read the CSV file and select the desired columns and rows
    edges_df = pd.read_csv(ANAT_net_file+'.csv', usecols=columns)
    edges=[(alias_2geneid[x], alias_2geneid[y]) for (x,y) in list((zip(edges_df['ID1'],edges_df['ID2'])))]
    edges_filename=FILENAME+'.edges'
    if write:
        with open(EDGES_DIR+edges_filename, 'wb') as f:
            pickle.dump(edges,f)
    return edges, edges_filename

def get_random_edges(graph, n):
    edges_filename='random'+str(n)+'.edges'
    random_edges=[list(graph.edges)[random.randint(0, len(graph.edges)-1)] for x in range(n)] 
    with open(EDGES_DIR+edges_filename, 'wb') as f:
        pickle.dump(random_edges,f)
    
    return edges_filename


def generate_SIGNAL_features(graph, edges_filename, multith):
    # Define the command to execute the second script with arguments
    print('GENERATING SIGNAL FEATURES')
    # LOAD SIGNAL FEATURES INPUTS
    
    #load network: already loaded
    
    # load perturbation signatures dictionaries
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
         
    start=time()
    if not multith:
        knockout_names, results = create_the_features_different_knockouts_iterative(raw_matrix, edges, gene_indexes, matrix, plus_targets_of_deletion,\
                              minus_targets_of_deletion,  num_genes, PROPAGATE_ALPHA, \
                                  PROPAGATE_ITERATIONS,PROPAGATE_EPSILON )
    else:
        packed_results = Parallel(n_jobs=N_JOBS)(delayed(create_the_features_different_knockouts)\
                                    (raw_matrix, edge, gene_indexes, matrix, plus_targets_of_deletion, minus_targets_of_deletion, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON)\
                                        for edge in edges)
        knockout_names, results = zip(*packed_results)

    print("time passed", time()-start)
    data = pd.DataFrame().from_records(results, index=[0,1], columns = knockout_names[0])
    with open(FT_DIR+edges_filename+'_'+PERT_MAP+'.ft','wb') as f:
        pickle.dump(data, f)
        
    return

def apply_SIGNAL(features_filename, training_model_name, MOD_DIR, write=False):
    from applySIGNAL import predictSIGNAL
    print('CALCULATING SIGNAL SCORE')
    print('we', features_filename, FT_DIR)

    with open( FT_DIR+features_filename, 'rb') as f:
        features_table = pickle.load(f)
    SIGNALscores = predictSIGNAL(features_table, training_model_name, MOD_DIR, train=False)
    if write:
        SIGNALscore_filename=features_filename.rstrip('ft')+'_'+training_model_name.rstrip('.rf')
        SIGNALscores.to_csv(SIGNAL_DIR+os.sep+SIGNALscore_filename+'.sgnl', sep=' ', header=False)
    return SIGNALscores

def print_SIGNAL( SIGNALscores_filename):
    print('SIGNAL SCORE OF EDGES FROM ANCHOR TO TERMINAL:')
    data=pd.read_csv(SIGNAL_DIR+SIGNALscores_filename,sep=' ', header=None, index_col=([0,1]))
    print(data)
#%%
def main():
    
    # Print inputs
    print('SIGNAL')
    print(NET_FILE)
    print(PERT_MAP)
    print(TRAIN_DATA)
    
    # LOAD inputs 
    with open(NET_DIR+NET_FILE, 'rb') as f:
        graph=pickle.load(f)
    
    with open(DICT_DIR+DICT_FILE, 'rb') as f:
        alias_2geneid=pickle.load(f)
    
    # Generate random edges file
    # edges_filename = get_random_edges(graph, n=9)

    # use trainig edges
    # edges_filename='alltrain' 
    # with open(EDGES_DIR+edges_filename, 'wb') as f:
    #     pickle.dump(list(graph.edges),f)
    
    #generate edges from ANAT
    edges, edges_filename = generate_edges_from_anat('TLM_all_KOterms', alias_2geneid, write=False)

    # # GENERATE FEATURES FOR SIGNAL:
    print(edges_filename)
    
    features_filename=edges_filename+'_'+PERT_MAP+'.ft'
    print(features_filename)
    if not os.path.isfile(FT_DIR+features_filename):
        generate_SIGNAL_features(graph, edges_filename, multith=True)
    
    # Print features
    # with open(FT_DIR+features_filename,'rb') as f:
    #     features=pickle.load(f)
    # print(features)
    
    # Apply signal and predict SIGNAL score
    SIGNAL_scores_filename=edges_filename+'_'+PERT_MAP+'.sgnl'
    # if not os.path.isfile(SIGNAL_DIR+SIGNAL_scores_filename):
        
    tmp = '_'.join(TRAIN_DATA)
    training_model_name=tmp+'_'+PERT_MAP+'.rf'
    signal_scores=apply_SIGNAL(features_filename, training_model_name, MOD_DIR, write=True)
    print(signal_scores)
    print('SIGNAL scores saved to:',SIGNAL_scores_filename)
    
    # Print the results
    # print_SIGNAL(SIGNAL_scores_filename)

if __name__ == '__main__':
    main()
