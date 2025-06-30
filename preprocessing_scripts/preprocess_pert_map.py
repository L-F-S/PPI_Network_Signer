# -*- coding: utf-8 -*-
"""
Preprocesses perturbation dataset and signed datasets to create inputs for network_signer.py
Inputs are saved in /features_and_labels

v2:
    -remove DATANAME flags
    -only use knockdown_map as feature creation signatures
    -select base network (undirected, or Directed (daniel data))
    -load and add all datasets with signing info together and add to base net
    -create features for all datasets separately
    -only AFTER, train on different datasets and decide which are worth keeping for training and validation etc.
v3: 
    -add human
    -removeddirected base network option

v4 10 01 2024 removed redundant parts. preprocess_network MUST BE RUN BEFORE.
"""

import os
import pickle
from preproc_utils import  add_edges_from_labels,graph_from_dataframe,\
    preprocess_signed_datasets, readname2geneid, get_perturbations_map,\
         extract_knockotut_effect_pairs_from_data
from glob_vars import init_all
         

##############################################################################
#  INPUTS
##############################################################################
# SPECIES = "H_sapiens"#
SPECIES =  "S_cerevisiae" #

HOME_DIR, _, _, _, NET_DIR, PRT_DIR, DICT_DIR, _, _, _, _, NET_FILE = init_all(SPECIES)

RAW_DATA_DIR = '\path\to\raw\file' 

with open( DICT_DIR+'alias_2geneid.pkl', 'rb') as f:
    alias_2geneid = pickle.load(f)
#%%
# Load Network
print('Make sure to build base network with preprocess_network.py before, if needed')
with open(NET_DIR+NET_FILE, 'rb') as f:
    graph=pickle.load(f)  

genes=sorted(graph.nodes)

print('\t -Base network:', len(genes),'nodes;',len(graph.edges),'(directed) edges.')
#%%# Generate perturbation maps for subsequent feature generation:

    #%% 1 FOr yeast  main perturbation map used: Kemmeren // Kemmeren pert map:
if SPECIES == "S_cerevisiae":
    PERT_MAP_NAME ='Kemmeren'#'Kemmeren'#' they are the same.
    filename_of={'Kemmeren':'kemmeren_data_s1.cdt'}

perturbations_map=get_perturbations_map(RAW_DATA_DIR, alias_2geneid,SPECIES, filename_of[PERT_MAP_NAME], translate=True)
#%% 2 for human: using clustered_mean_gene_experssion data  from figures 2 and 4
if SPECIES == "H_sapiens":
    
    PERT_MAP_NAME = 'CMGE'# ADPBH
    filename_of = {'CMGE':'clustered_mean_gene_expression_figs2-4.csv.gz'}
                 #'ADPBH':'K562_gwps_normalized_bulk_01.h5ad'}

    
    
    import pandas as pd
    perturbations_map=pd.read_csv(RAW_DATA_DIR+filename_of[PERT_MAP_NAME], index_col=0, \
                      skiprows=[1,2],  header=0) #from fig 2a
    #columns= gene transcript
    #rows = gene name
    perturbations_map.drop(columns=perturbations_map.columns[0], inplace=True)
    perturbations_map.rename(columns= lambda x : x.split('_')[3], inplace=True)
    from preproc_utils import translate_axes
    from collections import defaultdict
    seen_id=defaultdict(list)
    seengene=[]
    dupedid=defaultdict(list)
    dupedgene=[]
    for i in perturbations_map.columns:
        if i in seengene:
            dupedgene.append(i)
            try:
                dupedid[alias_2geneid[i]].append( i)
            except:
                continue
        else:
            seengene.append(i)
            #fin qui non ci sn duplicati
        try:
            if alias_2geneid[i] in seen_id.keys():
                dupedid[alias_2geneid[i]].append(i)
            seen_id[alias_2geneid[i]].append(i)
        except:
            continue
    print([(i,seen_id[i]) for i in dupedid.keys()]) # ci sono alcuni ID che sono con piu nomi
    #nell index, e nel column ce ne sono 3 ripetuti lol
    
    translate_axes(perturbations_map, alias_2geneid) 
    perturbations_map=perturbations_map.loc[:,~perturbations_map.columns.duplicated()] # drop cols
    perturbations_map = perturbations_map[~perturbations_map.index.duplicated(keep=False)] #drop rows

#%% 3 for reimand yeast perturb map:
# from preproc_utils import translate_axes
# import networkx as nx
# filename='reimand.txt'
# pert_graph = graph_from_dataframe(RAW_DATA_DIR, SPECIES, net_type="dir",filename=filename)
# perturbations_map=nx.to_pandas_adjacency(pert_graph).T
# perturbations_map=translate_axes(perturbations_map, alias_2geneid)
#%% turn perturbations_map into 2 dictionaries
if SPECIES == 'S_cerevisiae' :
    threshold = 1.7
if SPECIES=='H_sapiens':
    threshold=0#1.009
plus_targets_of_deletion, minus_targets_of_deletion = extract_knockotut_effect_pairs_from_data(perturbations_map, genes, threshold=threshold)
print(len(plus_targets_of_deletion.keys()))

#%%
with open(PRT_DIR+'plus_targets_'+PERT_MAP_NAME+'.pkl', 'wb') as f:
    pickle.dump(plus_targets_of_deletion, f)
with open(PRT_DIR+'minus_targets_'+PERT_MAP_NAME+'.pkl','wb') as f:
    pickle.dump(minus_targets_of_deletion, f)
#%%
print("\n--------------- DATA PREPROCESSED SUCCESFULLY---------------")
print('- Positive knockout experiment targets:', len(plus_targets_of_deletion),'\n- Negative knockout experiment targets:',len(minus_targets_of_deletion))

