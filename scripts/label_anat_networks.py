# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 01:35:01 2023

@author: los4
"""

import os
import sys


from glob_vars import SPECIES, TRAIN_DATA, PERT_MAP, LBL_DIR,EDGES_DIR,\
    FT_DIR, SIGNAL_DIR, MOD_DIR, PRT_DIR, NET_DIR,  NET_FILE,\
    PROPAGATE_ALPHA, PROPAGATE_EPSILON ,PROPAGATE_ITERATIONS, DICT_DIR,\
    DICT_FILE, ANAT_VAL_DIR, N_JOBS, get_terms, get_anchors, ANAT_VAL_HOME, ANAT_VAL_OUT
import pickle
import pandas as pd
import networkx as nx
import numpy as np
from preproc_utils import load_training_data, graph_from_dataframe
from score_edges import generate_similarity_matrix, create_the_features_different_knockouts_iterative,\
    create_the_features_different_knockouts
from joblib import Parallel, delayed
from time import time

    
with open( DICT_DIR+DICT_FILE, 'rb') as f:
    alias_2geneid = pickle.load(f)
geneid2alias ={y:x for (x,y) in alias_2geneid.items()}

term_data_name='TLM_strong_normal' # 'TLM_all_KOterms' #
FILENAME = term_data_name+'.edges_Kemmeren._patkar_kegg_kpi_ubinet2_Kemmeren.sgnl'#'TLM_all_KOterms_850696'#
SIGNAL_scores=SIGNAL_scores=pd.read_csv(SIGNAL_DIR+FILENAME, sep=' ', header=None, names=['id1','id2','SIGNAL'], index_col=(0,1))
FILENAME = term_data_name+'_nodes.csv'#'TLM_all_KOterms_850696'#
NODES=pd.read_csv(ANAT_VAL_DIR+FILENAME)
FILENAME = term_data_name+'.csv'#'TLM_all_KOterms_850696'#
EDGES=pd.read_csv(ANAT_VAL_DIR+FILENAME)
TLMphenotypesdf=pd.read_csv(ANAT_VAL_HOME+'1 ANAT Build networks'+os.sep+'Input'+os.sep+term_data_name+'_phenotype_labelled.txt', header=0, sep='\t') # TLMphenotypesnodupes

#%%
def apply_loose_label(x):
    if x.startswith('DAmP'):
        return None
    if x.startswith('slightly'):
        return None
    if x == 'ss':
        return 0
    if  x == 'sl':
        return 1
    if x.lower().startswith('l'):
        return 1
    if x == 'VL':
        return 1
    if x.lower().startswith('s'):
        return 0
    if x=='VS':
        return 0
    
TLMphenotypesdf['label'] = TLMphenotypesdf['Telomere Phenotype*'].apply(lambda x: apply_loose_label(x))
loose_labels=TLMphenotypesdf.dropna()
loose_labels['label'].value_counts()
loose_label_of = dict(zip(loose_labels['Gene (corrected for NGE)'], loose_labels['label']))
loose_label_ofid = {alias_2geneid[x]:y for (x,y) in loose_label_of.items()}
terms=get_terms(termfile=term_data_name+'_phenotype_labelled')
anchors=get_anchors()
#%%
def set_pht(x):
    if x in loose_label_of.keys():
        return loose_label_of[x]
    return -1

NODES['phenotype']=NODES['name'].apply(lambda x: set_pht(x))
#%%
def set_phtid(x):
    if x in loose_label_ofid.keys():
        return loose_label_ofid[x]
    return -1

def set_status(x):
    return NODES['status'][NODES['shared name']==x].iloc[0]
    
EDGES['ID1phenotype'] = EDGES['ID1'].apply(lambda x: set_pht(x))
EDGES['ID2phenotype'] = EDGES['ID2'].apply(lambda x: set_pht(x))
EDGES['statusID1'] = EDGES['ID1'].apply(lambda x: set_status(x))
EDGES['statusID2'] = EDGES['ID2'].apply(lambda x: set_status(x))

# aggiungi signal score
EDGES['SIGNAL'] = range(EDGES.shape[0])
for i, row in EDGES.iterrows():
    edge=(alias_2geneid[row['ID1']], alias_2geneid[row['ID2']])
    EDGES['SIGNAL'].loc[i] = SIGNAL_scores.loc[edge].values
    # salva file
EDGES.drop(columns=['constraint', 'edgeProbability', 'interaction',
       'isDirected', 'name', 'pubmedID', 'redundancy', 'selected',
       'shared interaction', 'shared name', 'status', 'wasDirected'], inplace=True)
EDGES.to_csv(ANAT_VAL_OUT+term_data_name+'_SIGNAL.csv')

#%% come tradurre edges files e node files in un grafo da dare a cytoschiappa??
#%% fatto ma scomodo da usare questo

#########################                   
#   BUILD graphml file  #
#########################
# with open(NET_DIR+NET_FILE, 'rb') as f:
#     graph=pickle.load(f)

# with open(DICT_DIR+DICT_FILE, 'rb') as f:
#     alias_2geneid=pickle.load(f)

# # only keep nodes of anat graph
# graph.remove_nodes_from([n for n in graph if n not in set(NODES.xrefID)])

# geneid_2alias={alias_2geneid[node]:node for node in NODES.name}

# loos_label_of_genename = {alias_2geneid[genename]: label for (genename, label) in loose_label_of.items()}
    
# nx.set_node_attributes(graph, {gene:(geneid_2alias[gene] if gene in geneid_2alias.keys() else gene) for gene in graph.nodes}, name='CommonName')
# nx.set_node_attributes(graph, loos_label_of_genename, name='terminal_phenotype')

# scores_dict=dict(zip(SIGNAL_scores.index, SIGNAL_scores['SIGNAL']))
# nx.set_edge_attributes(graph, {edge:(signal if edge in scores_dict.keys() else 0) for (edge, signal) in scores_dict.items()}, name='SIGNAL')
# #%%
# nx.readwrite.graphml.write_graphml(graph, ANAT_VAL_OUT+ term_data_name + '.graphml',named_key_ids=True)
