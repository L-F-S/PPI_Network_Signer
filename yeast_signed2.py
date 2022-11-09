# -*- coding: utf-8 -*-
"""
v2:
    -remove DATANAME flags
    -only use Holst as feature creation signatures
    -select base network (undirected, or Directed (daniel data))
    -load and add all datasets with signing info together and add to base net
    -create features for all datasets separately
    -only AFTER, train on different datasets and decide which are worth keeping for training and validation etc.
"""

import collections
import itertools
import lxml.etree as etree
import os
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from random import seed, sample
import networkx
from time import time
from datetime import date
from IPython.display import clear_output
from joblib import Parallel, delayed
from data_preproc2 import converttod2ddirected,get_kegg_Kpi,\
 get_protein_complexes_data, read_network_from_file, add_edges_from_labels,\
     graph_from_dataframe,\
     readname2geneid, get_ubiquitin_data, get_mutational_signatures,\
         extract_knockotut_effect_pairs_from_data, write
from train_and_vis import random_even_subsample, custom_k_fold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

##############################################################################
#  INPUTS
##############################################################################
SPECIES = "S_cerevisiae"

HOME_DIR  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
HOME_DIR  =  "G:" +os.sep+"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
#HOME_DIR  =  "/content/drive" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 

MAIN_DATA_DIR = HOME_DIR+'Data'+os.sep
DATA_DIR=HOME_DIR+"network_signing"+os.sep+'data'+os.sep
DIRECTED_DIR = HOME_DIR+"network_signing"+os.sep
OUTDIR=DIRECTED_DIR+'output'+os.sep
#FEATURES_DIR = HOME_DIR+"network_signing"+os.sep+"features_creation_data"+os.sep
HOLSTEGE_DIR= MAIN_DATA_DIR+os.sep+'Holstege_S1Data'+os.sep
IMG_DIR=HOME_DIR+"network_signing"+os.sep+'imgs'+os.sep

DATANAME= 'scUndirHol_'+str(date.today()) #'SchUndirHol_'+str(date.today())  # Sch: kpi ppi, sc, only ppi 'scnewD2DHolclean_2022-03-02'
DATASFX='.csv'
net_type='undir' #CHANGE HERE to 'dir' or 'undir danieldir'
if 'Und' in DATANAME:
    NET_FILENAME=HOME_DIR+"Data"+os.sep+"S_cerevisiae.net"

# Propagation parameters:
PROPAGATE_ALPHA = 0.6
PROPAGATE_EPSILON = 1e-5
PROPAGATE_ITERATIONS = 100

# Initioalize dictionary of aliases
alias_2geneid =  readname2geneid(MAIN_DATA_DIR, SPECIES)

##############################################################################
# ## Get training sets:
##############################################################################

kegg_Kpi_labels, kegg_Kpi_edge_weights = get_kegg_Kpi(DATA_DIR,alias_2geneid) #TODO why is this so fucking slow? 
pcomp_labels, pcomp_edge_weights = get_protein_complexes_data(MAIN_DATA_DIR, alias_2geneid)
ubiq_labels, ubiq_edge_weights = get_ubiquitin_data(MAIN_DATA_DIR, alias_2geneid)
#####
#%%
# write training labels:
write(OUTDIR, kegg_Kpi_labels,pcomp_labels,ubiq_labels )
#%%
#TDIDI make  the multi index into an index of tuples MAYBE? cos the original is like that

#TEMP: to check quanto fa schifo scud
    # in scud data: ubiq substrate: 'sytem_name'
    # e3: 1) substrate receptor 2 e3 subclass 3) e3 class
#     #ma quanti ce ne sono davvero ? printiamoli, e sono pochissimi
# tempdatadir=HOME_DIR+'Data'+os.sep
# ubiq_labels = pd.read_csv(tempdatadir+"SCUD2009_substrate.tsv", sep="\t",)
# print('e3 sub', ubiq_labels['E3_subclass'].unique())
# print('sub receptor', ubiq_labels['E3_subclass'].unique())

print("\n>>>>>>> loading base network to propagte on:")
network=read_network_from_file(NET_FILENAME, net_type)
if net_type == 'danieldir':
    threshold = 0.9
    network = converttod2ddirected(network, threshold)
print('lenght of network', network.shape[0])
graph = graph_from_dataframe(network, net_type)
#ADD edges from training labels which are not present:

graph = add_edges_from_labels(graph, kegg_Kpi_edge_weights, flag='kegg_kpi')
graph = add_edges_from_labels(graph, pcomp_edge_weights, flag='p_complex') 
graph = add_edges_from_labels(graph, ubiq_edge_weights, flag='ubiq')
genes = sorted(graph.nodes)
# generate validation features


holst = get_mutational_signatures(HOLSTEGE_DIR, MAIN_DATA_DIR, alias_2geneid)
plus_targets_of_deletion, minus_targets_of_deletion = extract_knockotut_effect_pairs_from_data(holst, genes)

print("\n--------------- DATA PREPROCESSED SUCCESFULLY---------------")
print('- Positive knockout experiment targets:', len(plus_targets_of_deletion),'\n- Negative knockout experiment targets:',len(minus_targets_of_deletion))
print('- Labels (0 for potive and 1 for negative signe of directed interaction):\n',len(kegg_Kpi_labels.index),len(pcomp_labels.index),len(ubiq_labels.index))
print('- Network size (for propagation): ','edges:', len(graph.edges), 'nodes:', len(graph.nodes)) # da mette a posto!! #TODO

from score_edges import  generate_similarity_matrix, create_the_features_different_knockouts 
raw_matrix = networkx.to_scipy_sparse_matrix(graph, genes, format='csc').T #N.b. siccome graph e' networkx.DiGraph, questa matrice sa se la edge e undirected ci saranno 2 punti nell adjacency matrix, se e directed, un punto solo, per ogni edge.  ALSO: .T is important because by default networkx directed graph returns a matrix where element i,j is the directed edge from i to j, while it should be vicevers afo r matrix normalizaition and for propagation
matrix, raw_col_sums = generate_similarity_matrix(raw_matrix,PROPAGATE_ALPHA) #normalized similarity matrix
num_genes     = len(genes)
gene_indexes  = dict([(gene, index) for (index, gene) in enumerate(genes)]) #all genes present in matrix
