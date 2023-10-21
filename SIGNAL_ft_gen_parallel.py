# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: L-F-S

SIGNAL
SIGN Annotation aLgorithm

Features generator

Inputs:
    Deletion signature dictionaries
    PPI interaction network
    Edges of interest

V4: Major refactoring. Edges input are now pkled files of lists of tuples.
 Network input is now pck of networkx.DiGraph (with all edges from training data included)

To update network, traning edges file, and perturbation dictionaries,
run preprocessing scripts inutility_scripts
"""
import os
import argparse
import pickle
import pandas as pd
import networkx
from time import time
from joblib import Parallel, delayed
from score_edges import generate_similarity_matrix, create_the_features_different_knockouts
from glob_vars import SPECIES, PERT_MAP, TRAIN_DATA, HOME_DIR, LBL_DIR,\
    EDGES_DIR, FT_DIR, NET_DIR, PRT_DIR, NET_FILE, PROPAGATE_ALPHA,\
    PROPAGATE_EPSILON, PROPAGATE_ITERATIONS, N_JOBS

parser = argparse.ArgumentParser()
parser.add_argument('-e', dest='edges', type=str, nargs='*', default=None,
                    help='Optional. Network edges to create features for.\
                        Options: None: creates features for training edges only\
                        \'all\': creates features for all edges in the PPI network,\
                        <fiilename>: creates features for pickled list of tuples corresponding to edge names.\
                        Edges must be part of base PPI network.')
parser.add_argument('-c', dest='N_JOBS', type=int, nargs='?', default=None,
                    help='number of corse to assign')
parser.add_argument('-s', dest='SPECIES', type=str, nargs='?', default=None,
                    help='species: [\'H_sapiens\', \'S_cerevisiae\']\ndefault: S_cerevisiae')
parser.add_argument('-p', dest='PERT_MAP', type=str, nargs='?', default=None,
                    help='default: \'Holstege\' , other options: [\'reimand\', ADPBH, CMGE..]')
parser.add_argument('-ld', dest='LBL_DIR', type=str, nargs='?', default=None,
                    help='Optional, directory of training labels')
parser.add_argument('-ed', dest='EDGES_DIR', type=str, nargs='?', default=None,
                    help='Optional, directory of .edges file (with which edges to make features for)')
parser.add_argument('-nd', dest='NET_DIR', type=str, nargs='?', default=None,
                    help='Optional, directory of network file')
parser.add_argument('-nf', dest='NET_FILE', type=str, nargs='?', default=None,
                    help='Optional, network file name (file must be a pickled networkx.DiGraph object)')
parser.add_argument('-kd', dest='PRT_DIR', type=str, nargs='?', default=None,
                    help='Optional, directory of deletion signatures files (files must be a pickled dictionary)')
parser.add_argument('-fd', dest='FT_DIR', type=str, nargs='?', default=None,
                    help='Optional, directory of SIGNAL features (input for applySIGNAL)')
args = parser.parse_args()

SPECIES= SPECIES if not args.SPECIES else args.SPECIES
PERT_MAP=PERT_MAP if not args.PERT_MAP else args.PERT_MAP

# input dirs:
HOME_DIR = HOME_DIR
os.chdir(HOME_DIR)
LBL_DIR = LBL_DIR if not args.LBL_DIR else args.LBL_DIR
EDGES_DIR = EDGES_DIR if not args.EDGES_DIR else args.EDGES_DIR
NET_DIR = NET_DIR if not args.NET_DIR else args.NET_DIR
PRT_DIR = PRT_DIR if not args.PRT_DIR else args.PRT_DIR

# output dir:
FT_DIR= FT_DIR if not args.FT_DIR else args.FT_DIR

# File names:
NET_FILE = NET_FILE if not args.NET_FILE else args.NET_FILE

# Multi-threading optiions:
N_JOBS = N_JOBS if not args.N_JOBS else args.N_JOBS
print('------------------------------------------')
print('-------> SIGNAL features creation <-------')
print('------------------------------------------')
print()
print('>Inputs:')

# Load Network
with open(NET_DIR+NET_FILE, 'rb') as f:
    graph=pickle.load(f)  
print('\t -Base network:', len(graph.nodes),'nodes;',len(graph.edges),'(directed) edges.')

# Load perturbation signatures:
with open(PRT_DIR+'plus_targets_'+PERT_MAP+'.pkl', 'rb') as f:
    plus_targets_of_deletion = pickle.load(f)
with open(PRT_DIR+'minus_targets_'+PERT_MAP+'.pkl', 'rb') as f:
    minus_targets_of_deletion = pickle.load(f)
print('\t -Perturbation signatures:\n\t\t- Positive knockout experiment targets:',\
      len(plus_targets_of_deletion),'\n\t\t- Negative knockout experiment targets:',len(minus_targets_of_deletion))

# Load Edges  
if not args.edges:
    edges_list=[]
    for data in TRAIN_DATA:
        with open(EDGES_DIR+data+'.edges','rb') as f:
            edges=pickle.load(f) #list of edge names tuples
            edges_list+=[edges]
    names=TRAIN_DATA
elif args.edges == 'all':
    edges_list = [list(graph.edges)]
    names=['all']
else:
    names=args.edges
    edges_list=[]
    for name in names:
        with open(EDGES_DIR+name,'rb') as f:
            edges=pickle.load(f) #list of edge names tuples
            edges_list+=[edges]
print('\t - Edges to create features for: ')
for name, edges in zip(names, edges_list):
    print('\t\t -', name, len(edges))
print('\t\t Total: ',sum([len(edges) for edges in edges_list]))
print('------------------------------------------')
print()
#%%
def generate_similarity_matrix_wrapper(graph):
    print('Generating similarity matrix..')
    genes = sorted(graph.nodes)
    raw_matrix = networkx.to_scipy_sparse_matrix(graph, genes, format='csc').T
    matrix, raw_col_sums = generate_similarity_matrix(raw_matrix) #normalized similarity matrix
    num_genes     = len(genes)
    gene_indexes  = dict([(gene, index) for (index, gene) in enumerate(genes)]) #all genes present in matrix
    
    return genes, raw_matrix, matrix, raw_col_sums, num_genes, gene_indexes

genes, raw_matrix, matrix, raw_col_sums, num_genes, gene_indexes = generate_similarity_matrix_wrapper(graph)

print('------------------------------------------')
print('----> Running defective propagations <----')
print('------------------------------------------')
print('..')
# The slow part!

print('Number of threads:', N_JOBS)
for name, edges in zip(names, edges_list):
    start=time()
    packed_results = Parallel(n_jobs=N_JOBS)(delayed(create_the_features_different_knockouts)\
                                (raw_matrix, edge, gene_indexes, matrix, plus_targets_of_deletion, minus_targets_of_deletion, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON)\
                                    for edge in edges)
    print("time passed", time()-start)
    knockout_names, results =zip(*packed_results)
    data = pd.DataFrame().from_records(results, index=[0,1], columns = knockout_names[0])
    with open(FT_DIR+name+'_'+PERT_MAP+'.ft','wb') as f:
        pickle.dump(data, f)
