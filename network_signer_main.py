# -*- coding: utf-8 -*-
"""
v2:
    -remove DATANAME flags
    -only use Holst as feature creation signatures
    -initially undirected base network
    -load and add all datasets with signing info together and add to base net
    -create features for all datasets separately
    INPUT
"""

import os
import argparse
import pickle
import pandas as pd
import networkx
from time import time
from joblib import Parallel, delayed
from preproc_utils import add_edges_from_labels, graph_from_dataframe,\
    readname2geneid, load_training_data
from score_edges import generate_similarity_matrix, create_the_features_different_knockouts

###############################################################################
#   INPUTS
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('datasets', type=str, nargs='*',
                    help='dataset names:  options (and default) for S_cerevisiae: [kegg_kpi, ubiq, p_complex],\
                        options (anmd default) for H_sapiens: [\'kegg\',\'PSP\',\'depod\',\'ubinet2\']')
parser.add_argument('-c', dest='N_JOBS', type=int, nargs='?', default=8,
                    help='number of corse to assign')
parser.add_argument('-s', dest='SPECIES', type=str, nargs='?', default='H_sapiens',
                    help='species: [\'H_sapiens\', \'S_cerevisiae\']')
parser.add_argument('-p', dest='perturbation_filename', type=str, nargs='?', default='',
                    help='default: \'Holstege\' , other options: [\'reimand\', ADPBH..]')
args = parser.parse_args()

N_JOBS = args.N_JOBS
SPECIES = args.SPECIES
if len(args.datasets) == 0:
    datasets = ['kegg_kpi', 'ubiq', 'p_complex'] if SPECIES == 'S_cerevisiae' else ['kegg','PSP','depod','ubinet2']
else:
    datasets = args.datasets
print(N_JOBS, datasets)

HOME_DIR  =  '/home/bnet/lorenzos/signed/signedv3/'
DATA_DIR=HOME_DIR+"input"+os.sep+SPECIES+os.sep
DIRECTED_DIR = HOME_DIR+"network_signing"+os.sep
OUTDIR=HOME_DIR+'output'+os.sep+SPECIES+os.sep

# Propagation parameters:
PROPAGATE_ALPHA = 0.6
PROPAGATE_EPSILON = 1e-5
PROPAGATE_ITERATIONS = 100

###############################################################################
#     Load training sets:
###############################################################################
print("\n>>>>>>> loading signed edges datasets:")
alias_2geneid =  readname2geneid(DATA_DIR, SPECIES)
signed_datasets_labels, signed_datasets_edge_weights = load_training_data(DATA_DIR, datasets, SPECIES)

###############################################################################
#     Load PPI network:
###############################################################################
print("\n>>>>>>> loading base network to propagte on:")
graph = graph_from_dataframe(DATA_DIR, SPECIES)
#ADD edges from training labels which are not present:

print("\n>>>>>>> adding missing edges from signed data to base network:")
for dataname, signed_data in signed_datasets_edge_weights.items():
    graph = add_edges_from_labels(graph, signed_data, v=False)

###############################################################################
#     Load perturbation maps:
###############################################################################

genes = sorted(graph.nodes)

perturbations_name=args.perturbation_filename
with open(DATA_DIR+'plus_targets_'+perturbations_name+'.pkl', 'rb') as f:
    plus_targets_of_deletion = pickle.load(f)
with open(DATA_DIR+'minus_targets_'+perturbations_name+'.pkl', 'rb') as f:
    minus_targets_of_deletion = pickle.load(f)

print("\n--------------- INPUT DATA ---------------")
print('- Positive knockout experiment targets:', len(plus_targets_of_deletion),'\n- Negative knockout experiment targets:',len(minus_targets_of_deletion))
print('- Labels (0 for positive and 1 for negative signe of directed interaction):')
for name, label in signed_datasets_labels.items():
    print('\t',name,':')
    for i in label.value_counts().index:
        print('\t\t'+str(i)+'\t'+str(label.value_counts()[i]))
print('- Base network size: ','\n\t',len(graph.edges()),'edges','\n\t',len(graph.nodes()),'nodes')
#%%
###############################################################################
#    Generate similarity matrix:
###############################################################################
print('Generating similarity matrix.')
raw_matrix = networkx.to_scipy_sparse_matrix(graph, genes, format='csc').T
matrix, raw_col_sums = generate_similarity_matrix(raw_matrix,PROPAGATE_ALPHA) #normalized similarity matrix
num_genes     = len(genes)
gene_indexes  = dict([(gene, index) for (index, gene) in enumerate(genes)]) #all genes present in matrix
#%%
###############################################################################
#    Generate features (the slow part!):
###############################################################################
print('Starting propagations') #todo non funzionana in locale

for datasetname, dataset_label in signed_datasets_labels.items():
    print("-------------CREATING FEATURES for dataset "+datasetname+"------------------")
    start=time()
    
    packed_results = Parallel(n_jobs=N_JOBS)(delayed(create_the_features_different_knockouts)\
                                (raw_matrix, edge, gene_indexes, matrix, plus_targets_of_deletion, minus_targets_of_deletion, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON)\
                                    for edge in dataset_label.index)
    
    print("time passed", time()-start)
    knockout_names, results =zip(*packed_results)
    data = pd.DataFrame().from_records(results, index=[0,1], columns = knockout_names[0])
    
    print(data.head())
    
    data.to_csv(OUTDIR+datasetname+'_'+perturbations_name+'.ft.csv')
