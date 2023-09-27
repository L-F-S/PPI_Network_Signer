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
                    help='dataset names:  options (and default) for S_cerevisiae: [\'kegg\',\'kpi\', \'ubiq\'],\
                        options (anmd default) for H_sapiens: [\'kegg\',\'PSP\',\'depod\',\'ubinet2\']')
parser.add_argument('-c', dest='N_JOBS', type=int, nargs='?', default=8,
                    help='number of corse to assign')
parser.add_argument('-e', dest='edges', type=str, nargs='?', default=None,
                    help='Optional. Network edges to create features for.\
                        Options: None: creates features for training edges only\
                        \'all\': creates features for all edges in the PPI network,\
                        \'validation\': creates features for test edges only (i.e. not in training data),\
                        <fiilename>: creates features for pickled list of tuples corresponding to edge names.\
                        Edges must be part of base PPI network.')
parser.add_argument('-s', dest='SPECIES', type=str, nargs='?', default='S_cerevisiae',
                    help='species: [\'H_sapiens\', \'S_cerevisiae\']\ndefault: S_cerevisiae')
parser.add_argument('-p', dest='perturbation_filename', type=str, nargs='?', default='Holstege',
                    help='default: \'Holstege\' , other options: [\'reimand\', ADPBH, CMGE..]')
parser.add_argument('-o', dest='OUTDIR', type=str, nargs='?', default=None,
                    help='Optional, output directory')
parser.add_argument('-ed', dest='EDGESDIR', type=str, nargs='?', default=None,
                    help='Optional, directory of .edges file (with which edges to make features for)')

args = parser.parse_args()

N_JOBS = args.N_JOBS
SPECIES = args.SPECIES
if len(args.datasets) == 0:
    datasets = ['kegg','kpi', 'ubiq'] if SPECIES == 'S_cerevisiae' else ['kegg','PSP','depod','ubinet2']
else:
    datasets = args.datasets
print(N_JOBS, datasets)
print(SPECIES)

HOME_DIR  =  '/home/bnet/lorenzos/signed/signedv3/'
DATA_DIR=HOME_DIR+"input"+os.sep+SPECIES+os.sep
DIRECTED_DIR = HOME_DIR+"network_signing"+os.sep
if not args.OUTDIR:
    OUTDIR=HOME_DIR+'output'+os.sep+SPECIES+os.sep+'features'+os.sep
else:
    OUTDIR=args.OUTDIR

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
#%% Select edges to use
        
if not args.edges:
    edges_list = [d.index for d in signed_datasets_labels.values()]
    names=list(signed_datasets_labels.keys())
elif args.edges == 'all':
    edges_list = [list(graph.edges)]
    names=['all']
elif args.edges == 'valid':
    edges = list(set(graph.edges)-set(pd.concat(list(signed_datasets_labels.values())).index))
    edges_list = [edges]
    names=['validation']
else:
    names=[args.edges]
    if not args.EDGESDIR:
        EDGESDIR=INPUT_DIR
    else:
        EDGESDIR=args.EDGESDIR
    with open(EDGESDIR+args.edges+'.edges','rb') as f:
        edges=pickle.load(f)
    edges_list=[edges]
print('- # edges to create features for: ', sum([len(edges) for edges in edges_list]))
print('------------------------------------------')
#%%
###############################################################################
#    Generate similarity matrix:
###############################################################################
print('Generating similarity matrix.')
raw_matrix = networkx.to_scipy_sparse_matrix(graph, genes, format='csc').T
matrix, raw_col_sums = generate_similarity_matrix(raw_matrix,PROPAGATE_ALPHA) #normalized similarity matrix
num_genes     = len(genes)
gene_indexes  = dict([(gene, index) for (index, gene) in enumerate(genes)]) #all genes present in matrix
###############################################################################
#%%    Generate features (the slow part!):
###############################################################################
print('Generating features...')
for name, edges in zip(names, edges_list):
    start=time()
    packed_results = Parallel(n_jobs=N_JOBS)(delayed(create_the_features_different_knockouts)\
                                (raw_matrix, edge, gene_indexes, matrix, plus_targets_of_deletion, minus_targets_of_deletion, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON)\
                                    for edge in edges)
    print("time passed", time()-start)
    knockout_names, results =zip(*packed_results)
    data = pd.DataFrame().from_records(results, index=[0,1], columns = knockout_names[0])
    #data.to_csv(OUTDIR+name+'_'+perturbations_name+'.ft.csv')
    with open(OUTDIR+name+'_'+perturbations_name+'.ft','wb') as f:
        pickle.dump(data, f)
