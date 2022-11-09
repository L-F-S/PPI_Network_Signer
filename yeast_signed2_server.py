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
import pandas as pd
import networkx
from time import time
from datetime import date
from joblib import Parallel, delayed
from data_preproc2 import wrapper_get_training_data, read_network_from_file,\
    add_edges_from_labels, graph_from_dataframe, readname2geneid, \
        get_mutational_signatures, extract_knockotut_effect_pairs_from_data,\
             write
from score_edges import generate_similarity_matrix, create_the_features_different_knockouts


##############################################################################
#  INPUTS
##############################################################################
SPECIES = "S_cerevisiae"

HOME_DIR  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# HOME_DIR  =  "G:" +os.sep+"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
#HOME_DIR  =  "/content/drive" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 

SPECIES = "S_cerevisiae"
HOME_DIR  =  '/home/bnet/lorenzos/signed/signedv2/'
DATA_DIR=HOME_DIR+"Data/"
MAIN_DATA_DIR=DATA_DIR
GENE_INFO_DIRECTORY = DATA_DIR
DIRECTED_DIR = DATA_DIR
HOLSTEGE_DIR= DATA_DIR
OUTDIR = HOME_DIR + 'output/'

DATASFX='.ft.csv'
net_type='undir' #CHANGE HERE to 'dir' or 'undir danieldir'
NET_FILENAME=HOME_DIR+"Data"+os.sep+"S_cerevisiae.net"

# Propagation parameters:
PROPAGATE_ALPHA = 0.6
PROPAGATE_EPSILON = 1e-5
PROPAGATE_ITERATIONS = 100


##############################################################################
# ## Get user inputs (N of cores, training daasets to use):
##############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('datasets', type=str, nargs='*',
                    default = ['kegg_kpi', 'ubiq', 'p_complex'],
                    help='dataset names: default and options: [kegg_kpi, ubiq, p_complex]')
parser.add_argument('-c', dest='N_JOBS', type=int, nargs='?', default=8,
                    help='number of corse to assign')

args = parser.parse_args()

N_JOBS = args.N_JOBS
datasets=args.datasets
print(N_JOBS, datasets)

##############################################################################
# ## Get training sets:
##############################################################################
# Initioalize dictionary of aliases
alias_2geneid =  readname2geneid(DATA_DIR, SPECIES)

labels_of, weights_of = wrapper_get_training_data(DATA_DIR, alias_2geneid,datasets)

# write training labels to file
write(OUTDIR, labels_of)


print("\n>>>>>>> loading base network to propagte on:")
network=read_network_from_file(NET_FILENAME, net_type)
print('lenght of network', network.shape[0])
graph = graph_from_dataframe(network, net_type)
#ADD edges from training labels which are not present:
for dataname, data_edge_weights in weights_of.items():
    print('adding edges from dataset', dataname)
    graph = add_edges_from_labels(graph, data_edge_weights, flag=dataname)
genes = sorted(graph.nodes)
# generate  features
holst = get_mutational_signatures(HOLSTEGE_DIR, MAIN_DATA_DIR, alias_2geneid)
plus_targets_of_deletion, minus_targets_of_deletion = extract_knockotut_effect_pairs_from_data(holst, genes)

print("\n--------------- DATA PREPROCESSED SUCCESFULLY---------------")
print('- Positive knockout experiment targets:', len(plus_targets_of_deletion),'\n- Negative knockout experiment targets:',len(minus_targets_of_deletion))
print('- Labels (0 for potive and 1 for negative signe of directed interaction):')
for name, label in labels_of.items():
    print(name, label.value_counts())
print('- Network size (for propagation): ','edges, \n\tof whch directed:') # da mette a posto!! #TODO

raw_matrix = networkx.to_scipy_sparse_matrix(graph, genes, format='csc').T #N.b. siccome graph e' networkx.DiGraph, questa matrice sa se la edge e undirected ci saranno 2 punti nell adjacency matrix, se e directed, un punto solo, per ogni edge. ALSO  .T is important because by default networkx directed graph returns a matrix where element i,j is the directed edge from i to j, while it should be vicevers afo r matrix normalizaition and for propagation
matrix, raw_col_sums = generate_similarity_matrix(raw_matrix,PROPAGATE_ALPHA) #normalized similarity matrix
num_genes     = len(genes)
gene_indexes  = dict([(gene, index) for (index, gene) in enumerate(genes)]) #all genes present in matrix
print('Preparing for propagations\n- Number of genes:',num_genes)

for datasetname, dataset_label in labels_of.items():
    print("-------------CREATING FEATURES for dataset "+datasetname+"------------------")
    start=time()
    
    results=Parallel(n_jobs=N_JOBS)(delayed(create_the_features_different_knockouts)\
                                (raw_matrix, edge, gene_indexes, matrix, plus_targets_of_deletion, minus_targets_of_deletion, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON)\
                                    for edge in dataset_label.index)
    
    print("time passed", time()-start)
    data = pd.DataFrame().from_records(results, index=0).drop(0, axis=1)   
    
    print(data.head())
    
    data.to_csv(OUTDIR+datasetname+DATASFX)
