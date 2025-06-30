# -*- coding: utf-8 -*-
"""
@author: L-F-S

Reformats ANAT network output

python SIGNAL_scorenetworks_server.py filename

-takes ANAT anchor-terminal subnetworks,
-extracts separate anchor-terminal paths
-generates signal features for all edges within those paths
-applies pre-trained SIGNAL model (can also train but untested in this script)
- and calculates SIGNAL scores for each edge in that path

MA MOMENTANEAMENTE 
"""

import pickle
import collections
import itertools
import lxml.etree as etree
import os
import pandas as pd
import networkx as nx
import numpy as np
from datetime import date
import subprocess
import sys
import argparse
# DRIVE_PATH='G:'+os.sep+'Il Mio Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep
HOME_DIR  =  '/home/bnet/lorenzos/signed/signedv3/'
# HOME_DIR  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 

DRIVE_PATH=HOME_DIR
sys.path.append(DRIVE_PATH)

# parser = argparse.ArgumentParser()
# parser.add_argument('datasets', type=str, nargs='*',
#                     help='dataset names:  options (and default) for S_cerevisiae: [\'kegg\',\'kpi\', \'ubiq\'],\
#                         options (anmd default) for H_sapiens: [\'kegg\',\'PSP\',\'depod\',\'ubinet2\']')
# parser.add_argument('-c', dest='N_JOBS', type=int, nargs='?', default=8,
#                     help='number of corse to assign')
# parser.add_argument('-e', dest='edges', type=str, nargs='?', default=None,
#                     help='Optional. Network edges to create features for.\
#                         Options: None: creates features for training edges only\
#                         \'all\': creates features for all edges in the PPI network,\
#                         \'validation\': creates features for test edges only (i.e. not in training data),\
#                         <fiilename>: creates features for pickled list of tuples corresponding to edge names.\
#                         Edges must be part of base PPI network.')
# parser.add_argument('-s', dest='SPECIES', type=str, nargs='?', default='S_cerevisiae',
#                     help='species: [\'H_sapiens\', \'S_cerevisiae\']\ndefault: S_cerevisiae')
# parser.add_argument('-p', dest='perturbation_filename', type=str, nargs='?', default='Holstege',
#                     help='default: \'Holstege\' , other options: [\'reimand\', ADPBH, CMGE..]')
# args = parser.parse_args()


SPECIES = "S_cerevisiae"
# SPECIES='H_sapiens'
PERT_MAP = 'Kemmeren'#'CMGE'
DATE=str(date.today().strftime("%d/%m/%Y"))

INPUT_DIR =  HOME_DIR+ 'input'+os.sep+SPECIES+os.sep
os.chdir(HOME_DIR)

outputpath=HOME_DIR+'output'+os.sep+SPECIES+os.sep
featurespath=outputpath+'features'+os.sep
modelpath=outputpath+'models'+os.sep
predictionspath=outputpath+'predictions'+os.sep

with open( INPUT_DIR+'alias_2geneid.pkl', 'rb') as f:
    alias_2geneid = pickle.load(f)
geneid2alias ={y:x for (x,y) in alias_2geneid.items()}

network=pd.read_csv(INPUT_DIR +SPECIES+'.net',sep="\t", header=None)
graph = nx.from_pandas_edgelist(network.reset_index(), 0, 1, 2)

#%%
if len(sys.argv)==1:

    
    if SPECIES=='S_cerevisiae':
        FILENAME='TLM_all_KOterms'
        f=open(INPUT_DIR+'TLManchors.txt')
        lines=[x.strip() for x in f.readlines()]
        ANCHORS=lines
        # ANCHORS=[str(alias_2geneid[node.upper()]) for node in ANCHORS]
        f.close()
    else:
        FILENAME = 'TUBB_v_all'
        ANCHORS='TUBB'
        TERMS=['SMAD3','JUN','TP53','RPS6KB1','EGFR','MARCKS','PRAS40','PTP2C','IKBA',
                'GSK3A','AKT1','HSPB1','H2A.X','MAPK11','MEK1','STAT3','ERK1','RSK1','CREB1','FAK1']
        TERMS = [str(alias_2geneid[node.upper()]) for node in TERMS]
        ANCHORS = [str(alias_2geneid[ANCHORS])]
else:
    args = sys.argv[1:]
    TERMS = []
    for arg in args[:-2]:
        TERMS.append(arg)
    ANCHORS =args[-2]
    FILENAME =args[-1]

#%%
def generate_edges_from_path(path):
    edges=[]
    first=True
    i=0
    for j in path:
        if first:
          first=False
        else:
            edges.append((i,j))
        i=j
    return edges

def generate_edges_dict(id_paths_to, anchor, write=True):
    edges_path_to={}
    for final_node, path in id_paths_to.items():
        
        edges_path_to[final_node]=[]
        edges = generate_edges_from_path(path)
        edges_path_to[final_node].append(edges)
        
        # write .edges file for SIGNAL input
        if write:
            with open(INPUT_DIR+'SIGNAL_input'+os.sep+str(anchor)+'_'+str(final_node)+'.edges', 'wb') as f:
                pickle.dump(edges,f)
    return edges_path_to

def generate_edges_from_ANAT(FILENAME, write=True):
    network_file=INPUT_DIR+FILENAME
    columns = ['ID1', 'ID2']
    # Read the CSV file and select the desired columns and rows
    edges_df = pd.read_csv(network_file+'.csv', usecols=columns)
    print('Generating edges file for SIGNAL from ANAT network of ',edges_df.shape[0],'edges')
    edges=[(alias_2geneid[x], alias_2geneid[y]) for (x,y) in list((zip(edges_df['ID1'],edges_df['ID2'])))]
    edges_filename=FILENAME+'.edges'
    if write:
        with open(INPUT_DIR+edges_filename, 'wb') as f:
            pickle.dump(edges,f)
    return edges, edges_filename

def generate_SIGNAL_features( edges_filename):
    # Define the command to execute the second script with arguments
    print('GENERATING SIGNAL FEATURES')
    command = ['python', DRIVE_PATH+'SIGNAL_ft_gen_PARALLEL.py',
                '-e', edges_filename, '-s', SPECIES, '-p', PERT_MAP,
                '-o', featurespath, '-ed', INPUT_DIR]
    
    # Execute the command and capture the output
    output = subprocess.check_output(command, universal_newlines=True)
    
    # Print the output
    print(output) # output
    return

def apply_SIGNAL(features_filename):
    print('CALCULATING SIGNAL SCORE')
    command = ['python', DRIVE_PATH+os.sep+'applySIGNAL_server.py',
               '-f', features_filename, '-s', SPECIES, '-p', PERT_MAP,
               '-i', featurespath,'-o',predictionspath]
   
    output = subprocess.check_output(command, universal_newlines=True)
    return

def print_SIGNAL( features_filename):
    print('SIGNAL SCORE OF EDGES FROM ANCHOR TO TERMINAL:', features_filename)
    data=pd.read_csv(predictionspath+features_filename+'.sgnl',sep=' ', header=None, index_col=([0,1]))
    print(data)

def main():
    
    # Generate edges file:
    edges_filename = FILENAME+'.edges'
    if not os.path.isfile(INPUT_DIR+edges_filename):
        edges, edges_filename = generate_edges_from_ANAT(FILENAME)
    
    # Generate features for SIGNAL (slow):
    features_filename=FILENAME+'_'+PERT_MAP+'.ft'
    if not os.path.isfile(featurespath+features_filename):
        generate_SIGNAL_features(FILENAME)
    
    # Apply signal and predict SIGNAL score
    SIGNAL_scores_filename=FILENAME+'_'+PERT_MAP+'.sgnl'
    if not os.path.isfile(predictionspath+SIGNAL_scores_filename):
        apply_SIGNAL(FILENAME+'_'+PERT_MAP)
    
    # Print the results
    print_SIGNAL(FILENAME+'_'+PERT_MAP)

if __name__ == '__main__':
    main()
