# -*- coding: utf-8 -*-
"""
Reformats ANAT network output

python 2SIGNALscorenetowrks.py filename

applies SIGNAL to edges of anchor-terminal subnetworks

V2 FIRST APPLY SIGNAL TO THE ANAT EDGES FILE AND AFTER ONLY AFTER FIND SUBPATHS!
"""

import pickle
import collections
import itertools
import lxml.etree as etree
import os
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from random import seed, sample
from datetime import date
import subprocess
import sys
# DRIVE_PATH='G:'+os.sep+'Il Mio Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep
DRIVE_PATH='G:'+os.sep+'My Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep
sys.path.append(DRIVE_PATH+'network_signing'+os.sep)

SPECIES = "S_cerevisiae"
# SPECIES='H_sapiens'
PERT_MAP = 'Kemmeren'#'CMGE'
DATE=str(date.today().strftime("%d/%m/%Y"))
HOME_DIR=DRIVE_PATH+'network_signing'+os.sep+\
    'SIGNAL Validations target sign reconstruction'\
        +os.sep+'ANAT_validation_pipeline'+os.sep
MAIN_DATA_DIR = DRIVE_PATH+'Data'+os.sep+SPECIES+os.sep
INPUT_DIR =  HOME_DIR+ '2 SIGNAL score networks'+os.sep
os.chdir(HOME_DIR)

outputpath=INPUT_DIR+'SIGNAL_output'+os.sep
featurespath=INPUT_DIR+'SIGNAL_features'+os.sep

with open( MAIN_DATA_DIR+'alias_2geneid.pkl', 'rb') as f:
    alias_2geneid = pickle.load(f)
geneid2alias ={y:x for (x,y) in alias_2geneid.items()}

network=pd.read_csv(MAIN_DATA_DIR +SPECIES+'.net',sep="\t", header=None)
graph = nx.from_pandas_edgelist(network.reset_index(), 0, 1, 2)


if len(sys.argv)==1:

    
    if SPECIES=='S_cerevisiae':
        FILENAME='TLM_all_KOterms'
        f=open(HOME_DIR+'1 ANAT Build networks'+os.sep+'Input'+os.sep+'TLManchors.txt')
        lines=[x.strip() for x in f.readlines()]
        ANCHORS=lines
        # ANCHORS=[str(alias_2geneid[node.upper()]) for node in ANCHORS]
        f.close()
        TLMphenotypesdf=pd.read_csv(HOME_DIR+'1 ANAT Build networks'+os.sep+'Input'+os.sep+'TLMphenotypes.txt', header=0, sep='\t')
        TERMS=list(TLMphenotypesdf[TLMphenotypesdf.columns[0]].values)
        # TERMS=[str(alias_2geneid[x]) for x in list(TLMphenotypesdf[TLMphenotypesdf.columns[0]].values)]
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

def generate_edges_from_anat(FILENAME, write=True):
    network_file='1 ANAT Build networks'+os.sep+'ANAT output'+os.sep+FILENAME
    columns = ['ID1', 'ID2']
    # Read the CSV file and select the desired columns and rows
    edges_df = pd.read_csv(network_file+'.csv', usecols=columns)
    edges=[(alias_2geneid[x], alias_2geneid[y]) for (x,y) in list((zip(edges_df['ID1'],edges_df['ID2'])))]
    edges_filename=FILENAME+'.edges'
    if write:
        with open(INPUT_DIR+'SIGNAL_input'+os.sep+edges_filename, 'wb') as f:
            pickle.dump(edges,f)
    return edges, edges_filename

def generate_SIGNAL_features( edges_filename):
    # Define the command to execute the second script with arguments
    print('GENERATING SIGNAL FEATURES')
    command = ['python', DRIVE_PATH+'network_signing'+os.sep+'SIGNAL_ft_gen_iterative.py',
                '-e', edges_filename, '-s', SPECIES, '-p', PERT_MAP,
                '-o', featurespath, '-ed', INPUT_DIR+'SIGNAL_input'+os.sep]
    
    # Execute the command and capture the output
    output = subprocess.check_output(command, universal_newlines=True)
    
    # Print the output
    print(output) # output
    return

def apply_SIGNAL(features_filename):
    print('CALCULATING SIGNAL SCORE')
    command = ['python', DRIVE_PATH+'network_signing'+os.sep+'applySIGNAL.py',
               '-f', features_filename, '-s', SPECIES, '-p', PERT_MAP,
               '-i', featurespath,'-o',outputpath]
   
    output = subprocess.check_output(command, universal_newlines=True)
    return

def print_SIGNAL( SIGNALscores_filename):
    print('SIGNAL SCORE OF EDGES FROM ANCHOR TO TERMINAL:')
    data=pd.read_csv(outputpath+SIGNALscores_filename,sep=' ', header=None, index_col=([0,1]))
    print(data)

def main():
    
    # Generate edges file
    edges, edges_filename = generate_edges_from_anat(FILENAME, write=False)
    # # GENERATE FEATURES FOR SIGNAL:
    features_filename=edges_filename+'_'+PERT_MAP+'.ft'
    if not os.path.isfile(featurespath+features_filename):
        generate_SIGNAL_features(edges_filename)
    
    # Apply signal and predict SIGNAL score
    SIGNAL_scores_filename=edges_filename+'_'+PERT_MAP+'.sgnl'
    if not os.path.isfile(outputpath+SIGNAL_scores_filename):
        apply_SIGNAL(features_filename)
    print('SIGNAL scores saved to:',SIGNAL_scores_filename)
    
    # Print the results
    print_SIGNAL(SIGNAL_scores_filename)
    # for ANCHOR in ANCHORS:
    #     if not ANCHOR=='TEN1':
    #         print('-------------------PROCESSING ANCHOR', ANCHOR,alias_2geneid[ANCHOR],'------------------------')
    #         with open(INPUT_DIR+FILENAME+'_'+ANCHOR+'.paths', 'rb') as f:
    #             paths_to=pickle.load(f)
    #         id_paths_to = {alias_2geneid[x]:[alias_2geneid[y] for y in path] for (x, path) in paths_to.items()} #assuming there is only one shortest pathway per TERMINAL

    #         # Generate edges-based path dictionary AND edges file:
    #         edges_path_to = generate_edges_dict(id_paths_to, alias_2geneid[ANCHOR], write=False)
    #         for final_node, path in edges_path_to.items():
    #             if final_node!=alias_2geneid[ANCHOR]:
    #                 print('****************************** terminal:', final_node)
    #                 features_filename= str(alias_2geneid[ANCHOR])+'_'+str(final_node)+'.edges_'+PERT_MAP+'.ft.pkl' 
                                    
    #                 # GENERATE FEATURES FOR SIGNAL:
    #                 if not os.path.isfile(featurespath+features_filename):
    #                     features_filename = generate_SIGNAL_features(final_node, features_filename,str(alias_2geneid[ANCHOR]))
                    
    #                 # Apply signal and predict SIGNAL score
    #                 if not os.path.isfile(outputpath+features_filename+'.sgnl'):
    #                     apply_SIGNAL(final_node, alias_2geneid[ANCHOR], features_filename)
                    
    #                 # Print the results
    #                 print_SIGNAL(final_node, features_filename)

if __name__ == '__main__':
    main()
