# -*- coding: utf-8 -*-
"""
Reformats ANAT network output

python 2get_subnets.py terminal1 terminaln ... anchor ANATfilenameWithoutextension

creates a .path pkl file of a dictionary of paths per terminal
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
import sys



SPECIES = "S_cerevisiae"
# SPECIES='H_sapiens'
DATE=str(date.today().strftime("%d/%m/%Y"))
HOME_DIR='G:'+os.sep+'Il Mio Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep+\
    'network_signing'+os.sep+'Validations'+os.sep+'ANAT_validation_pipeline'+os.sep
# HOME_DIR='G:'+os.sep+'My Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep+\
#     'network_signing'+os.sep+'Validations'+os.sep+'ANAT_validation_pipeline'+os.sep
MAIN_DATA_DIR = 'G:'+os.sep+'Il Mio Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep+'Data'+os.sep+SPECIES+os.sep
# MAIN_DATA_DIR = 'G:'+os.sep+'My Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep+'Data'+os.sep+SPECIES+os.sep
INPUT_DIR = HOME_DIR+'1 ANAT Build networks'+os.sep+'Input'+os.sep
os.chdir(HOME_DIR)

with open( MAIN_DATA_DIR+'alias_2geneid.pkl', 'rb') as f:
    alias_2geneid = pickle.load(f)

with open( MAIN_DATA_DIR+'symbol_2geneid.pkl', 'rb') as f:
    symbol_2geneid = pickle.load(f)
id_2alias={y:x for (x,y) in symbol_2geneid.items()}
# network=pd.read_csv(MAIN_DATA_DIR +SPECIES+'.net',sep="\t", header=None)
# graph = nx.from_pandas_edgelist(network.reset_index(), 0, 1, 2)
#%%
if len(sys.argv)==1:

    
    FILENAME='TLM_strong_normal'#'TLM_all_KOterms'#
    f=open(INPUT_DIR+'TLManchors.txt')
    lines=[x.strip() for x in f.readlines()]
    ANCHORS=lines
    f.close()
    f=open(INPUT_DIR+'TLM_strong_normal_phenotype.txt')
    lines=[x.strip() for x in f.readlines()]
    TERMS=lines
    f.close()
    # TLMphenotypesdf=pd.read_csv(HOME_DIR+'1 ANAT Build networks'+os.sep+'Input'+os.sep+'TLMphenotypesnodupes.txt', header=0, sep='\t')
    # TERMS=list(TLMphenotypesdf[TLMphenotypesdf.columns[0]].values)

else:
    args = sys.argv[1:]
    TERMS = []
    for arg in args[:-2]:
        TERMS.append(arg)
    ANCHORS =args[-2]
    FILENAME =args[-1]
    
#%% check if some nodes in graph are not present in ancohors or terminals
network_file='1 ANAT Build networks'+os.sep+'ANAT output'+os.sep+FILENAME

columns = ['ID1', 'ID2']
# Read the CSV file and select the desired columns and rows
edges_df = pd.read_csv(network_file+'.csv', usecols=columns)
graph_symbols=nx.from_pandas_edgelist(edges_df, source='ID1', target='ID2')

# to do this, also extract the nodes table from cytoscape, and make sure that they
# are all in the edges table as well

# Read the CSV file and select the desired columns and rows
nodesfile=HOME_DIR+'1 ANAT Build networks'+os.sep+'ANAT output'+os.sep+FILENAME+'_nodes.csv'
columns = ['name', 'status','xrefID','xrefName']
nodes_df = pd.read_csv(nodesfile, usecols=columns)
print('checking cytoscape nodes against  cytoscape edges using symbols')
disc_nodes=[]
for genename in nodes_df.xrefName:
    if not genename in graph_symbols.nodes:
        print('disconnected node:', genename)
        disc_nodes.append(genename)
#%%double check: remove disconnected terms in ANAT from TERM variable

updatedTERMS=[]
missingTERMS=[]
for genename in TERMS: 
    if not genename in graph_symbols.nodes:
        missingTERMS.append( genename)
    else:
        updatedTERMS.append(genename)
#%% doublecheck: check term nodes that are also anchors 
print(len(TERMS),len(nodes_df[nodes_df.status=='TERMINAL']))
difference = set(TERMS).difference(set(nodes_df['name'][nodes_df.status=='TERMINAL']))
print(difference)          
for node in difference:
    print(node, nodes_df['status'][nodes_df['name']==node])
#%%
paths_per_anchor ={}
for ANCHOR in ANCHORS:
    if not ANCHOR in disc_nodes:
        print('fidind path from:', ANCHOR)
        # Find all paths from ANCHOR to each final node
        
        # save all shortest paths (one more layer of depth to dictionary)
        path_per_term_per_anchor = {}
        for node in updatedTERMS:
            try: #paths may not exist if there are disconnected subnets
                path_per_term_per_anchor[node] = nx.shortest_path(graph_symbols, ANCHOR, node)
            except:
                continue
    
        paths_per_anchor[ANCHOR] = path_per_term_per_anchor
       
        with open( HOME_DIR+'2 SIGNAL score networks'+os.sep+FILENAME+'_'+ANCHOR+'.paths','wb') as f:
           pickle.dump(paths_per_anchor[ANCHOR],f)
        