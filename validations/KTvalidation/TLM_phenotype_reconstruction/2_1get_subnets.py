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
    'network_signing'+os.sep+'SIGNAL Validations target sign reconstruction'\
        +os.sep+'ANAT_validation_pipeline'+os.sep
HOME_DIR='G:'+os.sep+'My Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep+\
    'network_signing'+os.sep+'SIGNAL Validations target sign reconstruction'\
        +os.sep+'ANAT_validation_pipeline'+os.sep
MAIN_DATA_DIR = 'G:'+os.sep+'Il Mio Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep+'Data'+os.sep+SPECIES+os.sep
MAIN_DATA_DIR = 'G:'+os.sep+'My Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep+'Data'+os.sep+SPECIES+os.sep
INPUT_DIR =  HOME_DIR+ '1 Build Pacitaxel target - fosfoprotein paths'
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

    
    if SPECIES=='S_cerevisiae':
        FILENAME='TLM_all_KOterms'
        f=open(HOME_DIR+'1 ANAT Build networks'+os.sep+'Input'+os.sep+'TLManchors.txt')
        lines=[x.strip() for x in f.readlines()]
        ANCHORS=lines
        # ANCHORS=[str(alias_2geneid[node.upper()]) for node in ANCHORS]
        f.close()
        TLMphenotypesdf=pd.read_csv(HOME_DIR+'1 ANAT Build networks'+os.sep+'Input'+os.sep+'TLMphenotypes.csv', header=0, sep='\t')
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
#%% v1 (works with paclitaxel) build paths from anor oto terminal
network_file='1 ANAT Build networks'+os.sep+'ANAT output'+os.sep+FILENAME+'.txt'



# Function to find all paths from "TUBB" to a specific node
def find_paths(graph, current_node, target_node, path, paths_to):
    if current_node == target_node:
        paths_to[target_node]=path
    elif current_node in graph:
        for neighbor in graph[current_node]:
            find_paths(graph, neighbor, target_node, path + [neighbor], paths_to)

# Read the network file and construct the graph
graph = {}
with open(network_file, 'r') as f:
    for line in f:
        edge = line.strip().split()
        if len(edge) == 2:
            source, target = edge
            if source not in graph:
                graph[source] = []
            graph[source].append(target)
            
paths_per_anchor ={}
for ANCHOR in ANCHORS:
    print('fidind path from:', ANCHOR)
    # Find all paths from ANCHOR to each final node
    
    # Initialize the paths_to dictionary
    paths_per_anchor[ANCHOR] = {node: [] for node in TERMS}
    
    for node in TERMS:
        find_paths(graph, ANCHOR, node, [ANCHOR], paths_per_anchor[ANCHOR])

    # Print the paths_to dictionary
    for node, paths in paths_per_anchor[ANCHOR].items():
        print(f"{node}: {paths}")

    # Save file
    with open( HOME_DIR+'2 SIGNAL score networks'+os.sep+FILENAME+'_'+ANCHOR+'.paths','wb') as f:
       pickle.dump(paths_per_anchor[ANCHOR],f)
    
    
#%% v2 with networkx (has revealed the bug with TLM data)
network_file='1 ANAT Build networks'+os.sep+'ANAT output'+os.sep+FILENAME

columns = ['ID1', 'ID2']
# Read the CSV file and select the desired columns and rows
edges_df = pd.read_csv(network_file+'.csv', usecols=columns)
graph_symbols=nx.from_pandas_edgelist(edges_df, source='ID1', target='ID2')
#%%
#TODO networkfile.txt problu nn lo uso piu

#%% check if some nodes in graph are not present in ancohors or terminals
# to do this, also extract the nodes table from cytoscape, and make sure that they
# are all in the edges table as well


# Read the CSV file and select the desired columns and rows
nodesfile=HOME_DIR+'1 ANAT Build networks'+os.sep+'ANAT output'+os.sep+FILENAME+'_nodes.csv'
columns = ['name', 'status','xrefID','xrefName']
nodes_df = pd.read_csv(nodesfile, usecols=columns)

#%% vediamo ora se tutti i nodi di TERMS almeno sono dentro il graph (alcune di ste cells potrebbero ripeters):
print(len(TERMS),len(nodes_df[nodes_df.status=='TERMINAL']))
updatedTERMS=[]
missingTERMS=[]
for genename in list(TLMphenotypesdf[TLMphenotypesdf.columns[0]].values): # per prendere i genename s non tradotti
    if not genename in graph_symbols.nodes:
        missingTERMS.append( genename)
    else:
        updatedTERMS.append(genename)
        
        # mancano cmq molti geni dei temrinals! aggiungere manualmente a ANAT e rilanciare

#%%
paths_per_anchor ={}
for ANCHOR in ANCHORS:
    if ANCHOR!='TEN1': #TEN1, gene cher isultaa non connesso con gli altri nel net, quindi olo leiminiamo
        print('fidind path from:', ANCHOR)
        # Find all paths from ANCHOR to each final node
        
        # save all shortest paths (one more layer of depth to dictionary)
        paths_per_anchor[ANCHOR] = {node: nx.shortest_path(graph_symbols, ANCHOR, node) for node in updatedTERMS}
       
        with open( HOME_DIR+'2 SIGNAL score networks'+os.sep+FILENAME+'_'+ANCHOR+'.paths','wb') as f:
           pickle.dump(paths_per_anchor[ANCHOR],f)
        