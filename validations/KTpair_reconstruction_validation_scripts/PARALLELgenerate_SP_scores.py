# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

all files with 'PARALLEL' in name have server directories

Generates dictionaries for every Knockout of positive(negative) targets shortest path list,
with edges given by SIGNAL's scores'
"""


import os
# HOME_DIR  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# # HOME_DIR  =  "G:" +os.sep+"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# DIRECTED_DIR = HOME_DIR+"network_signing"+os.sep
# os.chdir(DIRECTED_DIR)
import pandas  as pd
import numpy  as np
from datetime import date
from preproc_utils import load_training_data ,graph_from_dataframe, add_edges_from_labels
from train_and_vis import  load_features
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import networkx as nx
import collections
import sys
##############################################################################
#  INPUTS
##############################################################################
# SPECIES = "H_sapiens"#
SPECIES =  "S_cerevisiae" #

# # Servier directories:
# HOME_DIR  =  '/home/bnet/lorenzos/signed/signedv3/'
# INPUT_DIR=HOME_DIR+"input"+os.sep+SPECIES+os.sep
# FEATURESDIR=HOME_DIR+'output'+os.sep+SPECIES+os.sep
# OUTDIR=FEATURESDIR+'validation_out'+os.sep
# BASENETDIR=INPUT_DIR

# Local directories
HOME_DIR  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# HOME_DIR  =  "G:" +os.sep+"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
DATA_DIR = HOME_DIR+'Data'+os.sep+SPECIES+os.sep
SIGNAL_DIR = HOME_DIR+'network_signing'+os.sep
other='SIGNAL Validations target sign reconstruction'
INPUT_DIR=SIGNAL_DIR+'features_and_labels'+os.sep+SPECIES+os.sep

OUTDIR=SIGNAL_DIR+'validation_out'+os.sep+SPECIES+os.sep
FEATURESDIR=INPUT_DIR
IMG_DIR=SIGNAL_DIR+'imgs'+os.sep+'v3'+os.sep+SPECIES+os.sep
BASENETDIR=DATA_DIR

DATE=str(date.today())

if SPECIES == 'S_cerevisiae':
    datasets = ['kegg','kpi','ubiq']
else:
    datasets = ['kegg','PSP','depod','ubinet2']
dataset_name = '_'.join(datasets)
pert_map = 'Holstege'#'ADPBH_1' #'ADPBH_1000'#'reimand' ##  human: 'ADPBH_1 'ADPBH_1000'  'K562gwps' 


# load knockdowns and targets
TSIGN = 'MINUS'#sys.argv[1]#
if TSIGN=='PLUS':
    with open(INPUT_DIR+'plus_targets'+'_'+pert_map+'.pkl', 'rb') as f:
        targets_of_deletion=pickle.load( f)
if TSIGN =='MINUS':
    with open(INPUT_DIR+'minus_targets'+'_'+pert_map+'.pkl','rb') as f:
        targets_of_deletion=pickle.load(f)
print('sources:', len(list(targets_of_deletion.keys())),'targets: ',TSIGN)
# print('train classifier' )

training_labels, training_labels_weights = load_training_data(INPUT_DIR, datasets, SPECIES)
training_features_table = load_features(FEATURESDIR, [i+'_'+pert_map for i in datasets], SPECIES)
training_labels=pd.concat(list(training_labels.values())) # works because python dictionaries are ordered in python3
# training_labels=training_labels.iloc[1:] #temp solo x ADPBH_1 e 1000 (vedi log 30/01/2023)
training_features_table= pd.concat(list(training_features_table.values()))
#%%
# from glob_vars import SPECIES, PERT_MAP, TRAIN_DATA, HOME_DIR, LBL_DIR,FT_DIR,\
#     PRT_DIR,DICT_DIR,DICT_FILE, NET_DIR,ANAT_VAL_SIGNAL_OUT,  get_terms, get_anchors,\
#     EDGES_DIR,  SIGNAL_DIR, MOD_DIR, KO_VAL_HOME, KO_VAL_OUT,\
#     ANAT_VAL_HOME, ANAT_VAL_SIGNAL_OUT
# refactored inputs for generate_SP_scores KT variant
# # load knockdowns and targets
# TSIGN = 'MINUS'#sys.argv[1]#
# if TSIGN=='PLUS':
#     with open(PRT_DIR+'plus_targets'+'_'+PERT_MAP+'.pkl', 'rb') as f:
#         targets_of_deletion=pickle.load( f)
# if TSIGN =='MINUS':
#     with open(PRT_DIR+'minus_targets'+'_'+PERT_MAP+'.pkl','rb') as f:
#         targets_of_deletion=pickle.load(f)
# print('sources:', len(list(targets_of_deletion.keys())),'targets: ',TSIGN)
# # print('train classifier' )

# dataset_name = '_'.join(TRAIN_DATA)
# training_labels, training_labels_weights = load_training_data(LBL_DIR, TRAIN_DATA, SPECIES)
# training_features_table = load_features(FT_DIR, [i+'_'+PERT_MAP for i in TRAIN_DATA], SPECIES)
# training_labels=pd.concat(list(training_labels.values())) # works because python dictionaries are ordered in python3
# # training_labels=training_labels.iloc[1:] #temp solo x ADPBH_1 e 1000 (vedi log 30/01/2023)
# training_features_table= pd.concat(list(training_features_table.values()))
#%%

BASE_GRAPH=graph_from_dataframe(BASENETDIR, SPECIES)
for i, data in training_labels_weights.items():
    BASE_GRAPH = add_edges_from_labels(BASE_GRAPH,data)
    
CHUNKNAMES=['netedgesI','netedgesII','netedgesIII','netedgesIV','netedgesV','netedgesVI']
#print('Load edges (indexes) and features (columns) for base net')
netind={}
for chunk in CHUNKNAMES:
    with open(INPUT_DIR+chunk,'rb') as f:
        netind[chunk]=pickle.load(f)

#print('find shortest path from source to term, then predict proba for edges in that path')

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

def load_features_for(edge): 
    for chunkname in CHUNKNAMES:
        if edge in netind[chunkname]:
            #'opening pickled file and only keep one line'
            with open(INPUT_DIR+chunkname+'_'+pert_map+'.ft.pkl','rb') as f:
                net_features_table=pickle.load(f)
            return net_features_table.loc[edge]

def get_features_for_sp_edges(edges_path, source):
    valid_features=[]
    for edge in edges_path:
        fts = load_features_for(edge)
        if type(fts)==pd.Series: #some edges may be absent from table
            valid_features.append(fts) #list of series of nfeatures length
    shortest_path_features=pd.DataFrame(valid_features)
    if not shortest_path_features.shape[0] ==0:
        return shortest_path_features.drop([str(source)+i for i in ['+','-']], axis=1) #dataframe of features of the shortest path edges, minus the features of the original knocout
    else:     # if ALL edges are absent from table (happens with 1-edge SPs:)
        return shortest_path_features
#%% nodes scores

def edges_to_nodes(edges_list):
    temp=list(zip(*edges_list))
    nodes_list=list(temp[0])
    nodes_list.append(temp[1][-1])
    return nodes_list

def get_outedges_SIGNER_score_for_nodes(nodes_list, source,outedges_signer_scores_for_pathway, seen_nodes):
     # outedges_signer_scores_for_pathway is an ordered dictionary of 'NODEid':[edgesignerscore1, edgesignerscore2]
    
    def load_Signerscore_for_edge(edge):
        for chunkname in CHUNKNAMES:
            if edge in netind[chunkname]:
                with open(OUTDIR+'KO-'+str(source)+'-'+chunkname+'_'+dataset_name+'_'+pert_map+'.pred','rb') as f:
                    SIGNERtable=pickle.load(f) #dataframe with '+' and '-' columns, and edges as indexes
                return SIGNERtable
        
        # raise ValueError('WARNING: EDGE DOES NOT EXIST:', edge)
        
        
    
    for node in nodes_list:
        allscorespernode=[]
        if not node in seen_nodes.keys():
            ons=list(BASE_GRAPH.successors(node))
            
            # print('neighoburs of ',node,':',len(ons))
            for j in ons:
                edge=(node,j)
                signertable=load_Signerscore_for_edge(edge)
                if type(signertable)==pd.DataFrame: #some edges not present
                    allscorespernode.append(signertable.loc[edge]['-']) #only take the negative probability as SIGNER. TODO: update SIGNER putput as negative probability.
            
            seen_nodes[node]=allscorespernode
        else:
            allscorespernode.append(seen_nodes[node])       
    
        if not len(allscorespernode)==0: 
            outedges_signer_scores_for_pathway[node]=allscorespernode
        else:# unluky hypothesis that no edges are present for that node
            outedges_signer_scores_for_pathway[node]=None

    return outedges_signer_scores_for_pathway, seen_nodes


# def get_features_for_sp_nodes(nodes_list, source):
#     'every NODE is now a COLLECTION Of edgefeatures of its outneighbours'
#     valid_features_of={}
    
#     for node in nodes_list:
#         # generate a feature table for all outedges for the given node
#         ons=BASE_GRAPH.successors(node) #list of out neighbours of i
#         valid_nodefeatures=[]
#         for outnode in ons:
#            fts = load_features_for((node, outnode)) 
#            if type(fts)==pd.Series: #some edges may be absent from table
#                valid_nodefeatures.append(fts)  #list of series of nfeatures length
        
#         shortest_path_features=pd.DataFrame(valid_nodefeatures)
#                 #DROP THE NANSSS
#         if not shortest_path_features.shape[0] ==0:
#              shortest_path_features.drop([str(source)+i for i in ['+','-']], axis=1, inplace=True) #dataframe of features of the shortest path edges, minus the features of the original knocout
        
        
#         valid_features_of[node] = shortest_path_features
#     return valid_features_of
#     # else:     # if ALL edges are absent from table (happens with 1-edge SPs:)
#     #     return shortest_path_features

def function(source, targets, scoring_type='edge'):
    '''
    scoring_type: str ['edge', 'node']. Assign scores to edges or calculate a
        node score first and assign score to the node.
    '''
    print('--- training classifier without KO:',source,'and with targets:', targets)
    
    #TEMP levato step 1 for testing
    # 1. train classifier with all features minus current KO:
    temp_training_features_table = training_features_table.drop([str(source)+i for i in ['+','-']], axis=1) 
    classifier=RandomForestClassifier() 
    classifier.fit(StandardScaler().fit_transform(temp_training_features_table), training_labels) 

    #2. find shortest path between KO and its targets:
        # edge based:
    if scoring_type == 'edge':
        avg_edge_score_per_sp_of_target={}
        edge_score_per_sp_of_target={}
         # node-based-scoring
    if  scoring_type == 'node':
        node_scores_per_shortest_path_per_target = collections.defaultdict(list) #dictionary of lists of dictionaries of lists. As god intended. (will be iterated over all sources and kept in a further level of dictionary)
        seen_nodes={} # for node-based-scoring. to keep track of already seen nodes
    
    for target in targets:
        if target!= source: 
            paths = nx.all_shortest_paths(BASE_GRAPH, source, target) 
            edges_lists=[generate_edges_from_path(path) for path in paths]
            print('>t', target,'#shortest paths:',len(edges_lists),'len shortest paths:',len(edges_lists[0]))
            
            # 3. extract its features from the base net feature table 
            # (BIG FILE! divided into 6 chunks)
            # and apply to those
            # spf_list=[]
            sppred_list=[]
            for edges_path in edges_lists:
                if scoring_type == 'node': #implement
                    
                    nodes_path=edges_to_nodes(edges_path) #list of nodes
                    current_sp_nodes_scores = {}
                    current_sp_nodes_scores, seen_nodes = get_outedges_SIGNER_score_for_nodes(nodes_path, source, current_sp_nodes_scores, seen_nodes)
                    
                    node_scores_per_shortest_path_per_target[target].append(current_sp_nodes_scores) #because every pair has multiple paths
                    
                
                else: #Edge based scores
                    shortest_path_features=get_features_for_sp_edges(edges_path, source)
                    # only save shortestpath feature tables with full edges (some edges may be missing from feature table)
                    if shortest_path_features.shape[0] == len(edges_path): 
                        predictions=classifier.predict_proba(StandardScaler().fit_transform(shortest_path_features))
                        sppred_list.append(predictions)
                
                
            # predictions sono le predictions di ogni edge del mio shortest path
            if scoring_type == 'edge':
                avg=np.mean(np.array(sppred_list), axis=0) # the mean over all positions of the path (i.e if there are 3 paths of 10 elements, the mean will be between the three paths for each fo the 10 elements)
            
                avg_edge_score_per_sp_of_target[target] = avg
                edge_score_per_sp_of_target[target] = sppred_list
                #
        else:
                print(source,target,'same node')
    if scoring_type == 'edge':
        return avg_edge_score_per_sp_of_target, edge_score_per_sp_of_target, source
    else:
        print('Writing KO', source)
        with open(OUTDIR+os.sep+TSIGN+'_KO-'+str(source)+'-SP_nodes_per_source_target_pair_'+dataset_name+'_'+pert_map+'.pkl', 'wb') as f:
            pickle.dump(node_scores_per_shortest_path_per_target,f)
        print(source ,'DONE')
        return node_scores_per_shortest_path_per_target, source, seen_nodes
# #%% TESTINGG
# source, targets = list(targets_of_deletion.items())[0]
# node_scores_per_shortest_path_per_target, source, seen_nodes= function(source,targets,'node')


# #%% Iterative version (ONLY FOR TESTING)
# print('iterative test')
# asd=[]
# sources=[]
# for source, targets in list(targets_of_deletion.items())[:2]:
#     print(source)
#     sources.append(source)
#     node_scores_per_shortest_path_per_target, source, seen_nodes=function(source, targets, 'node') 
#     asd.append(node_scores_per_shortest_path_per_target)
# print('end')
#%%
N_JOBS=64
SCORE='node'
from joblib import Parallel, delayed

packed_results = Parallel(n_jobs=N_JOBS)(delayed(function)\
                            (source, targets, SCORE)\
                                for source, targets in list(targets_of_deletion.items()))

#%%
if SCORE=='edge':
    Lavg_edge_score_per_sp_of_target, Ledge_score_per_sp_of_target, Lsources =zip(*packed_results)
    avg_edge_score_per_sp_of = {}
    edge_score_per_sp_of = {}
    for i, source in enumerate(Lsources):
        avg_edge_score_per_sp_of[source]= Lavg_edge_score_per_sp_of_target[i]
        edge_score_per_sp_of[source]= Ledge_score_per_sp_of_target[i]
    with open(OUTDIR+os.sep+TSIGN+'_avg_SP_edges_per_source_target_pair_'+dataset_name+'_'+pert_map+'.pkl', 'wb') as f:
        pickle.dump(avg_edge_score_per_sp_of,f)
    with open(OUTDIR+os.sep+TSIGN+'_SP_edges_per_source_target_pair_'+dataset_name+'_'+pert_map+'.pkl', 'wb') as f:
        pickle.dump(edge_score_per_sp_of,f)

else:
    print('new version, saving one dictionary per KO')
    # Lnode_scores_per_shortest_path_per_target, Lsource, Lseen_nodes = zip(*packed_results)
    # node_scores_per_shortest_path_per_target_per_ko = {} #dictionary of dictionaries of list of dictionaries of list. As God indented.
    # for i, source in enumerate(Lsource):
    #     if len(Lnode_scores_per_shortest_path_per_target[i])>0:
    #         node_scores_per_shortest_path_per_target_per_ko[source]=Lnode_scores_per_shortest_path_per_target[i] #all while assuming this process is made by ordered things. TODO BIG TODO: CHECK THAT IT IS TRUE! IT mya not be, although yes cos they re produced at the same time
    # with open(OUTDIR+os.sep+TSIGN+'_SP_nodes_per_source_target_pair_'+dataset_name+'_'+pert_map+'.pkl', 'wb') as f:
    #     pickle.dump(node_scores_per_shortest_path_per_target_per_ko,f)
#%%
