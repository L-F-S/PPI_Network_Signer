# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

Generates dictionaries for every TLM anchor-terminal shortest path list, divided by phenotype
with edges given by SIGNAL's scores'
essentially taken from /ANAT_validation_pipeline/scripts/3scorenetworks_TLM.ipynb
"""


import os
HOME_DIR  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# HOME_DIR  =  "G:" +os.sep+"Il Mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# HOME_DIR  =  "G:" +os.sep+"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
DIRECTED_DIR = HOME_DIR+"network_signing"+os.sep
os.chdir(DIRECTED_DIR)
import pandas  as pd
import numpy  as np
import itertools
from datetime import date
from preproc_utils import load_training_data ,graph_from_dataframe, add_edges_from_labels
from train_and_vis import  load_features
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import networkx as nx
import collections
import sys
from glob_vars import SPECIES, PERT_MAP, TRAIN_DATA, HOME_DIR, LBL_DIR,FT_DIR,\
    PRT_DIR,DICT_DIR,DICT_FILE, NET_DIR,ANAT_VAL_SIGNAL_OUT,ANAT_VAL_HOME,  get_terms, get_anchors,\
    EDGES_DIR,  SIGNAL_DIR, MOD_DIR, KO_VAL_HOME, KO_VAL_OUT,\
    ANAT_VAL_HOME, ANAT_VAL_SIGNAL_OUT

##############################################################################
#  INPUTS 
##############################################################################

dataset_name = '_'.join(TRAIN_DATA)

NET_FILE = SPECIES+'_'+dataset_name+'.nx'
with open(NET_DIR+NET_FILE, 'rb') as f:
    BASE_GRAPH=pickle.load(f) 

val_data_name='TLM_all_KOterms'#'TLM_strong_normal'#
# #v2comment out this part:
SIGNAL_net_name=val_data_name+'.edges_'+PERT_MAP+'.sgnl' #signal nets are csv files
SIGNAL_scores = pd.read_csv(ANAT_VAL_SIGNAL_OUT+SIGNAL_net_name, sep=' ', header=None, names=['id1','id2','SIGNAL'])
ALL_GENES = np.unique(list(SIGNAL_scores['id1'])+list(SIGNAL_scores['id2']))
#%% preoprocess SIGNAL_scores
# filter anchorsand terms
TERMS=get_terms(termfile=val_data_name+'_phenotype')
ANCHORS=get_anchors()
with open( DICT_DIR+DICT_FILE, 'rb') as f:
    alias_2geneid = pickle.load(f)

anchor_to_skip=[]
for anchor in ANCHORS:
    anchorid=alias_2geneid[anchor] 
    #v2dont care if geenes are not in anat net, cos im looking across whole network
    if not anchorid in ALL_GENES:
        anchor_to_skip.append(anchor)
# v2: remove this
t=0
TERMS2=[]
for term in TERMS:
    termid=alias_2geneid[term]
    if not termid in ALL_GENES:
        print(term, termid)
    else:
        t+=1
        
        TERMS2.append(term)
TERMS=TERMS2 
to_remove=[]
for term in TERMS:
    if term in ANCHORS:
        to_remove.append(term)
TERMS3=[i for i in TERMS if not i in to_remove]
TERMS=TERMS3
# map to anchor and term 
id2name = {alias_2geneid[gene]:gene for gene in ANCHORS+TERMS}

def renamegene(x):
    if x in id2name.keys():
        return id2name[x]
    return x

def label(x):
    if x in ANCHORS:
        return('A')
    if x in TERMS:
        return('T')
    else:
        return('O')
#v2  remove elow lines:
# SIGNAL_scores['name1'] = SIGNAL_scores['id1'].apply( lambda x: renamegene(x))
# SIGNAL_scores['name2'] = SIGNAL_scores['id2'].apply( lambda x: renamegene(x))
# SIGNAL_scores['name1lbl'] = SIGNAL_scores['name1'].apply( lambda x: label(x))
# SIGNAL_scores['name2lbl'] = SIGNAL_scores['name2'].apply( lambda x: label(x))
#%% extract signal values for all shortest paths
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


def find_SP_dicts(SIGNAL_net, anchors, terms):

    SIGNAL_of = {}
    SIGNAL_of_avg = {}

    for anchor in anchors:
        if not anchor in anchor_to_skip: # disconnected from net (TEN1  for all data, EST2 only valid with TLM strong normal data)
            for term in terms:
                print('anchor',anchor,'term',term)
                if nx.has_path(SIGNAL_net, anchor, term):
                    paths = list(nx.all_shortest_paths(SIGNAL_net, anchor, term))
                
                            
                    # Extract SIGNAL values for all paths
                    allSIGNALS=np.zeros((len(paths),len(paths[0])-1))
                    for i, path in enumerate(paths):
                        edges = generate_edges_from_path(path)
                        SIGNALs=[SIGNAL_net.edges[edge]['SIGNAL'] for edge in edges]
                        allSIGNALS[i]=SIGNALs
                        
                    if type(anchor)==int: # anchor and terms are in id format (works with base network), translate for future readability
                        anchor_genename=id2name[anchor]
                        term_genename=id2name[term]
                    else:
                        anchor_genename=anchor
                        term_genename=term
                        
                    SIGNAL_of[(anchor_genename,term_genename)]= allSIGNALS 
                    avgSIGNALS=allSIGNALS.mean(axis=0) # take per-edge mean of all paths
                    SIGNAL_of_avg[(anchor_genename,term_genename)]= avgSIGNALS
    return SIGNAL_of, SIGNAL_of_avg


#%% run with ANAT TLM SIGNAL network   v2: non lo faccio piu xke uso il full base net only      
# ANAT_net = nx.from_pandas_edgelist(SIGNAL_scores, 'name1','name2','SIGNAL') # turn to network
# SIGNAL_of, SIGNAL_of_avg = find_SP_dicts(ANAT_net, ANCHORS, TERMS)
#%% ..or run with SIGNAL full base network 
all_chunks=[]
for i in range(6):
    all_chunks.append( pd.read_csv(SIGNAL_DIR+'net'+str(i+1)+'.edges_Kemmeren.sgnl', header=None, sep=' ',names=['id1','id2','SIGNAL']))
SIGNAL_base_df=pd.concat(all_chunks)
SIGNAL_base_net =  nx.from_pandas_edgelist(SIGNAL_base_df, 'id1','id2','SIGNAL') 
SIGNAL_of, SIGNAL_of_avg = find_SP_dicts(SIGNAL_base_net, {alias_2geneid[gene] for gene in ANCHORS}, {alias_2geneid[gene] for gene in TERMS})

#%%  29?10?2-24Find and save shortest paths with node names (only greater than 100)


def find_SP(SIGNAL_net, anchors, terms, nSPS):
    '''Save dictinoary of (anchor gene name, term gene name): [paths as geneids (including anchor and term gene id)]'''

    SPs_of = collections.defaultdict(list)

    for anchor in anchors:
        if not anchor in anchor_to_skip: # disconnected from net (TEN1  for all data, EST2 only valid with TLM strong normal data)
            print('anchor',anchor)
            for term in terms:
                pair=(id2name[anchor], id2name[term])
                if nx.has_path(SIGNAL_net, anchor, term):
                    paths = list(nx.all_shortest_paths(SIGNAL_net, anchor, term))
                    
                    if len(paths)>nSPS:
                        SPs_of[pair]=paths

                    # for i, path in enumerate(paths):
                    #     print(path)
                        
    return SPs_of

nSPS=100
SPs_dict=find_SP(SIGNAL_base_net, {alias_2geneid[gene] for gene in ANCHORS}, {alias_2geneid[gene] for gene in TERMS}, nSPS)
with open(KO_VAL_OUT+os.sep+val_data_name+'_SPs_gene_ids'+'.pkl', 'wb') as f:
    pickle.dump(SPs_dict,f)
#%% load and preprocess labels

TLMphenotypesdf=pd.read_csv(ANAT_VAL_HOME+'1 ANAT Build networks'+os.sep+'Input'+os.sep+val_data_name+'_phenotype_labelled.txt', header=0, sep='\t')
TLMphenotypesdf =  TLMphenotypesdf[TLMphenotypesdf['Gene (corrected for NGE)'].isin(TERMS)]
def simplify(x):
    if x==('DAmP Long'):
        return 'DAmP L'
    if x==('DAmP Short'):
        return 'DAmP S'
    if x=='VL'or x =='L' or x == 'VS' or x == 'S':
        return x
    if x == 'Long':
        return 'L'
    if x == 'long':
        return 'L'
    if x == 'slightly long' or x=='sl':
        return 'SL'
    if x == 'Short' or x =='short':
        return 'S'
    if x=='slightly short' or x=='ss':
        return 'SS'
    else:
        print(x,'not in labels')
TLMphenotypesdf['pht']=TLMphenotypesdf['Telomere Phenotype*'].apply(lambda x: simplify(x))
all_long_tags=['VL','L','SL', 'DAmP L']
all_short_tags=['VS','S','SS', 'DAmP S']
#%% enlarge dictionary to match Reconstuct_KT_pairs required input, and split based on label

# def label_reshape(SIGNAL_of, flag):
#     SIGNAL_of_2_L = {}
#     SIGNAL_of_2_S = {}
#     for pair, SP in SIGNAL_of.items():
#         anchor=pair[0]
#         term=pair[1]
        
#         label=TLMphenotypesdf['pht'][TLMphenotypesdf['Gene (corrected for NGE)']==term].iloc[0]
        
#         if  label.endswith(flag):
            
#             # build a nested dictionary by splitting each A-T key in a dictionary of Ts: SPs p per every A
#             if not anchor in SIGNAL_of_2_L.keys():
#                 SIGNAL_of_2_L[anchor] = {}
    
#             SIGNAL_of_2_L[anchor][term] = SP
            
            
#         else:
            
#             if not anchor in SIGNAL_of_2_S.keys():
#                 SIGNAL_of_2_S[anchor] = {}
#             SIGNAL_of_2_S[anchor][term] = SP
#     return SIGNAL_of_2_L, SIGNAL_of_2_S
def label_reshape(SIGNAL_of, lflags, sflags): #v2, specific label
    SIGNAL_of_2_L = {}
    SIGNAL_of_2_S = {}
    for pair, SP in SIGNAL_of.items():
        anchor=pair[0]
        term=pair[1]
        
        label=TLMphenotypesdf['pht'][TLMphenotypesdf['Gene (corrected for NGE)']==term].iloc[0]
        
        if  label in lflags:
            
            # build a nested dictionary by splitting each A-T key in a dictionary of Ts: SPs p per every A
            if not anchor in SIGNAL_of_2_L.keys():
                SIGNAL_of_2_L[anchor] = {}
    
            SIGNAL_of_2_L[anchor][term] = SP
            
        if label in sflags:
            
            if not anchor in SIGNAL_of_2_S.keys():
                SIGNAL_of_2_S[anchor] = {}
            SIGNAL_of_2_S[anchor][term] = SP
            
    return SIGNAL_of_2_L, SIGNAL_of_2_S
lflags=['L']
sflags=['S']
SIGNAL_of_2_L, SIGNAL_of_2_S=label_reshape(SIGNAL_of, lflags,sflags)
#%%
for i in range(len(all_short_tags)):
    # every round of i is a different combination.
    for short_tags, long_tags in zip(itertools.combinations(all_short_tags, i+1), itertools.combinations(all_long_tags, i+1)):
        print(short_tags)
        SIGNAL_of_2_L, SIGNAL_of_2_S=label_reshape(SIGNAL_of, long_tags,short_tags)
        spname_short='_SP_SIGNALv2_'+'_'.join(short_tags)
        spname_long='_SP_SIGNALv2_'+'_'.join(long_tags)

        with open(KO_VAL_OUT+os.sep+val_data_name+spname_long+dataset_name+'_'+PERT_MAP+'.pkl', 'wb') as f:
            pickle.dump(SIGNAL_of_2_L,f)
        with open(KO_VAL_OUT+os.sep+val_data_name+spname_short+dataset_name+'_'+PERT_MAP+'.pkl', 'wb') as f:
            pickle.dump(SIGNAL_of_2_S,f)
#%% open and check data
sp_name='_SP_SIGNALv2_'#'_all'
all_long_tags=['VL','L','SL', 'DAmP L']
all_short_tags=['VS','S','SS', 'DAmP S']

#%% Print TLM data
stats_of={}
print('mean # SPs:','max # SPs:', 'min #SPs:',  'tot #SPs:')
for tagl, tags in zip(all_long_tags, all_short_tags):
    spname_short='_SP_SIGNALv2_'+tags
    spname_long='_SP_SIGNALv2_'+tagl
    
    with open(KO_VAL_OUT+os.sep+val_data_name+spname_short+dataset_name+'_'+PERT_MAP+'.pkl', 'rb') as f:
        TLMS=pickle.load(f)
    with open(KO_VAL_OUT+os.sep+val_data_name+spname_long+dataset_name+'_'+PERT_MAP+'.pkl', 'rb') as f:
        TLML=pickle.load(f)
    def get_stats(TLM_dict, tag):
        max_n_paths=0
        mean_n_paths=-1
        min_n_paths=10000
        tot_n_paths=[]
        for (anchor, path_of_term) in TLM_dict.items():
            for (term, paths) in path_of_term.items():
                n_paths=len(paths)
                if n_paths>max_n_paths:
                    max_n_paths=n_paths
                if n_paths<min_n_paths:
                    min_n_paths=n_paths
                tot_n_paths.append(n_paths)
        mean_n_paths=np.mean(tot_n_paths)
        print(tag, np.round(mean_n_paths,2),max_n_paths,min_n_paths, len(tot_n_paths))
        return mean_n_paths, max_n_paths, min_n_paths
    stats_of[tagl]=get_stats(TLML, tagl)
    stats_of[tags]=get_stats(TLMS, tags)
