# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 21:48:03 2023

@author: L-F-S

    -1 re map genes names to 1) ANCHORS, 2) TERMINALS 3) OTHER
    -2 Make subnetworks of Anchor - terminals (it can be 1 : 1, one term to all anchors, or 1 anchor to all same labeled terminals (MAKES the msot sense, start with this)
        currently are subpaths of all_terms
    -3 LABEL PHENOTYPE OF SUbnetworks based on terminals
    -3.5 Plot them on cytoscape adding THE SIGNAL information and the FEATURE information

    -4 obtain a score
    -5 CV on that score to the expected labels
    
    
"""

import os
import sys
HOME_DIR='G:'+os.sep+'My Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep+'network_signing'+os.sep
sys.path.append(HOME_DIR)
from glob_vars import SPECIES, TRAIN_DATA, PERT_MAP, LBL_DIR,EDGES_DIR,\
    FT_DIR, SIGNAL_DIR, MOD_DIR, PRT_DIR, NET_DIR,  NET_FILE,\
    PROPAGATE_ALPHA, PROPAGATE_EPSILON ,PROPAGATE_ITERATIONS, DICT_DIR,\
    DICT_FILE, ANAT_VAL_DIR, ANAT_EXPERIMENTAL_INPUT_DIR, ANAT_VAL_OUT, get_terms, get_anchors, ANAT_VAL_HOME
import pickle
import collections
import itertools
import lxml.etree as etree
import os
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from applySIGNAL import predictSIGNAL
from joblib import Parallel, delayed


with open( DICT_DIR+DICT_FILE, 'rb') as f:
    alias_2geneid = pickle.load(f)
geneid2alias ={y:x for (x,y) in alias_2geneid.items()}

with open(NET_DIR+NET_FILE, 'rb') as f:
    basenet=pickle.load(f)

FILENAME = 'TLM_all_KOterms.edges_Kemmeren._patkar_kegg_kpi_ubinet2_Kemmeren.sgnl'#'TLM_all_KOterms_850696'#
SIGNAL_scores=pd.read_csv(SIGNAL_DIR+FILENAME, sep=' ', header=None, names=['id1','id2','SIGNAL'])

ALL_GENES = np.unique(list(SIGNAL_scores['id1'])+list(SIGNAL_scores['id2']))
len(ALL_GENES)

#%%
TERMS=get_terms()
ANCHORS=get_anchors()
print(len(TERMS), len(ANCHORS))

dupes = []
TERMSid = []
uniqueTERMS=[]

for gene in TERMS:
    if not alias_2geneid[gene] in TERMSid:
        TERMSid.append(alias_2geneid[gene])
        uniqueTERMS.append(gene)
    else:
        dupes.append(gene)
ANCHORSid = [alias_2geneid[x] for x in ANCHORS]
#%%
id2name = {alias_2geneid[gene]:gene for gene in ANCHORS+TERMS}

for anchorid in ANCHORSid:
    if not anchorid in ALL_GENES:
        print(anchorid)
     
t=0
TERMSid2=[]
for term in TERMSid:
    if not term in ALL_GENES:
        print(term)
    else:
        t+=1
        TERMSid2.append(term)
print(t)
TERMSid=TERMSid2
#%%
# id2name = {alias_2geneid[gene]:gene for gene in ANCHORS+TERMS}
# def renamegene(x):
#     if x in id2name.keys():
#         return id2name[x]
#     return x

# def label(x):
#     if x in ANCHORS:
#         return('A')
#     if x in TERMS:
#         return('T')
#     else:
#         return('O')

# SIGNAL_scores['name1'] = SIGNAL_scores['id1'].apply( lambda x: renamegene(x))
# SIGNAL_scores['name2'] = SIGNAL_scores['id2'].apply( lambda x: renamegene(x))
# SIGNAL_scores['name1lbl'] = SIGNAL_scores['name1'].apply( lambda x: label(x))
# SIGNAL_scores['name2lbl'] = SIGNAL_scores['name2'].apply( lambda x: label(x))
#%%

TLMphenotypesdf=pd.read_csv(ANAT_VAL_HOME+'1 ANAT Build networks'+os.sep+'Input'+os.sep+'TLMphenotypesnodupesBACKUP.txt', header=0, sep='\t')

## Drop TERMS that are anchors as well:
for i, row in TLMphenotypesdf.iterrows():
    if row['Gene (corrected for NGE)'] in ANCHORS:
        TLMphenotypesdf.drop(i, inplace=True)

# Translate

TLMphenotypesdf['id'] = TLMphenotypesdf['Gene (corrected for NGE)'].apply(lambda x: alias_2geneid[x])

# add labels to predict
# 0 = short
# 1 = long
def apply_label_stringent(x):
    if x.startswith('DAmP'):
        return None
    if x.startswith('slightly'):
        return None
    if x == 'ss' or x == 'sl':
        return None
    if x.lower().startswith('l'):
        return 1
    if x == 'VL':
        return 1
    if x.lower().startswith('s'):
        return 0
    if x=='VS':
        return 0
    
TLMphenotypesdf['label'] = TLMphenotypesdf['Telomere Phenotype*'].apply(lambda x: apply_label_stringent(x))
# see dupes:
dupes=[]
moving=[]
for i, row in TLMphenotypesdf.iterrows():
    idd =row['id']
    if not idd in moving:
        moving.append(idd)
    else:
        dupes.append(idd)
        print(idd,row.label)
# TEMP xke tlm fenotipes e' un file dimmerda pieno di duplicati
for x in dupes:
    print(TLMphenotypesdf[TLMphenotypesdf['id']==x])
    
TLMphenotypesdf.drop([19,126,74,164,8,73,368,276,424,271], inplace=True)
#%%
# pretty stringent
stringent_labels=TLMphenotypesdf.dropna()
stringent_labels['label'].value_counts()
stringent_label_of = dict(zip(stringent_labels['id'], stringent_labels['label']))

def apply_loose_label(x):
    if x.startswith('DAmP'):
        return None
    if x.startswith('slightly'):
        return None
    if x == 'ss':
        return 0
    if  x == 'sl':
        return 1
    if x.lower().startswith('l'):
        return 1
    if x == 'VL':
        return 1
    if x.lower().startswith('s'):
        return 0
    if x=='VS':
        return 0
    
TLMphenotypesdf['label'] = TLMphenotypesdf['Telomere Phenotype*'].apply(lambda x: apply_loose_label(x))
loose_labels=TLMphenotypesdf.dropna()
loose_labels['label'].value_counts()
loose_label_of = dict(zip(loose_labels['id'], loose_labels['label']))
#%% make anchor term paths
SIGNAL_net=nx.from_pandas_edgelist(SIGNAL_scores, source='id1',target='id2', edge_attr='SIGNAL')
# type_of={node: label(node) for node in SIGNAL_net.nodes}    
# nx.set_node_attributes(SIGNAL_net,type_of ,name='type')
#check
# nx.get_node_attributes(SIGNAL_net, 'type')[855471]
print(TLMphenotypesdf.shape, len(TLMphenotypesdf['id'].unique())) #todo still dupes
#%% make paths
l_paths = []
s_paths = []

TERMS_in_subset = [x for x in TERMSid if x in loose_label_of.keys()] # todo perche non uso loose_label_of.keys() ? xke non ci sono dentro ANAATnet. xke? non ha senso no? gia.. BUG TODO CONTROOLLLAA
for ANCHOR in ANCHORSid:
    if ANCHOR!= 850696 :#'TEN1': #TEN1, gene cher isultaa non connesso con gli altri nel net, quindi olo leiminiamo
        print('fidind path from:', ANCHOR)
        # Find all paths from ANCHOR to each final node
        
        
        l_paths+=[nx.shortest_path(SIGNAL_net, ANCHOR, node) for node in TERMS_in_subset if loose_label_of[node] == 1] 
        s_paths+=[nx.shortest_path(SIGNAL_net, ANCHOR, node) for node in TERMS_in_subset if loose_label_of[node] == 0]

#%% ADD SIGNAL
SIGNAL_scores.set_index(['id1', 'id2'], inplace=True)
#%%

l_SIGNAL_paths = []
for path in l_paths:
    pathdict={}
    for a, b in itertools.pairwise(path):
        if not (a,b) in list(SIGNAL_scores.index):
            
            pathdict[(a,b)] = SIGNAL_scores.loc[(b,a)]['SIGNAL'] 
        else:
            pathdict[(a,b)] = SIGNAL_scores.loc[(a,b)]['SIGNAL']
    l_SIGNAL_paths.append(pathdict)

print(l_SIGNAL_paths[0])
#%% 
s_SIGNAL_paths = []
for path in s_paths:
    pathdict={}
    for a, b in itertools.pairwise(path):
        if not (a,b) in list(SIGNAL_scores.index):
            
            pathdict[(a,b)] = SIGNAL_scores.loc[(b,a)]['SIGNAL'] 
        else:
            pathdict[(a,b)] = SIGNAL_scores.loc[(a,b)]['SIGNAL']
    s_SIGNAL_paths.append(pathdict)
#%% Obtain a score per pathway
def score_pathway(path, t=0.5):
    '''final sign is given by product of sign of edges
    sign of edges decided based on threshold
    Know that the edges SIGNAL scores represent the probability of the edge to be NEGATIVE'''
    
    return np.prod( np.where(np.array(path)>t, -1,1))

l_predictions = []
for path in l_SIGNAL_paths:
    l_predictions.append(score_pathway(list(path.values())))

s_predictions = []
for path in l_SIGNAL_paths:
    s_predictions.append(score_pathway(list(path.values())))
# problema che qst non mi da scores con 0.00 che posso CVare.. con l'altro 
# funzionava perche avevo molte pathways per un solo KO-T pair..
#%% IMBARCHIAMOCI NELLA SEGUENTE:
# PER OGNI A-T pair:
    #estraiamo tte le SPs dal basenet done
    # reindiamle list of edges DONE
    # calcoliamo lo scores dal Bnet Done
    # Insomma facciamo quello che feci con KO terms..
# for test_p in l(s)_SIGNAL_paths: 
tipo='stringent' #'loose
label_of = stringent_label_of if tipo=='stringent' else loose_label_of


TERMS_in_subset = [x for x in TERMSid if x in label_of.keys()] # todo perche non uso loose_label_of.keys() ? xke non ci sono dentro ANAATnet. xke? non ha senso no? gia.. BUG TODO CONTROOLLLAA


l_lists_of_paths=[]
s_lists_of_paths=[]
for ANCHOR in ANCHORSid:
    if ANCHOR!= 850696 :#'TEN1': #TEN1, gene cher isultaa non connesso con gli altri nel net, quindi olo leiminiamo
        l_lists_of_paths+=[list(nx.all_shortest_paths(basenet, ANCHOR, node)) for node in TERMS_in_subset if label_of[node] == 1] 
        s_lists_of_paths+=[list(nx.all_shortest_paths(basenet, ANCHOR, node)) for node in TERMS_in_subset if label_of[node] == 0]
print(len(l_lists_of_paths))
#%% save them
filename='short_'+tipo+'_lists_of_shortest_paths.txt'
with open(ANAT_VAL_OUT+filename, 'wb') as f:
    pickle.dump(s_lists_of_paths, f)
filename='long_'+tipo+'_lists_of_shortest_paths.txt'
with open(ANAT_VAL_OUT+filename, 'wb') as f:
    pickle.dump(l_lists_of_paths, f)
#%% Generate 
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
l_edges_lists=[[generate_edges_from_path(path) for path in list_of_paths] for list_of_paths in l_lists_of_paths]
s_edges_lists=[[generate_edges_from_path(path) for path in list_of_paths] for list_of_paths in s_lists_of_paths]

testp=l_edges_lists[0][0]
print(testp)
#%%
# generate features:
# LOAD features from NETEDGES. CHEEKY CHEEKY : TODO: devo rifare net edges.
# code gets messy here. just want to get it done...
netind={}
CHUNKNAMES = ['netedgesI','netedgesII','netedgesIII','netedgesIV','netedgesV','netedgesVI']
OLD_INPUT_DIR = HOME_DIR+'features_and_labels'+os.sep+SPECIES+os.sep
for chunk in CHUNKNAMES:
    with open(OLD_INPUT_DIR+chunk,'rb') as f:
        netind[chunk]=pickle.load(f)
#
#%%
def load_features_for(edge): 
    for chunkname in CHUNKNAMES:
        if edge in netind[chunkname]:
            #'opening pickled file and only keep one line'
            with open(OLD_INPUT_DIR+chunkname+'_'+PERT_MAP+'.ft.pkl','rb') as f:
                net_features_table=pickle.load(f)
            return net_features_table.loc[edge]
        
def get_features_for_sp_edges(edges_path):
    valid_features=[]
    for edge in edges_path:
        fts = load_features_for(edge)
        if type(fts)==pd.Series: #some edges may be absent from table
            valid_features.append(fts) #list of series of nfeatures length
    shortest_path_features=pd.DataFrame(valid_features)
    return shortest_path_features
#%% GENERATE FEATURES (loading old ones) and PREDICT SCORE for Long phenotype
#LUNGHISSIMOOOO
l_SIGNAL_lists= []
n=0
for list_of_AT_paths in l_edges_lists:
    n+=1
    if n%10==0:
        print(n, 'LLL', len(l_edges_lists))
    # SIGNAL_lists=[]
    def function(path):        
        
        # get features
        path_features_table = get_features_for_sp_edges(path)
        # predict SIGNAL score
        path_SIGNAL = predictSIGNAL(path_features_table, 'patkar_kegg_kpi_ubinet2_Kemmeren.rf', MOD_DIR, train=False)
        SIGNAL_lists.append(path_SIGNAL)
        return SIGNAL_lists

        
    SIGNAL_lists = Parallel(n_jobs=9)(delayed(function)\
                                (path)\
                                    for path in list_of_AT_paths)
    l_SIGNAL_lists.append(SIGNAL_lists)

# write list of lists of dataframes.
filename='long_'+tipo+'_lists_of_shortest_paths_SIGNAL_scores.pkl'
with open(ANAT_VAL_OUT+filename,'wb') as f:
    pickle.dump(l_SIGNAL_lists, f)
    
#%% GENERATE FEATURES (loading old ones) and PREDICT SCORE for short phenotype
# LUNGHISSIMOOOOOOOOOOOOOOO
s_SIGNAL_lists= []
n=0
for list_of_AT_paths in s_edges_lists:
    n+=1
    if n%10==0:
        print(n, 'SSSS', len(s_edges_lists))
    SIGNAL_lists=[]
    
    for path in list_of_AT_paths:        
        
        # get features
        path_features_table = get_features_for_sp_edges(path)
        # predict SIGNAL score
        path_SIGNAL = predictSIGNAL(path_features_table, 'patkar_kegg_kpi_ubinet2_Kemmeren.rf', MOD_DIR, train=False)
        SIGNAL_lists.append(path_SIGNAL)
    
    

    
    s_SIGNAL_lists.append(SIGNAL_lists)

# write list of lists of dataframes.
filename='short_'+tipo+'_lists_of_shortest_paths_SIGNAL_scores.pkl'
with open(ANAT_VAL_OUT+filename,'wb') as f:
    pickle.dump(s_SIGNAL_lists, f)
    #%%
from sklearn.metrics import roc_curve,  auc

#%%mockdata
# testsignal=pd.read_csv(SIGNAL_DIR+'random9.edges_patkar_kegg_kpi_ubinet2_Kemmeren.sgnl',sep=' ', header=None, index_col=([0,1]))
# l_lists = [[testsignal for x in range(5)] for y in range(100)]
# s_lists = [[testsignal for x in range(5)] for y in range(100)]
#%%
def predict_path_sign(plustsp, minustsp,t=0.5, n_SPs=0, node=False):
    ''' taking into account reversal of signs
    t=threshold for negative edges
    n_SPs=minimum n of shortest paths to use
    '''
   
    def score_negative_paths(paths,t, node):
        '''paths is a list of series where the index 
        is the edge tuple and the value is the SIGNAL score'''
        Sminus=0
        Splus=0
        # print(paths)
        for path in paths:
            path = path.values.flatten()
                        
            #transform into a (0,1) array based on threshold (1= negative edge)
                
            neg_nodes_p= np.where(path>=t, 1, 0)
            # print(neg_nodes_p, len(neg_nodes_p))
            if sum(neg_nodes_p)%2==1:
                Sminus+=1
            else:
                Splus+=1
        
        return round(Sminus/(Sminus+Splus),3)

    # S score for negative paths for positive pathways
    Y_true_plus = []
    Y_pred_plus = []
    for SPS in plustsp:
        if (len(SPS)>n_SPs): #change here which pathways to use
            Y_pred_plus.append(score_negative_paths(SPS,t, node))
            Y_true_plus+=[1] # 0 is + T, 1 is -T
    
    Y_true_minus = []
    Y_pred_minus = []
    for SPS in minustsp:
        if (len(SPS)>n_SPs): #change here which pathways to use
            Y_pred_minus.append(score_negative_paths(SPS,t, node))
            Y_true_minus+=[1] # 0 is + T, 1 is -T


    return Y_true_plus+Y_true_minus, Y_pred_plus+Y_pred_minus

def calcROCAUC(Y_true, Y_pred):
    fprs, tprs, s_thresholds = roc_curve(Y_true, Y_pred)
    aucvalue=auc(fprs, tprs)
    return fprs, tprs, s_thresholds, aucvalue

y_pred, y_true = predict_path_sign(l_SIGNAL_lists, s_SIGNAL_lists)

#%% mock data
#%%
import random
from sklearn.model_selection import train_test_split
tprs_of={}
fprs_of={}
AUC_of={}
best_AUC=0.0
best_params={'SP':0,'t':0}
for SP in (0, 100, 200):
    print('-------------SP:',SP)
    for t in np.arange(0,1,0.1):
        print('\tcalculating all s thresholds for t', t)
        ## TEMP MOCK DATA
        Y_true = [random.randint(0,1) for x in range(1000)] # tempo mock data!
        Y_pred = [random.random() for x in range(1000)]
        ###
        # Y_true, Y_pred = predict_path_sign(PLUSTSP, MINUSTSP, t, SP)
        
        Y_true_train, Y_true_test, Y_pred_train, Y_pred_test = train_test_split(Y_true, Y_pred, test_size=0.5, random_state=42)
        fprs, tprs, s_thresholds, AUC = calcROCAUC(Y_true_train, Y_pred_train)
        tprs_of[t] = tprs
        fprs_of[t] = fprs
        AUC_of[t] = AUC
        fprs, tprs, s_thresholds, testAUC = calcROCAUC(Y_true_test, Y_pred_test)
        print('\t\ttrain AUC:', AUC_of[t],'\tvalid AUC:',testAUC)
        
        #update best model:
        if testAUC>=best_AUC:
            print('DING DING DING! best model found')
            best_AUC=testAUC
            best_params['SP']=SP
            best_params['t']=t
#%%
