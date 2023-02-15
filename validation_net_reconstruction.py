# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
Predict sign of leftout knockout/target pairs
"""


import os
import pandas  as pd
import numpy  as np
from datetime import date
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
from preproc_utils import load_training_data ,graph_from_dataframe, add_edges_from_labels
from train_and_vis3_5 import k_fold, read_features, log, load_features, AUCs
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import networkx as nx

##############################################################################
#  INPUTS
##############################################################################
# SPECIES = "H_sapiens"#
SPECIES =  "S_cerevisiae" #

HOME_DIR  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
HOME_DIR  =  "G:" +os.sep+"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 

DATA_DIR = HOME_DIR+'Data'+os.sep+SPECIES+os.sep
DIRECTED_DIR = HOME_DIR+"network_signing"+os.sep
INPUTDIR=DIRECTED_DIR+'features_and_labels'+os.sep+SPECIES+os.sep
OUTDIR=DIRECTED_DIR+'models_and_predictions'+os.sep+SPECIES+os.sep
IMG_DIR=HOME_DIR+"network_signing"+os.sep+'imgs'+os.sep+'v3'+os.sep+SPECIES+os.sep
logfile=DIRECTED_DIR+os.sep+'validation_out'+os.sep+'crossvalidationslog.txt'
DATE=str(date.today())

if SPECIES == 'S_cerevisiae':
    datasets = ['kegg','kpi','ubiq']
else:
    datasets = ['kegg','PSP','depod','ubinet2']
dataset_name = '_'.join(datasets)
pert_map = 'Holstege'#'ADPBH_1' #'ADPBH_1000'#'reimand' ##  human: 'ADPBH_1 'ADPBH_1000'  'K562gwps' 
sampling = ''#'_DS'
finaldataname=dataset_name+'_'+pert_map+sampling
comments=sampling

# load knockdowns and targets
with open(INPUTDIR+'plus_targets'+'_'+pert_map+'.pkl', 'rb') as f:
    plus_targets_of_deletion=pickle.load( f)
with open(INPUTDIR+'minus_targets'+'_'+pert_map+'.pkl','rb') as f:
    minus_targets_of_deletion=pickle.load(f)


#%% train classifier 

training_labels, training_labels_weights = load_training_data(INPUTDIR, datasets, SPECIES)
training_features_table = load_features(INPUTDIR, [i+'_'+pert_map for i in datasets], SPECIES)
training_labels=pd.concat(list(training_labels.values())) # works because python dictionaries are ordered in python3
# training_labels=training_labels.iloc[1:] #temp solo x ADPBH_1 e 1000 (vedi log 30/01/2023)
training_features_table= pd.concat(list(training_features_table.values()))

#%% ALTRA COSA: MAGARI PREDICT KNOCKDOWN/TARGET FROM original data that 
# didnt make it into the targets dictionary?
base_graph=graph_from_dataframe(DATA_DIR, SPECIES)
for i, data in training_labels_weights.items():
    base_graph = add_edges_from_labels(base_graph,data)
    #%%
CHUNKNAMES=['netedgesI','netedgesII','netedgesIII','netedgesIV','netedgesV','netedgesVI']
#%% Load edges (indexes) and features (columns) for base net
netind={}
for chunk in CHUNKNAMES:
    with open(INPUTDIR+chunk,'rb') as f:
        netind[chunk]=pickle.load(f)


#%%
#dictionaries are ordered in pyhton 3.7+, so the order is the same as the features
#(checked with:)
# netfeatures = load_features(INPUTDIR, ['netedgesI_'+pert_map], SPECIES)
# netfeatures=list(netfeatures.values())[0]
# netcols=netfeatures.columns
# netcols==feature_columns
feature_columns = [str(i)+'+' for i in list(plus_targets_of_deletion.keys())]+[str(i)+'-' for i in list(plus_targets_of_deletion.keys())]
#%% UTIL: load features and rewrite them as pickles
for netname in CHUNKNAMES:
    print( netname)
    netfeatures = load_features(INPUTDIR, [netname+'_'+pert_map], SPECIES)
    print('writing')
    #rewrite them to pickles
    with open(INPUTDIR+netname+pert_map+'.ft.pkl','wb') as f:
        pickle.dump(list(netfeatures.values())[0], f)

#%% find shortest path from source to term, then predict proba for edges in that 
# path


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

def load_features_for(edge): #comunque buggato xke ritorna solo la prima edge lolma intanto TODOOO
    for chunkname in CHUNKNAMES:
        if edge in netind[chunkname]:
            #'opening pickled file and only keep one line'
            with open(INPUTDIR+chunkname+'_'+pert_map+'.ft.pkl','rb') as f:
                net_features_table=pickle.load(f)
            return net_features_table.loc[edge]
   
n=0            
for source, targets in minus_targets_of_deletion.items():
    if len(targets)>1:
        print('--- training classifiermwithout KO:',source,'and with targets:', targets)
        n+=1
        # 1. train classifier with all features minus current KO:
        temp_training_features_table = training_features_table.drop([str(source)+i for i in ['+','-']], axis=1)
        classifier=RandomForestClassifier()
        classifier.fit(StandardScaler().fit_transform(temp_training_features_table), training_labels)
    
    #2. find shortest path between KO and its targets
    for target in targets[:1]:
        if target!= source: 
            print('>t', target)
            path = nx.shortest_path(base_graph, source, target)
            edges=generate_edges_from_path(path)
            print('shortest path ewdges:',edges)
            
            # 3. extract its features from the base net feature table (BIG FILE! divided into 6 chunks)
            # and apply to those
            valid_features=[]
            for edge in edges:
                valid_features.append(load_features_for(edge))
            shortest_path_features=pd.DataFrame(valid_features)
            shortest_path_features.drop([str(source)+i for i in ['+','-']], axis=1, inplace=True)
            print(shortest_path_features)
            print('predicting on edges')
            predictions=classifier.predict_proba(StandardScaler().fit_transform(shortest_path_features))
            print('now do something with this boy:\n', predictions)
            # predictions sono le predictions di ogni edge del mio shortest path
    if n==2:
        break
