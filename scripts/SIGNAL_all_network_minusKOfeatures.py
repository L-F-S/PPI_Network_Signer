# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

utility script for KT pairs reconstuction validation 

Predicts SIGNER scores for all base network, leaving one KO out every time.
output: KOless predictions for every KOgene, for all networks (divided into chunks)
"""


import os
# SPECIES = "H_sapiens"#
SPECIES =  "S_cerevisiae" #


# Servier directories:
HOME_DIR  =  '/home/bnet/lorenzos/signed/signedv3/'
INPUT_DIR=HOME_DIR+"input"+os.sep+SPECIES+os.sep
OUTDIR=HOME_DIR+'output'+os.sep+SPECIES+os.sep+'validation_out'
FEATURESDIR=OUTDIR
BASENETDIR=INPUT_DIR

# # Local directories
# HOME_DIR  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# # HOME_DIR  =  "G:" +os.sep+"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# DIRECTED_DIR = HOME_DIR+"network_signing"+os.sep
# os.chdir(DIRECTED_DIR)
# # HOME_DIR  =  "G:" +os.sep+"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# DATA_DIR = HOME_DIR+'Data'+os.sep+SPECIES+os.sep
# INPUT_DIR=DIRECTED_DIR+'features_and_labels'+os.sep+SPECIES+os.sep
# OUTDIR=DIRECTED_DIR+'validation_out'+os.sep+SPECIES+os.sep
# FEATURESDIR=INPUT_DIR
# IMG_DIR=HOME_DIR+"network_signing"+os.sep+'imgs'+os.sep+'v3'+os.sep+SPECIES+os.sep
# BASENETDIR=DATA_DIR

import pandas  as pd
from datetime import date
from preproc_utils import load_training_data ,graph_from_dataframe, add_edges_from_labels
from train_and_vis3_5 import  load_features
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

##############################################################################
#  INPUTS
##############################################################################



DATE=str(date.today())

if SPECIES == 'S_cerevisiae':
    datasets = ['kegg','kpi','ubiq']
else:
    datasets = ['kegg','PSP','depod','ubinet2']
dataset_name = '_'.join(datasets)
pert_map = 'Kemmeren'#'ADPBH_1' #'ADPBH_1000'#'reimand' ##  human: 'ADPBH_1 'ADPBH_1000'  'K562gwps' 


# load knockdowns and targets
TSIGN = 'MINUS'
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


BASE_GRAPH=graph_from_dataframe(BASENETDIR, SPECIES)
for i, data in training_labels_weights.items():
    BASE_GRAPH = add_edges_from_labels(BASE_GRAPH,data)
    
CHUNKNAMES=['netedgesI','netedgesII','netedgesIII','netedgesIV','netedgesV','netedgesVI']
#print('Load edges (indexes) and features (columns) for base net')
netind={}
for chunk in CHUNKNAMES:
    with open(INPUT_DIR+chunk,'rb') as f:
        netind[chunk]=pickle.load(f)
        #%%

def predict_edges(source, training_features_table, test_features_table):
    temp_training_features_table = training_features_table.drop([str(source)+i for i in ['+','-']], axis=1) 
    classifier=RandomForestClassifier() 
    classifier.fit(StandardScaler().fit_transform(temp_training_features_table), training_labels) 
    
    predictions = classifier.predict_proba(StandardScaler().fit_transform(test_features_table.drop([str(source)+i for i in ['+','-']], axis=1)))
    return pd.DataFrame(data=predictions, index=test_features_table.index, columns=['+','-'])

def function(source):
    for chunkname in CHUNKNAMES:
        print(chunkname)
        with open(INPUT_DIR+chunkname+'_'+pert_map+'.ft.pkl','rb') as f:
            chunk_net_features_table=pickle.load(f)
        
        predictions=predict_edges(source, training_features_table, chunk_net_features_table)
        
        with open(OUTDIR+os.sep+'KO-'+str(source)+'-'+chunkname+'_'+dataset_name+'_'+pert_map+'.sgnr', 'wb') as f:
            pickle.dump(predictions,f)


#%% PREDICT ALL EDGES for different KT pairs
# for source in list(targets_of_deletion.keys())[:1]:
#     print(source)
#     function(source)
#%% PREDICT ALL EDGES for different KT pairs
N_JOBS=64
from joblib import Parallel, delayed

packed_results = Parallel(n_jobs=N_JOBS)(delayed(function)\
                            (source,)\
                                for source in list(targets_of_deletion.keys()))