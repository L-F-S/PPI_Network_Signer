# -*- coding: utf-8 -*-
"""
@author: L-F-S

Generate KOless features before running this script

utility script for KT pairs reconstuction validation 

Predicts SIGNER scores for all base network, leaving one KO out every time.
output: KOless predictions for every KOgene, for all networks (divided into chunks)

"""


import os
from glob_vars import PERT_MAP, TRAIN_DATA, FT_DIR, NET_DIR, SIGNAL_DIR,\
    DICT_DIR, LBL_DIR, EDGES_DIR, SPECIES, PRT_DIR
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
print(SPECIES)
DATE=str(date.today())

if SPECIES == 'S_cerevisiae':
    TRAIN_DATA = ['kegg','kpi','ubinet2']
else:
    TRAIN_DATA = ['kegg','PSP','depod','ubinet2']
dataset_name = '_'.join(TRAIN_DATA)


# load knockdowns and targets
TSIGN = 'MINUS'
if TSIGN=='PLUS':
    with open(PRT_DIR+'plus_targets'+'_'+PERT_MAP+'.pkl', 'rb') as f:
        targets_of_deletion=pickle.load( f)
if TSIGN =='MINUS':
    with open(PRT_DIR+'minus_targets'+'_'+PERT_MAP+'.pkl','rb') as f:
        targets_of_deletion=pickle.load(f)
print('sources:', len(list(targets_of_deletion.keys())),'targets: ',TSIGN)
# print('train classifier' )

training_labels, training_labels_weights = load_training_data(LBL_DIR, TRAIN_DATA, SPECIES)
training_features_table = load_features(FT_DIR, [i+'_'+PERT_MAP for i in TRAIN_DATA], SPECIES)
training_labels=pd.concat(list(training_labels.values())) # works because python dictionaries are ordered in python3
training_features_table= pd.concat(list(training_features_table.values()))


BASE_GRAPH=graph_from_dataframe(NET_DIR, SPECIES)
for i, data in training_labels_weights.items():
    BASE_GRAPH = add_edges_from_labels(BASE_GRAPH,data)
    
CHUNKNAMES=['netedgesI','netedgesII','netedgesIII','netedgesIV','netedgesV','netedgesVI']
#print('Load edges (indexes) and features (columns) for base net')
netind={}
for chunk in CHUNKNAMES:
    with open(EDGES_DIR+chunk,'rb') as f:
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
        with open(FT_DIR+chunkname+'_'+PERT_MAP+'.ft.pkl','rb') as f:
            chunk_net_features_table=pickle.load(f)
        
        predictions=predict_edges(source, training_features_table, chunk_net_features_table)
        
        with open(SIGNAL_DIR+os.sep+'KO-'+str(source)+'-'+chunkname+'_'+dataset_name+'_'+PERT_MAP+'.sgnl', 'wb') as f:
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
