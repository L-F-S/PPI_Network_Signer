# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 11:55:44 2023

@author: L-F-S


"""
import os
import pandas  as pd
from datetime import date
from preproc_utils import load_training_data
from train_and_vis import  load_features
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import argparse
from glob_vars import SPECIES, TRAIN_DATA, PERT_MAP, HOME_DIR, LBL_DIR,\
    EDGES_DIR, FT_DIR, SIGNAL_DIR, MOD_DIR

#%% create separate files:
    
def get_training_edges(dataname):
    edges_filename=dataname+'.edges'
    signed_datasets_labels, signed_datasets_edge_weights = load_training_data(LBL_DIR, [dataname], SPECIES)
    trainedges = list(signed_datasets_labels[dataname].index)
    with open(EDGES_DIR+edges_filename, 'wb') as f:
        pickle.dump(trainedges,f)
    return edges_filename

for data in TRAIN_DATA:
    get_training_edges(data)
#%% create one single concatenated file:
signed_datasets_labels, signed_datasets_edge_weights = load_training_data(LBL_DIR, TRAIN_DATA, SPECIES)
training_labels=pd.concat(list(signed_datasets_labels.values()))
trainedges = list(training_labels.index)
with open(EDGES_DIR+'alltrain.edges', 'wb') as f:
    pickle.dump(trainedges,f)

#%%

# Concat all training features (created after running defective propagations)
signed_TRAIN_DATA_features = load_features(FT_DIR, [i+'_'+PERT_MAP for i in TRAIN_DATA], SPECIES) #attenzione, aggiungi _PERT_MAP se non e' hosltege
features_table= pd.concat(list(signed_TRAIN_DATA_features.values()))
with open(FT_DIR+'alltrain_'+PERT_MAP+'.ft', 'wb') as f:
    pickle.dump(features_table,f)