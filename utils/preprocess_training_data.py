# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 11:55:44 2023

@author: los4
"""
import os
import pandas  as pd
from datetime import date
from preproc_utils import load_training_data
from train_and_vis3_5 import  load_features
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import argparse
from glob_vars import SPECIES, TRAIN_DATA, PERT_MAP, HOME_DIR, LBL_DIR,\
    EDGES_DIR, FT_DIR, SIGNAL_DIR, MOD_DIR

def get_training_edges(dataname):
    edges_filename=dataname+'.edges'
    signed_datasets_labels, signed_datasets_edge_weights = load_training_data(HOME_DIR+'features_and_labels'+\
                                                                              os.sep+SPECIES+os.sep, [dataname], SPECIES)
    trainedges = list(signed_datasets_labels[dataname].index)
    with open(EDGES_DIR+edges_filename, 'wb') as f:
        pickle.dump(trainedges,f)
    return edges_filename

for data in TRAIN_DATA:
    get_training_edges(data)