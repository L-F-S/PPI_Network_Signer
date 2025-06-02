# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:34:19 2025

@author:L-F-S

merges different signal files into one single signed PPI base network
	input: 
		base net signal scoress)
		training data signal scores

	output: signed_base_net_TRAINDATA_PERT_MAP.sgnl

"""

import os
import pandas  as pd
import numpy as np
from preproc_utils import load_training_data
from train_and_vis import  load_features
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import argparse
from glob_vars import SPECIES, TRAIN_DATA, PERT_MAP, HOME_DIR, LBL_DIR,\
    EDGES_DIR, FT_DIR, SIGNAL_DIR, MOD_DIR
    
print(SPECIES, TRAIN_DATA, PERT_MAP, HOME_DIR)

#%% Utility functions

def merge_yeast_signed_PPI():

    # load all PPI edges except training edges

    list_of_DFs=[]
    for i in range(6):
        filename=SIGNAL_DIR+'net'+str(i+1)+'.edges_'+'_'.join(TRAIN_DATA)+'_'+PERT_MAP+'.sgnl'
        chunk_signed_PPI=pd.read_csv(filename, sep=' ', index_col=[0,1], header=None)
        list_of_DFs.append(chunk_signed_PPI)

    # load all training edges
    train_data='_'.join(TRAIN_DATA)
    train_edges_signed_PPI=pd.read_csv(SIGNAL_DIR+train_data+'_'+train_data+'_'+PERT_MAP+'.sgnl', sep=' ', index_col=[0,1], header=None)
    list_of_DFs.append(train_edges_signed_PPI)

    return pd.concat(list_of_DFs)
    
merged_signed_yeast_PPI=merge_yeast_signed_PPI()
print(merged_signed_yeast_PPI.shape)

#%% check for uniques (there are 8 more than the orignal PPI)
print('original data with dupes:', len(merged_signed_yeast_PPI))
print('np.unique indexes',len(np.unique(merged_signed_yeast_PPI.index))) # it s the original number
    
signed_yeast_PPI=merged_signed_yeast_PPI.reset_index().drop_duplicates(subset=[0,1], keep='last').set_index([0,1])
print(len(signed_yeast_PPI))
#%% write for later use
SIGNALscore_filename='signed_base_net_'+'_'.join(TRAIN_DATA)+'_'+PERT_MAP
signed_yeast_PPI.to_csv(SIGNAL_DIR+os.sep+SIGNALscore_filename+'.sgnl', sep=' ', header=False)
