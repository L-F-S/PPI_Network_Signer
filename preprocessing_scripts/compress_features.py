# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:59:50 2024

@author:L-F-S
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

print(TRAIN_DATA, SPECIES)
for train in TRAIN_DATA:
    df=pd.read_csv(FT_DIR+train+'_'+PERT_MAP+'.ft.csv', index_col=[0,1], header=0)
    print(df.head(2))
    with open(FT_DIR+train+'_'+PERT_MAP+'.ft','wb') as f:
        pickle.dump(df, f)