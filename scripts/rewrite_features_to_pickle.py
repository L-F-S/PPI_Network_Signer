# -*- coding: utf-8 -*-
"""
@author: L-F-S

"""

import pickle
import collections
import itertools
import lxml.etree as etree
import os
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from random import seed, sample
from datetime import date
from train_and_vis3_5 import k_fold, read_features, log, load_features, AUCs




SPECIES =  "S_cerevisiae" #

HOME_DIR  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# HOME_DIR  =  "G:" +os.sep+"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 

DATA_DIR = HOME_DIR+'Data'+os.sep+SPECIES+os.sep
DIRECTED_DIR = HOME_DIR+"network_signing"+os.sep
INPUT_DIR=DIRECTED_DIR+'features_and_labels'+os.sep+SPECIES+os.sep
OUTDIR=DIRECTED_DIR+'models_and_predictions'+os.sep+SPECIES+os.sep
IMG_DIR=HOME_DIR+"network_signing"+os.sep+'imgs'+os.sep+'v3'+os.sep+SPECIES+os.sep
pert_map = 'Kemmeren'#'ADPBH_1' #'ADPBH_1000'#'reimand' ##  human: 'ADPBH_1 'ADPBH_1000'  'K562gwps' 

#%%
#dictionaries are ordered in pyhton 3.7+, so the order is the same as the features
#(checked with:)
# netfeatures = load_features(INPUTDIR, ['netedgesI_'+pert_map], SPECIES)
# netfeatures=list(netfeatures.values())[0]
# netcols=netfeatures.columns
# netcols==feature_columns
# feature_columns = [str(i)+'+' for i in list(plus_targets_of_deletion.keys())]+[str(i)+'-' for i in list(plus_targets_of_deletion.keys())]
#%% UTIL: load features and rewrite them as pickles
CHUNKNAMES=['netedgesI','netedgesII','netedgesIII','netedgesIV','netedgesV','netedgesVI']

for netname in CHUNKNAMES:
    print( netname)
    netfeatures = load_features(INPUT_DIR, [netname+'_'+pert_map], SPECIES)
    print('writing')
    #rewrite them to pickles
    with open(INPUT_DIR+netname+pert_map+'.ft.pkl','wb') as f:
        pickle.dump(list(netfeatures.values())[0], f)
