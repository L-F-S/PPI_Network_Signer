# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:32:43 2024

@author:L-F-S
"""

# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

Predict output and save TLM pathways genes based on predicted sign

"""

import os
HOME_DIR  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# HOME_DIR  =  "G:" +os.sep+"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
DIRECTED_DIR = HOME_DIR+"network_signing"+os.sep
os.chdir(DIRECTED_DIR)
import pandas  as pd
import numpy  as np
from datetime import date
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import itertools
import networkx as nx
from glob_vars import SPECIES, PERT_MAP, TRAIN_DATA, HOME_DIR, LBL_DIR,\
    EDGES_DIR, FT_DIR, SIGNAL_DIR, MOD_DIR, KO_VAL_HOME, KO_VAL_OUT, IMG_DIR
    
##############################################################################
#  INPUTS
##############################################################################

dataset_name = '_'.join(TRAIN_DATA)

#%% 
#####################
# TLM DATA:
######################

val_data_name='TLM_all_KOterms'  #v1 not needed naymore'TLM_strong_normal'
sp_name='_SP_SIGNALv2_'#'_all'
long_tags=['VL', 'DAmP L']
short_tags=['VS', 'DAmP S']
t=0.5
nSPS=100

#%%
spname_short='_SP_SIGNALv2_'+'_'.join(short_tags)
spname_long='_SP_SIGNALv2_'+'_'.join(long_tags)
with open(KO_VAL_OUT+os.sep+val_data_name+spname_short+dataset_name+'_'+PERT_MAP+'.pkl', 'rb') as f:
    TLMS=pickle.load(f)
with open(KO_VAL_OUT+os.sep+val_data_name+spname_long+dataset_name+'_'+PERT_MAP+'.pkl', 'rb') as f:
    TLML=pickle.load(f)
#%% 

def calcROC_SIGNAL_only(PLUSTSP, MINUSTSP,t=0.5, n_SPs=0, precrec=False):
    ''' taking into account reversal of signs
    Input is a dictionary of dictionaries of SPs of Signal scores (instead of wd array of (1-SIGNAL, SIGNAL) for KT data )
    WARNING: signs are inverted wrt KO validation, because KO actually are supposed to show the inverse of the sign
    (gene upregulated with KO = downregulated in standard condition ), since this function is copied from calcROC,
    I kept the signs unchanged, so you currently need to input the - with the + and vice versa.
    '''
    Y_true = {}
    
    SplusT = defaultdict(dict)
    SminusT = defaultdict(dict)

    def score_negative_paths(paths,t):
        Sminus=0
        Splus=0
        for path in paths:
            # print('p',path)
            #transform into a (0,1) array based on threshold (1= negative edge)
            neg_edges_p= np.where(path>=t, 1, 0) 
            # print('n', neg_edges_p)
            if sum(neg_edges_p)%2==1:
                Sminus+=1
            else:
                Splus+=1
                # print('Sminus',Sminus,'Splus',Splus)
        return round(Sminus/(Sminus+Splus),3)

    # S score for negative paths for positive KT pairs
    for KO, SPSdict in PLUSTSP.items():
        for T, SPS in SPSdict.items():
            pair=(KO,T)
            if len(SPS)>n_SPs:
                SplusT[pair] = score_negative_paths(SPS,t)
                Y_true[pair]=1 # 0 is + T, 1 is -T
    
    # S score for negative paths for positive KT pairs
    for KO, SPSdict in MINUSTSP.items():
        for T, SPS in SPSdict.items():
            pair=(KO,T)
            if len(SPS)>n_SPs:
                SplusT[pair] = score_negative_paths(SPS,t)
                Y_true[pair]=1 # 0 is + T, 1 is -T
            SminusT[pair] = score_negative_paths(SPS,t)
            Y_true[pair]=0 # 0 is + T, 1 is -T
    SplusT=pd.Series(SplusT)
    SminusT=pd.Series(SminusT)
    
    # if precrec:
    #     tprs, fprs, s_thresholds = precision_recall_curve(Y_true, list(pd.concat([SplusT,SminusT]))) 
    # else:
    #     fprs, tprs, s_thresholds = roc_curve(Y_true, list(pd.concat([SplusT,SminusT])) )
    
    Y_true=pd.Series(Y_true)
    return  SplusT, SminusT, Y_true

#%%

SplusT, SminusT, Y_true =  calcROC_SIGNAL_only(TLMS, TLML,t=t, n_SPs=nSPS)
# auc=np.round(auc(fprs, tprs),2)

#
