# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

Predict sign of leftout knockout/target pairs

To run with the output of PARALLELgenerate_SP_scores.py

"""

import os
import sys
HOME_DIR='G:'+os.sep+'Il Mio Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep+'network_signing'+os.sep+'SIGNAL_release'+os.sep
HOME_DIR='G:'+os.sep+'My Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep+'network_signing'+os.sep+'SIGNAL_release'+os.sep
sys.path.append(HOME_DIR)
import pandas  as pd
import numpy  as np
from datetime import date
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from glob_vars import SPECIES, PERT_MAP, TRAIN_DATA, HOME_DIR, LBL_DIR,\
    EDGES_DIR, FT_DIR, SIGNAL_DIR, MOD_DIR, KO_VAL_HOME,\
        KO_VAL_OUT, IMG_DIR, tau, nSPS
    
##############################################################################
#  INPUTS
##############################################################################

# Servier directories:
# HOME_DIR  =  '/home/bnet/lorenzos/signed/signedv3/'
# INPUT_DIR=HOME_DIR+"input"+os.sep+SPECIES+os.sep
# OUTDIR=HOME_DIR+'output'+os.sep+SPECIES+os.sep
# FEATURESDIR=OUTDIR
# BASENETDIR=INPUT_DIR

dataset_name = '_'.join(TRAIN_DATA)

#%% KT DATA:

TSIGN='MINUS'
with open(KO_VAL_OUT+TSIGN+'_SP_edges_per_source_target_pair_'+dataset_name+'_'+PERT_MAP+'.pkl', 'rb') as f:
    MINUSTSP=pickle.load(f)
TSIGN='PLUS'
with open(KO_VAL_OUT+TSIGN+'_SP_edges_per_source_target_pair_'+dataset_name+'_'+PERT_MAP+'.pkl', 'rb') as f:
    PLUSTSP=pickle.load(f)

    #%%

def calcROC(PLUSTSP, MINUSTSP,t=0.5, n_SPs=0):
    ''' taking into account reversal of signs'''
    Y_true = []
    SplusT = defaultdict(dict)
    SminusT = defaultdict(dict)

    def score_negative_paths(paths,t):
        Sminus=0
        Splus=0
        for path in paths:
            if path.size>1:
                #transform into a (0,1) array based on threshold (1= negative edge)
                neg_edges_p= np.where(path[:,1]>=t, 1, 0) 
                if sum(neg_edges_p)%2==1:
                    Sminus+=1
                else:
                    Splus+=1
        return round(Sminus/(Sminus+Splus),3)

    # S score for negative paths for positive KT pairs
    for KO, SPSdict in PLUSTSP.items():
        for T, SPS in SPSdict.items():
            pair=(KO,T)
            if (len(SPS)>n_SPs): #change here which pathways to use
                SplusT[pair] = score_negative_paths(SPS,t)
                Y_true+=[1] # 0 is + T, 1 is -T
    
    # S score for negative paths for positive KT pairs
    for KO, SPSdict in MINUSTSP.items():
        for T, SPS in SPSdict.items():
            pair=(KO,T)
            if len(SPS)>n_SPs: #change here which pathways to use
                SminusT[pair] = score_negative_paths(SPS,t)
                Y_true+=[0] # 0 is + T, 1 is -T
    SplusT=pd.Series(SplusT)
    SminusT=pd.Series(SminusT)
    print('\t\t+',len(SplusT), '-', len(SminusT))
    fprs, tprs, s_thresholds = roc_curve(Y_true, list(pd.concat([SplusT,SminusT])) )
    return fprs, tprs, s_thresholds

#%% calculate signal:
fprs, tprs, s_thresholds = calcROC(PLUSTSP, MINUSTSP, t=tau, n_SPs=nSPS)
print(auc(fprs, tprs))
# print(s_thresholds)
#%% 
#####################
# TLM DATA:
######################

val_data_name='TLM_all_KOterms'  #v1 not needed naymore'TLM_strong_normal'
sp_name='_SP_SIGNALv2_'#'_all'
all_long_tags=['VL','L','SL', 'DAmP L']
all_short_tags=['VS','S','SS', 'DAmP S']

#%%
def calcROC_SIGNAL_only(PLUSTSP, MINUSTSP,t=0.5, n_SPs=0, precrec=False):
    ''' taking into account reversal of signs
    Input is a dictionary of dictionaries of SPs of Signal scores 
    (instead of wd array of (1-SIGNAL, SIGNAL) for KT data )
    WARNING: signs are inverted wrt KO validation,
    because KO actually are supposed to show the inverse of the sign
    (gene upregulated with KO = downregulated in standard condition ), 
    since this function is copied from calcROC,
    I kept the signs unchanged, so you currently need to input the - with the + and vice versa.
    '''
    Y_true = []
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
                Y_true+=[1] # 0 is + T, 1 is -T
    
    # S score for negative paths for positive KT pairs
    for KO, SPSdict in MINUSTSP.items():
        for T, SPS in SPSdict.items():
            pair=(KO,T)
            if len(SPS)>n_SPs:
                SplusT[pair] = score_negative_paths(SPS,t)
                Y_true+=[1] # 0 is + T, 1 is -T
            SminusT[pair] = score_negative_paths(SPS,t)
            Y_true+=[0] # 0 is + T, 1 is -T
    SplusT=pd.Series(SplusT)
    SminusT=pd.Series(SminusT)
    print('\t\t+',len(SplusT), '-', len(SminusT), '% of 1 in Y_true', sum(Y_true)/len(Y_true))
    
    if precrec:
        tprs, fprs, s_thresholds = precision_recall_curve(Y_true, list(pd.concat([SplusT,SminusT]))) 
    else:
        fprs, tprs, s_thresholds = roc_curve(Y_true, list(pd.concat([SplusT,SminusT])) )
    
    return fprs, tprs, s_thresholds, len(SplusT), len(SminusT)

# print(s_thresholds)

#%%
long_tags=['VL', 'DAmP L']
short_tags=['VS', 'DAmP S']
spname_short='_SP_SIGNALv2_'+'_'.join(short_tags)
spname_long='_SP_SIGNALv2_'+'_'.join(long_tags)
with open(KO_VAL_OUT+os.sep+val_data_name+spname_short+dataset_name+'_'+PERT_MAP+'.pkl', 'rb') as f:
    TLMS=pickle.load(f)
with open(KO_VAL_OUT+os.sep+val_data_name+spname_long+dataset_name+'_'+PERT_MAP+'.pkl', 'rb') as f:
    TLML=pickle.load(f)

fprs, tprs, s_thresholds, datapointplus, datapointminus  = calcROC_SIGNAL_only(TLMS, TLML,t=tau, n_SPs=nSPS)
current_auc=np.round(auc(fprs, tprs),2)
print(tau, nSPS)
print(short_tags, long_tags, current_auc, datapointplus, datapointminus)

