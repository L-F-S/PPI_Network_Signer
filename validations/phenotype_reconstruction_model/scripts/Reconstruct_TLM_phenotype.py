# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

Reconstruct TLM phenotype 
    To run with the output of generate_SP_scores_TLM_inputs.py
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
        KO_VAL_OUT, IMG_DIR, tau, nSPS, epsilon
from path_sign_reconstruction import predict_path_sign, \
    predict_path_sign_TLM_paths, calc_AUC
 
##############################################################################
#  INPUTS
##############################################################################

# Servier directories:
# HOME_DIR  =  '/home/of/SIGNAL'
# INPUT_DIR=HOME_DIR+"input"+os.sep+SPECIES+os.sep
# OUTDIR=HOME_DIR+'output'+os.sep+SPECIES+os.sep
# FEATURESDIR=OUTDIR
# BASENETDIR=INPUT_DIR

dataset_name = '_'.join(TRAIN_DATA)

print('tau = ', tau)
print('epsilon = ', epsilon)

#%% 
#####################
# TLM DATA:
######################

val_data_name='TLM_all_KOterms' 
sp_name='_SP_SIGNALv2_'#'_all'
# all_long_tags=['VL','L','SL', 'DAmP L']
# all_short_tags=['VS','S','SS', 'DAmP S']
long_tags=['VL', 'DAmP L']
short_tags=['VS', 'DAmP S']
spname_short='_SP_SIGNALv2_'+'_'.join(short_tags)
spname_long='_SP_SIGNALv2_'+'_'.join(long_tags)
with open(KO_VAL_OUT+os.sep+val_data_name+spname_short+dataset_name+'_'+PERT_MAP+'.pkl', 'rb') as f:
    TLMS=pickle.load(f)
with open(KO_VAL_OUT+os.sep+val_data_name+spname_long+dataset_name+'_'+PERT_MAP+'.pkl', 'rb') as f:
    TLML=pickle.load(f)
#%%

Y_true, Y_pred  = predict_path_sign_TLM_paths(TLMS, TLML, tau, n_SPs=nSPS, e=epsilon)
fprs, tprs, s_thresholds, AUC = calc_AUC(Y_true, Y_pred, True)
# fprs, tprs, s_thresholds, datapointplus, datapointminus  = predict_path_sign_TLM_paths(TLMS, TLML,t=tau, n_SPs=nSPS)
current_auc=np.round(auc(fprs, tprs),2)
print(short_tags, long_tags, 'auc', current_auc, 'e',epsilon, 'nSPS',nSPS)

