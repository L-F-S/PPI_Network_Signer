# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:32:32 2021

@author: los4
3.5 removed p_complex validations
"""

import os
import pandas  as pd
import numpy  as np
from datetime import date
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
from preproc_utils import load_training_data
from train_and_vis import k_fold, read_features, log, load_features, AUCs
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
from glob_vars import SPECIES, TRAIN_DATA, PERT_MAP, HOME_DIR, LBL_DIR,\
    FT_DIR, MOD_DIR, IMG_DIR, PRT_DIR

##############################################################################
#  INPUTS
##############################################################################


os.chdir(HOME_DIR)

logfile=HOME_DIR+os.sep+'validation_out'+os.sep+'crossvalidationslog.txt'
DATE=str(date.today())

mean_AUROC_of = {}
mean_AUPR_of = {}
std_AUROC_of = {}
std_AUPR_of = {}

PERT_MAP +='parallelrefactored'
print(PERT_MAP)

dataset_name = '_'.join(TRAIN_DATA)
sampling = ''#'_DS'
finaldataname=dataset_name+'_'+PERT_MAP+sampling
comments=sampling
#%%  merge selected TRAIN_DATA and perform crossvalidation
signed_TRAIN_DATA_labels, _ = load_training_data(LBL_DIR, TRAIN_DATA, SPECIES)
# signed_TRAIN_DATA_labels['kegg'] = signed_TRAIN_DATA_labels['kegg'].iloc[1:]#temp todo: only for kegg CMGE che x qlc motivo la features table manca el primo coos
signed_TRAIN_DATA_features = load_features(FT_DIR, [i+'_'+PERT_MAP for i in TRAIN_DATA], SPECIES) #attenzione, aggiungi _PERT_MAP se non e' hosltege


training_labels=pd.concat(list(signed_TRAIN_DATA_labels.values())) # works because python dictionaries are ordered in python3
features_table= pd.concat(list(signed_TRAIN_DATA_features.values()))

# load perturbation signatures
with open(PRT_DIR+'plus_targets'+'_'+PERT_MAP+'.pkl', 'rb') as f:
    plus_targets_of_deletion=pickle.load( f)
with open(PRT_DIR+'minus_targets'+'_'+PERT_MAP+'.pkl','rb') as f:
    minus_targets_of_deletion=pickle.load(f)

#%% merged TRAIN_DATA CV:
classifier=RandomForestClassifier()#LogisticRegression(max_iter=1000)#
classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
    mean_AUROC_of[finaldataname], std_AUROC_of[finaldataname] =k_fold(classifier,5,\
                            features_table,training_labels, finaldataname, \
                                IMG_DIR, metric="ROC", downsampling=sampling,\
                                    plot=True) 

# classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
#     mean_AUPR_of[finaldataname], std_AUPR_of[finaldataname]=k_fold(classifier,5,\
#                             features_table,training_labels, finaldataname,\
#                                 IMG_DIR, metric="PrecRec", downsampling=sampling,
#                                 plot=True)
mean_AUPR_of[finaldataname]=0
std_AUPR_of[finaldataname]=0
print("ROC AUC:", finaldataname, np.round(mean_AUROC_of[finaldataname],2))
print("PrecRec AUC:", finaldataname, np.round(mean_AUPR_of[finaldataname],2))
#%%
log(dataset_name,np.round(mean_AUROC_of[finaldataname],2),\
    np.round(mean_AUPR_of[finaldataname],2),logfile, DATE, SPECIES,\
            pert_map=PERT_MAP, comments=comments)

with open(MOD_DIR+finaldataname+'.rf', 'wb') as f:
    pickle.dump(classifier, f)
#%% dataset_wise CV:
### 
for dataset_name, training_labels in signed_TRAIN_DATA_labels.items():
    print(dataset_name)
    finaldataname=dataset_name+'_'+PERT_MAP+sampling
    current_features_table = signed_TRAIN_DATA_features[dataset_name+"_"+PERT_MAP]
    classifier=RandomForestClassifier()#LogisticRegression(max_iter=1000)#
    classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
        mean_AUROC_of[finaldataname], std_AUROC_of[finaldataname] =k_fold(classifier,5,\
                                current_features_table,training_labels, finaldataname, \
                                    IMG_DIR, metric="ROC", downsampling=sampling,\
                                        plot=False) 
    
    # classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
    #     mean_AUPR_of[finaldataname], std_AUPR_of[finaldataname]=k_fold(classifier,5,\
    #                             current_features_table,training_labels, finaldataname,\
    #                                 IMG_DIR, metric="PrecRec", downsampling=sampling,
    #                                 plot=True)
    mean_AUPR_of[finaldataname]=0
    std_AUPR_of[finaldataname]=0
    print("ROC AUC:", finaldataname, np.round(mean_AUROC_of[finaldataname],2))
    print("PrecRec AUC:", finaldataname, np.round(mean_AUPR_of[finaldataname],2))
    
    log(dataset_name,np.round(mean_AUROC_of[finaldataname],2),\
        np.round(mean_AUPR_of[finaldataname],2),logfile, DATE, SPECIES,\
            PERT_MAP=PERT_MAP, comments=comments)
    
    with open(MOD_DIR+finaldataname+'.rf', 'wb') as f:
        pickle.dump(classifier, f)
#%% 
with open(MOD_DIR+finaldataname+'.rf','rb') as f:
    classifier=pickle.load(f)
