# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:32:32 2021

@author: L-F-S
3.5 removed p_complex validations
"""

import os
import sys
import pandas  as pd
import numpy  as np
from datetime import date
import matplotlib.pyplot as plt
from preproc_utils import load_training_data
from train_and_vis import k_fold, log, load_features
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import pickle
from glob_vars import SPECIES, TRAIN_DATA, PERT_MAP, LBL_DIR,\
    FT_DIR, MOD_DIR, IMG_DIR, PRT_DIR, CV_OUT, CV_LOG

##############################################################################
#  INPUTS
##############################################################################
print('Current inputs for crossvalidation:')
print(SPECIES)
print(TRAIN_DATA)
print(PERT_MAP)


logfile=CV_OUT+CV_LOG
DATE=str(date.today())

mean_AUROC_of = {}
mean_AUPR_of = {}
std_AUROC_of = {}
std_AUPR_of = {}


dataset_name = '_'.join(TRAIN_DATA)
sampling = ''#'_DS'
std=True #!! attenzione! NON PROVATECI A CASA
if not std:
    sampling = 'UNSTANDARDIZED'
finaldataname=dataset_name+'_'+PERT_MAP+sampling
comments=sampling
#  merge selected TRAIN_DATA and perform crossvalidation
signed_TRAIN_DATA_labels, _ = load_training_data(LBL_DIR, TRAIN_DATA, SPECIES)
signed_TRAIN_DATA_features = load_features(FT_DIR, [i+'_'+PERT_MAP for i in TRAIN_DATA], SPECIES) #attenzione, aggiungi _PERT_MAP se non e' hosltege


training_labels=pd.concat(list(signed_TRAIN_DATA_labels.values())) # works because python dictionaries are ordered in python3
features_table= pd.concat(list(signed_TRAIN_DATA_features.values()))

# # load perturbation signatures
# with open(PRT_DIR+'plus_targets'+'_'+PERT_MAP+'.pkl', 'rb') as f:
#     plus_targets_of_deletion=pickle.load( f)
# with open(PRT_DIR+'minus_targets'+'_'+PERT_MAP+'.pkl','rb') as f:
#     minus_targets_of_deletion=pickle.load(f)

#%% merged TRAIN_DATA CV:
classifier=RandomForestClassifier()#LogisticRegression(max_iter=1000)#
classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
    mean_AUROC_of[finaldataname], std_AUROC_of[finaldataname] =k_fold(classifier,5,\
                            features_table,training_labels, finaldataname, \
                                IMG_DIR, metric="ROC", downsampling=sampling,\
                                    plot=False, std=std) 

# classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
#     mean_AUPR_of[finaldataname], std_AUPR_of[finaldataname]=k_fold(classifier,5,\
#                             features_table,training_labels, finaldataname,\
#                                 IMG_DIR, metric="PrecRec", downsampling=sampling,
#                                 plot=True)

print("ROC AUC:", finaldataname, np.round(mean_AUROC_of[finaldataname],2))
# print("PrecRec AUC:", finaldataname, np.round(mean_AUPR_of[finaldataname],2))
#%%
log(dataset_name,np.round(mean_AUROC_of[finaldataname],2),\
    '-',logfile, DATE, SPECIES,\
            pert_map=PERT_MAP, comments=comments)
print('saving model ', finaldataname)
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
    print("ROC AUC:", finaldataname, np.round(mean_AUROC_of[finaldataname],2))
    # print("PrecRec AUC:", finaldataname, np.round(mean_AUPR_of[finaldataname],2))
    
    log(dataset_name,np.round(mean_AUROC_of[finaldataname],2),\
        '-',logfile, DATE, SPECIES,\
            PERT_MAP=PERT_MAP, comments=comments)
    
    with open(MOD_DIR+finaldataname+'.rf', 'wb') as f:
        pickle.dump(classifier, f)
#%% 
with open(MOD_DIR+finaldataname+'.rf','rb') as f:
    classifier=pickle.load(f)
#%%% 10/03 different thing: train on smthj, test on 
# -kegg: non funge, proviamo il prossimo
# kpi 20% out
kpi_features=load_features(FT_DIR, ['kpi_'+PERT_MAP], SPECIES)
kpi_features=kpi_features['kpi_'+PERT_MAP]
kpi_labels, _ =load_training_data(LBL_DIR, ['kpi'], SPECIES)
kpi_labels=kpi_labels['kpi']
# take 1/3 kpi off as test
test_features_table=kpi_features.iloc[0:int(kpi_features.shape[0]/3)]
rest_of_kpi_features=kpi_features.iloc[int(kpi_features.shape[0]/3):]
test_labels = kpi_labels[0:int(kpi_features.shape[0]/3)]
rest_of_kpi_labels = kpi_labels[int(kpi_features.shape[0]/3):]


other_datasets= ['patkar_kegg']#, 'ubinet2']
signed_TRAIN_DATA_labels, _ = load_training_data(LBL_DIR, other_datasets, SPECIES)
# signed_TEST_DATA_labels, _ = load_training_data(LBL_DIR, test_data, SPECIES)
signed_TRAIN_DATA_features = load_features(FT_DIR, [i+'_'+PERT_MAP for i in other_datasets], SPECIES) 
# signed_TEST_DATA_features = load_features(FT_DIR, [i+'_'+PERT_MAP for i in test_data], SPECIES) 


#%%
l_lbl=list(signed_TRAIN_DATA_labels.values())
# l_lbl.append(rest_of_kpi_labels)
training_labels=pd.concat(l_lbl) # works because python dictionaries are ordered in python3
l_ftr=list(signed_TRAIN_DATA_features.values())
# l_ftr.append(rest_of_kpi_features)
features_table= pd.concat(l_ftr)
# test_labels=pd.concat(list(signed_TEST_DATA_labels.values())) # works because python dictionaries are ordered in python3
# test_features_table= pd.concat(list(signed_TEST_DATA_features.values()))
#%%
x_train = pd.DataFrame(StandardScaler().fit_transform(features_table), columns=features_table.columns)
x_train=np.array(x_train)
x_test = pd.DataFrame(StandardScaler().fit_transform(test_features_table), columns=test_features_table.columns)
x_test=np.array(x_test)

y_train=np.array(training_labels)
y_test=np.array(test_labels)
#
classifier=RandomForestClassifier()#LogisticRegression(max_iter=1000)#
classifier.fit(x_train, y_train)       
y_predicted=classifier.predict_proba(x_test)[:,1:]
# plot auc
fpr, tpr, thresholds = roc_curve(y_test, y_predicted)
roc_auc= auc(fpr, tpr)
# fig, ax =plt.subplots()
# ax.plot(fpr, tpr, lw=1)
print(roc_auc)
# ax.set_xlabel(XLABEL_OF[metric])
# ax.set_ylabel(YLABEL_OF[metric])
#%%
comment='1/3kpi test,ALL REST TRAIN'
dataset_name = 'add manually'
log(dataset_name,np.round(roc_auc,2),\
    '-',logfile, DATE, SPECIES,\
            pert_map=PERT_MAP, comments=comment)
