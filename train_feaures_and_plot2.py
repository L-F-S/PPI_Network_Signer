# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:32:32 2021

@author: los4
"""

import os
import pandas  as pd
import numpy  as np
from datetime import date

import matplotlib.pyplot as plt
from train_and_vis import custom_k_fold, read_features, log
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


##############################################################################
#  INPUTS
##############################################################################
SPECIES = "S_cerevisiae"
DATE=str(date.today())
HOME_DIR  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# HOME_DIR  =  "G:" +os.sep+"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
#HOME_DIR  =  "/content/drive" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 

MAIN_DATA_DIR = HOME_DIR+'Data'+os.sep
DATA_DIR=HOME_DIR+"network_signing"+os.sep+'data'+os.sep
DIRECTED_DIR = HOME_DIR+"network_signing"+os.sep
OUTDIR=DIRECTED_DIR+'features_labels'+os.sep
OUTDIR=DIRECTED_DIR+'features_labels'+os.sep+'17_08 labels with new kegg from anat kegg'+os.sep
#FEATURES_DIR = HOME_DIR+"network_signing"+os.sep+"features_creation_data"+os.sep
HOLSTEGE_DIR= MAIN_DATA_DIR+os.sep+'Holstege_S1Data'+os.sep
IMG_DIR=HOME_DIR+"network_signing"+os.sep+'imgs'+os.sep+'v2'+os.sep
logfile=DIRECTED_DIR+os.sep+'validation_out'+os.sep+'crossvalidationslog.txt'

DATANAMES= ['kegg_kpi']#,'p_complex', 'ubiq']

# Initioalize dictionary of aliases
DOWNSAMPLING = False 
dwsstring=''
if DOWNSAMPLING:
    dwsstring = '.DS'


# store values for different datasets
# training_labels_of ={}
# features_table_of = {}

mean_AUROC_of = {}
mean_AUPR_of = {}
std_AUROC_of = {}
std_AUPR_of = {}
#%%
#
# choose only one  of the following 3 cells, based on what CV you wish to perform;
#
#%%  merge selected datasets and perform crossvalidation
# 03/08/2022: ubiq+kegg+kpi con basenet SENZA PCOMP:
training_labels, features_table, finaldataname, LOGedgedataname,\
    LOGedgeinbasenetname = read_features(OUTDIR, DATE, which=['k','u'], whichinbase=['k','u', 'p'])

# finaldataname = 'all_Network_all'+DATE
classifier=RandomForestClassifier()#LogisticRegression(max_iter=1000)#
classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
    mean_AUROC_of[finaldataname], std_AUROC_of[finaldataname] =custom_k_fold(classifier,5,\
                            pd.DataFrame(StandardScaler().fit_transform(features_table), columns=features_table.columns),\
                                training_labels, finaldataname, IMG_DIR, metric="ROC", plot=False) 

classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
    mean_AUPR_of[finaldataname], std_AUPR_of[finaldataname]=custom_k_fold(classifier,5,\
                            pd.DataFrame(StandardScaler().fit_transform(features_table), columns=features_table.columns),\
                                training_labels, finaldataname, IMG_DIR, metric="PrecRec",plot=False)
print("ROC AUC:", finaldataname, np.round(mean_AUROC_of[finaldataname],2))
print("PrecRec AUC:", finaldataname, np.round(mean_AUPR_of[finaldataname],2))
log(LOGedgedataname,np.round(mean_AUROC_of[finaldataname],2),np.round(mean_AUPR_of[finaldataname],2),logfile, DATE, SPECIES, edgedatainbasenet=LOGedgeinbasenetname)
#%% v1 data
from old_version import data_preproc
alias_2geneid=data_preproc.read_alias2geneid(MAIN_DATA_DIR, SPECIES)

training_labels, _ = data_preproc.get_training_data(DATA_DIR,True,alias_2geneid,'Undirected_training_features')
features_table=pd.read_csv(OUTDIR+'Undirected_training_features.csv', sep=',', header=0)
finaldataname = 'kegg_kpi_Network_kegg_kpi_(V1)'+DATE
l=[(line[0], line[1]) for i, line in features_table[['0','1']].iterrows()]
for index in training_labels.index:
    if not index in l:
        training_labels.drop(index, inplace=True)
for n, dataindex in enumerate(l):
    if not dataindex in training_labels.index:
        features_table.drop(n, inplace = True)
# remove edge names and keep only indexes for classifications 
features_table.drop(columns=['0','1'], inplace=True)
training_labels.index=(range(len(training_labels))) 
features_table.rename(columns={str(i):i-1 for i in range(2,474)}, inplace=True)
us=''
classifier=RandomForestClassifier()#LogisticRegression(max_iter=1000)#
classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
    mean_AUROC_of[finaldataname], std_AUROC_of[finaldataname] =custom_k_fold(classifier,5,\
                            pd.DataFrame(StandardScaler().fit_transform(features_table), columns=features_table.columns),\
                                training_labels, finaldataname, IMG_DIR, metric="ROC", downsampling=DOWNSAMPLING, upsampling=us, plot=True) 

classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
    mean_AUPR_of[finaldataname], std_AUPR_of[finaldataname]=custom_k_fold(classifier,5,\
                            pd.DataFrame(StandardScaler().fit_transform(features_table), columns=features_table.columns),\
                                training_labels, finaldataname, IMG_DIR, metric="PrecRec", downsampling=DOWNSAMPLING,upsampling=us, plot=True)
print("ROC AUC:", finaldataname, np.round(mean_AUROC_of[finaldataname],2))
print("PrecRec AUC:", finaldataname, np.round(mean_AUPR_of[finaldataname],2))



#%% PLOT all aucs together    
# convert dictionaries to series for plotting
mean_AUROC_of = pd.Series(mean_AUROC_of)
mean_AUPR_of = pd.Series(mean_AUPR_of)
std_AUROC_of = pd.Series(std_AUROC_of)
std_AUPR_of  = pd.Series(std_AUPR_of)

for DATANAME in DATANAMES:
    print(DATANAME,  round(mean_AUROC_of.loc[DATANAME+'+'],2), round(mean_AUPR_of.loc[DATANAME+'+'],2)
          , ' \ ', round(mean_AUROC_of.loc[DATANAME+'SMOTE+'],2), round(mean_AUPR_of.loc[DATANAME+'SMOTE+'],2))





#%%
merged_mean = {}
merged_std = {}
for n, i in enumerate(mean_AUROC_of.index):
    merged_mean[i+'ROC'] = mean_AUROC_of.loc[i]
    merged_mean[i+'PR'] = mean_AUPR_of.loc[i]
    merged_std[i+'ROC'] = std_AUROC_of.loc[i]
    merged_std[i+'PR'] = std_AUPR_of.loc[i]
merged_mean=pd.Series(merged_mean)
merged_std=pd.Series(merged_std)
def pickcolor(n):
    return 'r' if n%2 == 0 else 'b'


plt.figure()
fig, ax = plt.subplots(figsize=(30,10))
for n, datapoint in  enumerate(merged_mean.index):
    ax.scatter(datapoint, merged_mean[datapoint], marker = 's' , color = pickcolor(n))
 #   ax.errorbar(datapoint, merged_mean[datapoint], yerr=merged_std[datapoint], color = pickcolor(n), alpha=0.5)
ax.grid()
plt.yticks()
plt.xticks(rotation=45)
FIGNAME = IMG_DIR + 'AUCplots.png'
plt.savefig(FIGNAME,  bbox_inches='tight')
#%%
from sklearn.metrics import roc_curve, precision_recall_curve,  auc

#%%# Use kegg_kpi+ubiq trained on pcomp+ubiq+kegg_kpi network to classify pcomp:
#training set: training_labels, features_table

#VALIDATION set
validation_labels, validation_features, _, _,_= read_features(OUTDIR, DATE, which=['p'], whichinbase=['k','u','p'])

X_test=np.array(pd.DataFrame(StandardScaler().fit_transform(validation_features)))
y_test=np.array(validation_labels)

y_predicted=classifier.predict_proba(np.array(X_test))#0 for class 0, 1 for class 1
y_pred_train=classifier.predict_proba(np.array(pd.DataFrame(StandardScaler().fit_transform(features_table))))
# enrichments
import scipy.stats as st
def do_compare_predictions_on_datasets(y_p1, y_p2,t):
    relevant_p1=len(np.where(y_p1>t)[0])
    relevant_p2=len(np.where(y_p2>t)[0])
    p1=round(relevant_p1/len(y_p1),3)
    p2=round(relevant_p2/len(y_p2),3)
    #def do test
    return p1, p2

thresholds = [0.55,0.6,0.7] #TODO ceck dovrebbe essere cosi: xke i + sono 0 e y_predicted  è probability of 1
plus_pcomp=y_predicted[:,0]
plus_ku=y_pred_train[:,0]
minus_pcomp=y_predicted[:,1]
minus_ku=y_pred_train[:,1]
tot_pcomp=len(plus_pcomp)
tot_ku=len(plus_ku)
for t in thresholds:
    # see if there is enrich of positives in pcomplex
    print('t',t)
    print('+',do_compare_predictions_on_datasets(plus_pcomp, plus_ku,t))

    print('-',do_compare_predictions_on_datasets(minus_pcomp, minus_ku,t))
    
    # wht does enriched mean?#1) ci sn piu + che rando/ 2) + positivi che negli alitr dataset?
    # qst sarebbe un t test o  test o ranked sums vibe no?
    # gsea e' x vedere se dei gene of interest (like of a pathway)
    # are more at the top or bottom of a ranked list
    # of differentially expressed other things.. quindi
    # direi la 2...: check if 2 distr are different (se una e' piu grande dell altra)
    #spoiler: fallisce
    stat, pval= st.ranksums(y_predicted[:,0], y_pred_train[:,0], alternative='greater')
    print('pval +',round(pval,2))
    stat, pval= st.ranksums(y_predicted[:,1], y_pred_train[:,1], alternative='greater')
    print('pval -',round(pval,2))
    # proportion test: (qll ce stavo facendo con do_compare..)
    # applica threshold, get binary distr, see if they differ.
    cont_table=np.array([[len(plus_pcomp), len(plus_ku)],\
                         [len(minus_pcomp),len(minus_ku)]])
    odds_ratio, f_p_value = st.fisher_exact(cont_table, alternative='greater') #todo check alternative, less or greater give s the same alue
    print('fisher exact test that one population has more positives', round(f_p_value,2))