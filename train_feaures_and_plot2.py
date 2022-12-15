# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:32:32 2021

@author: los4
"""

import os
import pandas  as pd
import numpy  as np
from datetime import date
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
from preproc_utils import load_training_data
from train_and_vis3 import custom_k_fold, read_features, log, load_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle


##############################################################################
#  INPUTS
##############################################################################
SPECIES = "H_sapiens"#
SPECIES =  "S_cerevisiae" #

HOME_DIR  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# HOME_DIR  =  "G:" +os.sep+"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 

DATA_DIR = HOME_DIR+'Data'+os.sep
SPECIES_DATA_DIR = DATA_DIR+SPECIES+os.sep
DIRECTED_DIR = HOME_DIR+"network_signing"+os.sep
OUTDIR=DIRECTED_DIR+'features_and_labels'+os.sep+SPECIES+os.sep
PRED_DIR= DIRECTED_DIR+'predictions'+os.sep
IMG_DIR=HOME_DIR+"network_signing"+os.sep+'imgs'+os.sep+'v3'+os.sep
logfile=DIRECTED_DIR+os.sep+'validation_out'+os.sep+'crossvalidationslog.txt'
DATE=str(date.today())


# Initioalize dictionary of aliases
DOWNSAMPLING = False 
dwsstring=''
if DOWNSAMPLING:
    dwsstring = '.DS'


# store values for different datasets
# training_labels_of ={}
# features_table_of = {}


#%% 15 12 2022
import pickle 
with open(SPECIES_DATA_DIR+'alias_2geneid.pkl','rb') as f:
    alias_2geneid = pickle.load(f)

old_kegg_labels=pd.read_csv('data/yeast_kegg_signs.txt', sep='\t')
old_kegg_labels['A'].replace(alias_2geneid, inplace=True)
old_kegg_labels['B'].replace(alias_2geneid, inplace=True)
old_kegg_labels.rename(columns={'A':0,'B':1}, inplace=True)
old_kegg_labels.index=pd.MultiIndex.from_frame(old_kegg_labels[[0,1]])
old_kegg_labels.replace({"activation":0,"inhibition":1, 'binding/association':0, 'expression':0, 'dissociation':1}, inplace=True)
# old_kegg_labels.dropna(inplace=True)
old_kegg_labels.drop(axis=1, labels=[0,1,'annotation2'], inplace=True)

old_kpi_labels=pd.read_csv('data/yeast_kpi.txt', sep='\t')
old_kpi_labels['A'].replace(alias_2geneid, inplace=True)
old_kpi_labels['B'].replace(alias_2geneid, inplace=True)
old_kpi_labels.rename(columns={'A':0,'B':1}, inplace=True)
old_kpi_labels.index=pd.MultiIndex.from_frame(old_kpi_labels[[0,1]])
old_kpi_labels.dropna(inplace=True)
old_kpi_labels.drop(axis=1, labels=[0,1], inplace=True)

old_all_features=pd.read_csv(OUTDIR+'old'+os.sep+'Undirected_training_features.csv', sep=',', index_col=(0,1))
# # filter out labels
old_kegg_ind = []
todrop=[]
# index_of_label = lbl.index
for n,i in enumerate(old_kegg_labels.index):
    if i in old_all_features.index:
        old_kegg_ind.append(i)
    else:
        todrop.append(i)
old_kegg_features = old_all_features.loc[old_kegg_ind]
old_kegg_labels.drop(todrop, inplace=True)

old_kpi_ind = []
todrop=[]
for i in old_kpi_labels.index:
    if i in old_all_features.index:
        old_kpi_ind.append(i)
    else:
        todrop.append(i)
old_kpi_features = old_all_features.loc[old_kpi_ind]
old_kpi_labels.drop(todrop, inplace=True)
#####
old_kegg_features.reset_index(inplace=True)
old_kegg_labels = pd.Series(data=old_kegg_labels['annotation1'], index=old_kegg_labels.index)
old_kpi_features.reset_index(inplace=True)
old_kpi_labels = pd.Series(data=old_kpi_labels['sign'], index=old_kpi_labels.index)
mean_AUROC_of = {}
mean_AUPR_of = {}
std_AUROC_of = {}
std_AUPR_of = {}
#posthumously write the old kegg (so its not really him but oh well, close enough i guess right?)
old_kegg_weitghts = pd.Series(0.8, index=old_kegg_labels.index)
old_kegg_weitghts.name=2
old_kegg_weitghts.reset_index().to_csv(OUTDIR+'old_kegg.w8.tsv', sep='\t', header=None, index=False)
old_kegg_labels.reset_index().to_csv(OUTDIR+'old_kegg.lbl.tsv', sep='\t', header=None, index=False)
old_kegg_features.to_csv(OUTDIR+'old_kegg.ft.csv',sep=',', index=None)
#%%
for training_labels, features_table, LOGedgedataname in [(old_kegg_labels,old_kegg_features,'oldkeggv3'),(old_kpi_labels,old_kpi_features,'oldkpiv3')]:
    print(LOGedgedataname)
    pert_map = 'Holstege' 
    finaldataname=LOGedgedataname+'_'+pert_map
    classifier=RandomForestClassifier()#LogisticRegression(max_iter=1000)#
    classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
        mean_AUROC_of[finaldataname], std_AUROC_of[finaldataname] =custom_k_fold(classifier,5,\
                                pd.DataFrame(StandardScaler().fit_transform(features_table), columns=features_table.columns),\
                                    training_labels, finaldataname, IMG_DIR, metric="ROC", plot=False) 
    print("ROC AUC:", finaldataname, np.round(mean_AUROC_of[finaldataname],2))

    log(LOGedgedataname,np.round(mean_AUROC_of[finaldataname],2),'--',\
        logfile, DATE, SPECIES, pert_map=pert_map)

#%%  merge selected datasets and perform crossvalidation
mean_AUROC_of = {}
mean_AUPR_of = {}
std_AUROC_of = {}
std_AUPR_of = {}
datasets = ['kegg','kpi', 'ubiq'] if SPECIES == 'S_cerevisiae' else ['kegg','PSP','depod','ubinet2']
signed_datasets_labels, _ = load_training_data(OUTDIR, datasets, SPECIES)
signed_datasets_features = load_features(OUTDIR, datasets, SPECIES)
# def check_dupes(data):
#     seen = {}
#     dupes = []

#     for x in data.index:
#         if x not in seen:
#             seen[x] = 1
#         else:
#             if seen[x] == 1:
#                 dupes.append(x)
#             seen[x] += 1
#     return dupes
# print(check_dupes(pd.concat(list(signed_datasets_labels.values()))))

training_labels=pd.concat(list(signed_datasets_labels.values())) # works because python dictionaries are ordered in python3
features_table= pd.concat(list(signed_datasets_features.values()))
### 
#todo what if it s cerevisiae?
LOGedgedataname = '_'.join(datasets)
pert_map = 'Holstege' if SPECIES == 'S_cerevisiae' else'K562gwps' # toodo implement fot other kinds of human stuff
finaldataname=LOGedgedataname+'_'+pert_map

classifier=RandomForestClassifier()#LogisticRegression(max_iter=1000)#
classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
    mean_AUROC_of[finaldataname], std_AUROC_of[finaldataname] =custom_k_fold(classifier,5,\
                            pd.DataFrame(StandardScaler().fit_transform(features_table), columns=features_table.columns),\
                                training_labels, finaldataname, IMG_DIR, metric="ROC", plot=True) 

classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
    mean_AUPR_of[finaldataname], std_AUPR_of[finaldataname]=custom_k_fold(classifier,5,\
                            pd.DataFrame(StandardScaler().fit_transform(features_table), columns=features_table.columns),\
                                training_labels, finaldataname, IMG_DIR, metric="PrecRec",plot=True)
print("ROC AUC:", finaldataname, np.round(mean_AUROC_of[finaldataname],2))
print("PrecRec AUC:", finaldataname, np.round(mean_AUPR_of[finaldataname],2))

log(LOGedgedataname,np.round(mean_AUROC_of[finaldataname],2),np.round(mean_AUPR_of[finaldataname],2),logfile, DATE, SPECIES, pert_map=pert_map)

#%%
from sklearn.metrics import roc_curve, precision_recall_curve,  auc

#%%# Use kegg_kpi+ubiq trained on pcomp+ubiq+kegg_kpi network to classify pcomp:
#training set: training_labels, features_table

#VALIDATION set
validation_labels, _ = load_training_data(OUTDIR, ['p_complex', 'ubiq'], SPECIES)
validation_features = load_features(OUTDIR, ['p_complex'], SPECIES)

X_test=np.array(pd.DataFrame(StandardScaler().fit_transform(validation_features['p_complex'])))
y_test=np.array(validation_labels['p_complex'])

# predict orientations of validation (p_complex)
y_predicted=classifier.predict_proba(np.array(X_test))#0 for class 0, 1 for class 1
# see: print(classifier.classes_) to see the order of classes
# y_predicted[:,0] : class 0 = + # y_predicted[:,1] : class 1 = -
#%%
# predict orientations of all unsigned network edges (big data, takes some time)
X_net = load_features(OUTDIR, ['netedgesI','netedgesII','netedgesIII',\
                               'netedgesIV','netedgesV','netedgesVI'], SPECIES)
X_net= pd.concat(list(X_net.values()))
y_pred_net=classifier.predict_proba(np.array(pd.DataFrame(StandardScaler().fit_transform(X_net))))
with open(PRED_DIR+'unsigned_net_pred','wb') as f:
    pickle.dump(y_pred_net,f)
#%% Calculate enrichments of positive edges
import scipy.stats as st
def do_compare_predictions_on_datasets(y_p1, y_p2,t):
    '''Compare proportions between two populations 1 and 2''' #checked, it works correctly with the correct indexing 
    relevant_p1=len(np.where(y_p1>t)[0])
    relevant_p2=len(np.where(y_p2>t)[0])
    p1=round(relevant_p1/len(y_p1),2)
    p2=round(relevant_p2/len(y_p2),2)
    #2 proportions z test 
    z, p_val = proportions_ztest(
        count=[relevant_p1,relevant_p2],
        nobs=[len(y_p1), len(y_p2)],
        alternative='larger')
    return p1, p2, p_val


thresholds = [0.55,0.6,0.7] #TODO ceck dovrebbe essere cosi: xke i + sono 0 e y_predicted  è probability of 1
# y_predicted[:,0] : class 0 = + # y_predicted[:,1] : class 1 = -
plus_pcomp=y_predicted[:,0]
plus_ku=y_pred_net[:,0]
minus_pcomp=y_predicted[:,1]
minus_ku=y_pred_net[:,1]
tot_pcomp=len(plus_pcomp)
tot_ku=len(plus_ku)
for t in thresholds:
    # see if there is enrich of positives in pcomplex
    print('t',t)
    p1, p2, pval=do_compare_predictions_on_datasets(plus_pcomp, plus_ku,t)
    print('proportion of +:  for P_complex', p1, 'for all unsigned edges', p2, 'z-test for proportions p_val, hypotehsis p1>p2', pval)
    p1, p2, pval=do_compare_predictions_on_datasets(minus_pcomp, minus_ku,t)
    print('proportion of -:  for P_complex', p1, 'for all unsigned edges', p2, 'z-test for proportions p_val hypotehsis p1>p2 (more negatives in p_complex)', pval)
    
    # enrichment: 1) ci sn piu + che rando/ 2) + positivi che negli alitr dataset?
    # qst sarebbe un t test o  test o ranked sums vibe no?
    # gsea e' x vedere se dei gene of interest (like of a pathway)
    # are more at the top or bottom of a ranked list
    # of differentially expressed other things.. quindi
    # direi la 2...: check if 2 distr are different (se una e' piu grande dell altra)
    #spoiler: fallisce
    #check if the positively signed are more represented at the top of the list or at the bottom
    stat, pval= st.ranksums(y_predicted[:,0], y_pred_net[:,0], alternative='greater')
    print('ranksums test that distribution of + is greater in p_complex data. pvalue:',round(pval,5))
    stat, pval= st.ranksums(y_predicted[:,1], y_pred_net[:,1], alternative='greater')
    print('ranksums test  that distribution of - is greater in p_complex data. pvalue:',round(pval,5))
    # proportion test: (qll ce stavo facendo con do_compare..)
    # applica threshold, get binary distr, see if they differ.
    cont_table=np.array([[len(np.where(plus_pcomp>=t)[0]), len(np.where(plus_ku>=t)[0])],\
                         [len(np.where(minus_pcomp<t)[0]),len(np.where(minus_ku<t)[0])]])
    odds_ratio, f_p_value = st.fisher_exact(cont_table, alternative='greater') 
    print('hypergeometric test that one population has more positives. Pvalue', round(f_p_value,3))
    
#%%
pcom_m=sum(y_predicted[:,0])/len(y_predicted[:,0])
net_m = sum(y_pred_net[:,0])/len(y_pred_net[:,0])
print('mean normalized weight for + edge for p_comp:',pcom_m)
print('mean normalized weight for + edge for unsigned net:',net_m)
