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
from train_and_vis3_5 import k_fold, read_features, log, load_features, AUCs
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
INPUTDIR=DIRECTED_DIR+'features_and_labels'+os.sep+SPECIES+os.sep
OUTDIR=DIRECTED_DIR+'models_and_predictions'+os.sep+SPECIES+os.sep
IMG_DIR=HOME_DIR+"network_signing"+os.sep+'imgs'+os.sep+'v3'+os.sep+SPECIES+os.sep
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


#%%  merge selected datasets and perform crossvalidation
mean_AUROC_of = {}
mean_AUPR_of = {}
std_AUROC_of = {}
std_AUPR_of = {}
datasets = ['kpi','ubiq'] if SPECIES == 'S_cerevisiae' else ['kegg','PSP','depod','ubinet2']
LOGedgedataname = '_'.join(datasets)
pert_map = 'Holstege' if SPECIES == 'S_cerevisiae' else'K562gwps' # toodo implement fot other kinds of human stuff
finaldataname=LOGedgedataname+'_'+pert_map
#%%
signed_datasets_labels, _ = load_training_data(INPUTDIR, datasets, SPECIES)
signed_datasets_features = load_features(INPUTDIR, datasets, SPECIES)
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


classifier=RandomForestClassifier()#LogisticRegression(max_iter=1000)#
classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
    mean_AUROC_of[finaldataname], std_AUROC_of[finaldataname] =k_fold(classifier,5,\
                            features_table,training_labels, finaldataname, \
                                IMG_DIR, metric="ROC", plot=True) 

classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
    mean_AUPR_of[finaldataname], std_AUPR_of[finaldataname]=k_fold(classifier,5,\
                            features_table,training_labels, finaldataname,\
                                IMG_DIR, metric="PrecRec",plot=True)
        
print("ROC AUC:", finaldataname, np.round(mean_AUROC_of[finaldataname],2))
print("PrecRec AUC:", finaldataname, np.round(mean_AUPR_of[finaldataname],2))

log(LOGedgedataname,np.round(mean_AUROC_of[finaldataname],2),np.round(mean_AUPR_of[finaldataname],2),logfile, DATE, SPECIES, pert_map=pert_map)

with open(OUTDIR+finaldataname+'.rf', 'wb') as f:
    pickle.dump(classifier, f)
#%% 18/12/2022 no satisfactory validations
with open(OUTDIR+finaldataname+'.rf','rb') as f:
    classifier=pickle.load(f)

valid_data = 'patkar_kegg'
valid_datasets_labels, _ = load_training_data(INPUTDIR, [valid_data], SPECIES)
valid_datasets_features = load_features(INPUTDIR, [valid_data], SPECIES)
roc_auc, pr_auc = AUCs(classifier,valid_datasets_features[valid_data] , valid_datasets_labels[valid_data]  )
print(valid_data)
print('ROC AUC:', roc_auc,'\n PRAUC:', pr_auc)
#%% enrich on ubiq.


allfaeatures = load_features(INPUTDIR, ['kegg', 'p_complex', 'netedgesI','netedgesII','netedgesIII',\
                               'netedgesIV','netedgesV','netedgesVI'], SPECIES)
allfaeatures= pd.concat(list(allfaeatures.values()))
#%% from all features, remove the edges present in kpi, patkar_kegg, and ubiq
edges_to_remove, _ = load_training_data(INPUTDIR, ['ubiq','patkar_kegg','kpi'], SPECIES)
edges_to_remove = pd.concat(list(edges_to_remove.values()))
edges_to_remove = edges_to_remove.index
#%% running this cos netedgesI----IV were made to compare vs pcomplex, and with other kegg data
to_remove = set()
to_keep = set()
for i,edge in enumerate(allfaeatures.index):
    if edge in edges_to_remove:
        to_remove.add(i)
    else:
        to_keep.add(i)
        #%%
netfeatures=allfaeatures.iloc[list(to_keep)]
#%%
with open(OUTDIR+ 'patkar_kegg_kpi_Holstege.rf','rb') as f:
    classifer = pickle.load(f)

ubiq_features = load_features(INPUTDIR, ['ubiq'], SPECIES)
ubiq_features = ubiq_features['ubiq']
#%% Make predictions:

ubiq_pred = classifier.predict_proba(StandardScaler().fit_transform(ubiq_features))
net_pred = classifier.predict_proba(StandardScaler().fit_transform(netfeatures))
#%% test enrichment of negatives (1) on ubiquitin vs net
import scipy.stats as st

thresholds = [0.55,0.6,0.7] #TODO ceck dovrebbe essere cosi: xke i + sono 0 e y_predicted  è probability of 1
# y_predicted[:,0] : class 0 = + # y_predicted[:,1] : class 1 = -
plus_pcomp=ubiq_pred[:,0]
plus_ku=net_pred[:,0]
minus_pcomp=ubiq_pred[:,1]
minus_ku=net_pred[:,1]
tot_pcomp=len(plus_pcomp)
tot_ku=len(plus_ku)
for t in thresholds:
    print('threshold:',t)
    cont_table=np.array([[len(np.where(plus_pcomp>=t)[0]), len(np.where(plus_ku>=t)[0])],\
                         [len(np.where(minus_pcomp<t)[0]),len(np.where(minus_ku<t)[0])]])
    odds_ratio, f_p_value = st.fisher_exact(cont_table, alternative='less') 
    print('hypergeometric test that one population has more negatives. Pvalue', round(f_p_value,3))
#%% plots
plt.hist(ubiq_pred[:,0], weights=np.ones(len(ubiq_pred[:,0]))/len(ubiq_pred[:,0]), label='ubiq')
plt.hist(net_pred[:,0], weights=np.ones(len(net_pred[:,0]))/len(net_pred[:,0]), label='net', alpha=0.4)
plt.title('Distribution of positive predictions')
plt.legend()
plt.show()
