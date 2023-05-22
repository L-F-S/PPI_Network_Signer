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

DATA_DIR = HOME_DIR+'Data'+os.sep+SPECIES+os.sep
DIRECTED_DIR = HOME_DIR+"network_signing"+os.sep
INPUTDIR=DIRECTED_DIR+'features_and_labels'+os.sep+SPECIES+os.sep
OUTDIR=DIRECTED_DIR+'models_and_predictions'+os.sep+SPECIES+os.sep
IMG_DIR=HOME_DIR+"network_signing"+os.sep+'imgs'+os.sep+'v3'+os.sep+SPECIES+os.sep
logfile=DIRECTED_DIR+os.sep+'validation_out'+os.sep+'crossvalidationslog.txt'
DATE=str(date.today())

mean_AUROC_of = {}
mean_AUPR_of = {}
std_AUROC_of = {}
std_AUPR_of = {}
if SPECIES == 'S_cerevisiae':
    datasets = ['patkar_kegg','kpi'] #'ubiq', 'kegg'
    pert_map = 'Holstege'#'reimand'
else:
    datasets = ['kegg','PSP','depod']
    pert_map =  'CMGE' #'ADPBH_1000'  ##   human: 'K562gwps''ADPBH_1 'ADPBH_1000'
dataset_name = '_'.join(datasets)
sampling = ''#'_DS'
finaldataname=dataset_name+'_'+pert_map+sampling
comments=sampling
#%%  merge selected datasets and perform crossvalidation
signed_datasets_labels, _ = load_training_data(INPUTDIR, datasets, SPECIES)
# signed_datasets_labels['kegg'] = signed_datasets_labels['kegg'].iloc[1:]#temp todo: only for kegg CMGE che x qlc motivo la features table manca el primo coos
signed_datasets_features = load_features(INPUTDIR, [i+'_'+pert_map for i in datasets], SPECIES) #attenzione, aggiungi _pert_map se non e' hosltege


training_labels=pd.concat(list(signed_datasets_labels.values())) # works because python dictionaries are ordered in python3
features_table= pd.concat(list(signed_datasets_features.values()))

#%% #%%  12 01 2023 features knockout validation
with open(INPUTDIR+'plus_targets'+'_'+pert_map+'.pkl', 'rb') as f:
    plus_targets_of_deletion=pickle.load( f)
with open(INPUTDIR+'minus_targets'+'_'+pert_map+'.pkl','rb') as f:
    minus_targets_of_deletion=pickle.load(f)
#%%
# if plus_targets_of_deletion.keys() == minus_targets_of_deletion.keys(): #doublecheck
#     val_ko=int(len(plus_targets_of_deletion)*0.2)
#     features=len(features_table.columns)
#     plus_feat_cols=features_table.columns[0:int(features/2)] 
#     minus_feat_cols=features_table.columns[int(features/2):]
    
#     test_plus_features = features_table[features_table.columns[:int(features/2)-val_ko]]
#     validation_plus_features = features_table[features_table.columns[int(features/2)-val_ko:int(features/2)]]
    
#     test_minus_features = features_table[features_table.columns[int(features/2):features-val_ko]]
#     validation_minus_features = features_table[features_table.columns[int(features)-val_ko:]]
# features_table=pd.concat([test_plus_features,test_minus_features],axis=1)
#%% human: 66 features >=1000 targets + t>1
asd=signed_datasets_features = load_features(INPUTDIR, [i+'_ADPBH_1' for i in datasets], SPECIES)
features_table1=pd.concat(list(asd.values()))
with open(INPUTDIR+'plus_targets'+'_ADPBH_1.pkl', 'rb') as f:
    plus_1=pickle.load( f)
with open(INPUTDIR+'plus_targets'+'_ADPBH_1000.pkl', 'rb') as f:
    plus_1000=pickle.load( f)
with open(INPUTDIR+'minus_targets'+'_ADPBH_1.pkl','rb') as f:
    minus_1=pickle.load(f)
with open(INPUTDIR+'minus_targets'+'_'+pert_map+'.pkl','rb') as f:
    minus_1000=pickle.load(f)
import collections
common_keys = collections.defaultdict(list)
for i,k in enumerate(plus_1.keys()):
    if k in plus_1000.keys():
        common_keys[k].append([i])
for j, k in enumerate(plus_1000.keys()):
    if k in common_keys.keys():
        common_keys[k].append(j)
print(len(common_keys.keys())) #66
#%%
#11091: [[387], 388]
same_cols=[]
for c in features_table.columns:
    for c1 in features_table1.columns:
        if features_table[c][features_table[c]==features_table1[c1]].sum() > 0:
            print(c,c1)
            same_cols.append((c,c1))
            

#i think they should be the same ma non lo sono
#%% merged datasets CV:
classifier=RandomForestClassifier()#LogisticRegression(max_iter=1000)#
classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
    mean_AUROC_of[finaldataname], std_AUROC_of[finaldataname] =k_fold(classifier,5,\
                            features_table,training_labels, finaldataname, \
                                IMG_DIR, metric="ROC", downsampling=sampling,\
                                    plot=False) 

# classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
#     mean_AUPR_of[finaldataname], std_AUPR_of[finaldataname]=k_fold(classifier,5,\
#                             features_table,training_labels, finaldataname,\
#                                 IMG_DIR, metric="PrecRec", downsampling=sampling,
#                                 plot=False)
mean_AUPR_of[finaldataname]=0
std_AUPR_of[finaldataname]=0
print("ROC AUC:", finaldataname, np.round(mean_AUROC_of[finaldataname],2))
print("PrecRec AUC:", finaldataname, np.round(mean_AUPR_of[finaldataname],2))
#%%
log(dataset_name,np.round(mean_AUROC_of[finaldataname],2),\
    np.round(mean_AUPR_of[finaldataname],2),logfile, DATE, SPECIES,\
            pert_map=pert_map, comments=comments)
with open(OUTDIR+finaldataname+'.rf', 'wb') as f:
    pickle.dump(classifier, f)
#%% dataset_wise CV:
### 
for dataset_name, training_labels in signed_datasets_labels.items():
    print(dataset_name)
    finaldataname=dataset_name+'_'+pert_map+sampling
    current_features_table = signed_datasets_features[dataset_name+"_"+pert_map]
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
            pert_map=pert_map, comments=comments)
    
    with open(OUTDIR+finaldataname+'.rf', 'wb') as f:
        pickle.dump(classifier, f)
#%% 
with open(OUTDIR+finaldataname+'.rf','rb') as f:
    classifier=pickle.load(f)

# valid_data = 'patkar_kegg'
# valid_datasets_labels, _ = load_training_data(INPUTDIR, [valid_data], SPECIES)
# valid_datasets_features = load_features(INPUTDIR, [valid_data], SPECIES)
# roc_auc, pr_auc = AUCs(classifier,valid_datasets_features[valid_data] , valid_datasets_labels[valid_data]  )
# print(valid_data)
# print('ROC AUC:', roc_auc,'\n PRAUC:', pr_auc)
#%% enrich on ubiq. Generate net features, or....:
dataset_to_enrich = 'p_complex'
#%%
alldatanames= [i+'_'+pert_map  for i in ['netedgesI','netedgesII','netedgesIII',\
                                'netedgesIV','netedgesV','netedgesVI']]
allfaeatures = load_features(INPUTDIR, alldatanames, SPECIES)
allfaeatures= pd.concat(list(allfaeatures.values()))
# from all features, remove the edges present in kpi, patkar_kegg, and ubiq
edges_to_remove, _ = load_training_data(INPUTDIR, datasets+[dataset_to_enrich], SPECIES)
edges_to_remove = pd.concat(list(edges_to_remove.values()))
edges_to_remove = edges_to_remove.index
# running this cos netedgesI----IV were made to compare vs pcomplex, and with other kegg data
to_remove = set()
to_keep = set()
for i,edge in enumerate(allfaeatures.index):
    if edge in edges_to_remove:
        to_remove.add(i)
    else:
        to_keep.add(i)

netfeatures=allfaeatures.iloc[list(to_keep)]
#%% 
with open(OUTDIR+'netfeatures_'+finaldataname+'vs_'+dataset_to_enrich+'.ft.pkl','wb') as f:
    pickle.dump(netfeatures, f)
#%% ..or load net features
# with open(INPUTDIR+'netfeatures_'+finaldataname+'vs_'+dataset_to_enrich+'.ft.pkl','rb') as f: add may2023 also possible version
#     netfeatures=pickle.load(f)

#%%
# with open(OUTDIR+ 'patkar_kegg_kpi_Holstege.rf','rb') as f:
#     classifer = pickle.load(f)
features_to_enrich = load_features(INPUTDIR, [dataset_to_enrich+'_'+pert_map], SPECIES)
features_to_enrich  = features_to_enrich[dataset_to_enrich+'_'+pert_map]
#%% Make predictions:

to_enrich_pred = classifier.predict_proba(StandardScaler().fit_transform(features_to_enrich))
net_pred = classifier.predict_proba(StandardScaler().fit_transform(netfeatures))
# classifier.classes_
#%% test enrichment of negatives (1) on ubiquitin vs net: hyoergeometric test with different thresholds
import scipy.stats as st

thresholds = [0.55,0.6,0.7] #TODO ceck dovrebbe essere cosi: xke i + sono 0 e y_predicted  è probability of 1
# y_predicted[:,0] : class 0 = + # y_predicted[:,1] : class 1 = -
plus_sampled_population=to_enrich_pred[:,0]
plus_rest_of_population=net_pred[:,0]
minus_sampled_population=to_enrich_pred[:,1]
minus_rest_of_population=net_pred[:,1]
tot_pcomp=len(plus_sampled_population)
tot_ku=len(plus_rest_of_population)
for t in thresholds:
    print('->threshold:',t)
    cont_table=np.array([[len(np.where(minus_sampled_population>=t)[0]), len(np.where(minus_rest_of_population>=t)[0])],\
                         [len(np.where(plus_sampled_population<t)[0]),len(np.where(plus_rest_of_population<t)[0])]])
    odds_ratio, f_p_value = st.fisher_exact(cont_table, alternative='greater') 
    print('hypergeometric test that one population has more negatives.')
    print('proportion of negatives in sample of interest:',round(len(np.where(minus_sampled_population>=t)[0])/minus_sampled_population.shape[0],3))
    print('total',len(np.where(minus_sampled_population>=t)[0]))
    print('proportion of negatives in rest of population:',round(len(np.where(minus_rest_of_population>=t)[0])/minus_rest_of_population.shape[0],3))
    print('total',len(np.where(minus_rest_of_population>=t)[0]))
    print('Pvalue:', round(f_p_value,3))
#     #%%
#     # should be symmetrical:
#         # IT IS!!!!
# for t in thresholds:
#     print('->threshold:',t)
#     cont_table=np.array([[len(np.where(minus_sampled_population>=t)[0]), len(np.where(plus_sampled_population<t)[0])],\
#                          [len(np.where(minus_rest_of_population>=t)[0]),len(np.where(plus_rest_of_population<t)[0])]])
#     odds_ratio, f_p_value = st.fisher_exact(cont_table, alternative='greater') 
#     print('hypergeometric test that one population has more negatives.')
#     print('proportion of negatives in sample of interest:',round(len(np.where(minus_sampled_population>=t)[0])/minus_sampled_population.shape[0],3))
#     print('proportion of negatives in rest of population:',round(len(np.where(minus_rest_of_population>=t)[0])/minus_rest_of_population.shape[0],3))
#     print('Pvalue:', round(f_p_value,3))
#%% plots
plt.hist(to_enrich_pred[:,1], weights=np.ones(len(to_enrich_pred[:,1]))/len(to_enrich_pred[:,1]), label='ubiq')
plt.hist(net_pred[:,1], weights=np.ones(len(net_pred[:,1]))/len(net_pred[:,1]), label='net', alpha=0.4)
plt.title('Distribution of negative predictions')
plt.legend()
plt.show()
#%% test enrichment of negatives (1) on ubiquitin vs net: wilcoxon rank sums test
from scipy.stats import ranksums
print(ranksums(to_enrich_pred[:,1], net_pred[:,1], alternative='greater'))



############################
#%% Knockout validation part 2 (before 15 01 2023)
# we have our classifier,
# now, let's train
allfaeatures = load_features(INPUTDIR, ['kegg', 'p_complex', 'netedgesI','netedgesII','netedgesIII',\
                               'netedgesIV','netedgesV','netedgesVI'], SPECIES)
allfaeatures= pd.concat(list(allfaeatures.values()))
allfaeatures = allfaeatures[features_table.columns]
#%%
validation_knockouts= list(plus_targets_of_deletion.keys())[-int(len(plus_targets_of_deletion)*0.2):]
test_knockouts= list(plus_targets_of_deletion.keys())[:int(len(plus_targets_of_deletion)*0.2)]

test_plus_targets = {ko: plus_targets_of_deletion[ko] for ko in test_knockouts}
validation_plus_targets = {ko: plus_targets_of_deletion[ko] for ko in validation_knockouts}
test_minus_targets = {ko: minus_targets_of_deletion[ko] for ko in test_knockouts}
validation_minus_targets = {ko: minus_targets_of_deletion[ko] for ko in validation_knockouts}
#%%
plus=[]
minus=[]
for  k,i in validation_plus_targets.values():
    plus+=i
for  i in validation_minus_targets.values():
    minus+=i
#%% extract all edges insisting on plus(minus) target from the data, and 
# see for enrichment of plus (minus) sign
#maybe do it for all? or per target?
alledges=list(allfaeatures.index)
plus_edges=set()
minus_edges=set()
for edge in alledges:
    for plus_target in plus:
        if edge[1]==plus_target:
            plus_edges.add(edge)
    for minus_target in minus:
        if edge[1]==minus_target:
            minus_edges.add(edge)
#%%
plusfeatures=allfaeatures.loc[list(plus_edges)]
minusfeatures=allfaeatures.loc[list(minus_edges)]
#%%
with open(OUTDIR+'plusfeatures_valdation'+'.ft', 'wb') as f:
    pickle.dump(plusfeatures, f)
with open(OUTDIR+'minusfeatures_valdation'+'.ft', 'wb') as f:
    pickle.dump(minusfeatures, f)
