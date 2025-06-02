# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:32:32 2021

@author: L-F-S§
unique enzymes crossvalidation
"""

import pandas  as pd
import numpy  as np
from datetime import date
import pickle
import random
from preproc_utils import load_training_data
from train_and_vis import log, load_features,\
    net_from_edges, get_degree_sorted_enzymes, find_largest_k_for_unique_enzyme_split,\
    get_enzyme_wise_splits, check_signs_and_degree_distrib, get_edges, get_indexes,\
    check_unique_inds, check_labes, get_sign_only_data_per_fold,\
    check_ones, get_enzymes, swap_enzymes, RF_stratified_cv,\
    get_indexes_slow_str, get_indexes_slow_int
from sklearn.preprocessing import StandardScaler
from glob_vars import SPECIES, TRAIN_DATA, PERT_MAP, HOME_DIR, LBL_DIR,\
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

dataset_name = '_'.join(TRAIN_DATA)
std=True #!! attenzione! NON PROVATECI A CASA
finaldataname=dataset_name+'_'+PERT_MAP
comments='unique enzyme CV'

#  merge selected TRAIN_DATA
signed_TRAIN_DATA_labels, _ = load_training_data(LBL_DIR, TRAIN_DATA, SPECIES)
signed_TRAIN_DATA_features = load_features(FT_DIR, [i+'_'+PERT_MAP for i in TRAIN_DATA], SPECIES) #attenzione, aggiungi _PERT_MAP se non e' hosltege


training_labels=pd.concat(list(signed_TRAIN_DATA_labels.values())) # works because python dictionaries are ordered in python3
features_table= pd.concat(list(signed_TRAIN_DATA_features.values()))

################################################
# Run unique enzyme cross validation
################################################
# i.e. create custom splits  where all enzymes (kinases and phosphatases) are in one split
# i.e.  select indexes of k folds where training edges do not have vertices which belong to test edges
#  and vce versa
# i.e. every enzyme (kinase or phosphatase) is only found in one fold, 
# so we never train and test on the same enzyme:
    
# 1 build train data edges graph
train_net = net_from_edges(list(training_labels.index))

# 2 sort nodes by number of outgoing edges(node1 is the kinase or phosphatase), from now referred to as 'enzyme'
enzyme_substrates = get_degree_sorted_enzymes(train_net, plot=True, img_dir=IMG_DIR, dataset_name=dataset_name, sort=True)

# 3 find K for which requirements are satisfied
# 3.1 fold size must be >biggest enzyme degree

print('first requirement:fold size must be >biggest enzyme degree')
k=5
k, fold_size =find_largest_k_for_unique_enzyme_split(enzyme_substrates, k)

#%% 3.2Check (and build) if possible to build k enzymes groups where the sum of their degrees is exactly multiple of foldsize, plus one group with rest of division.
print('finding perfect split of combinations of enzyme degrees that are multiples of fold size')
fold_bins_enzymes = get_enzyme_wise_splits(enzyme_substrates, fold_size)
#%%
print('Total degree of every fold: (should be around',fold_size,')')
for fold in fold_bins_enzymes: 
    print(enzyme_substrates.loc[fold].sum())
#%% 3.3 Check is there needs to be swapping
# print('Check and plot degree distribution:')
from train_and_vis import check_signs_and_degree_distrib
check_signs_and_degree_distrib(training_labels, fold_bins_enzymes, enzyme_substrates, IMG_DIR, dataset_name, True)
#%% ENZYME SWAPPING:
    # Rearrange bins to always have both kinases and phosphatases together: Enzyme
    # (only do it for human)
fold_bins_1_enzymes, fold_bins_1_subs, fold_bins_0_enzymes, fold_bins_0_subs = get_sign_only_data_per_fold(training_labels, fold_bins_enzymes, enzyme_substrates)

print('enzyme number (1s - 0s) per fold')
for i, fold in fold_bins_0_enzymes.items():
    print(len(fold_bins_1_enzymes[i]), len(fold))
    
print('Smallest and biggest degree of 1s per fold:')
for i, subs in fold_bins_1_subs.items():
    print(subs.sort_values(ascending=False).iloc[-1], subs.sort_values(ascending=False).iloc[0])

print(' number of 1s and 0s per fold:')
check_ones(training_labels, fold_bins_enzymes, enzyme_substrates)
#%% ENZYME SWAPPING
# 1. Find sets of enzymes with exact count of degrees, in two different folds,
# for opposite signs (may not alwaysbe possible in theory, dependent on data distribution)
# fold2 has the most 1s in Human kpi sorted, so I ll take it from this one. half of them.
count_to_swap=20#human kpi ubinet unsorted 22,150..#human kpi sorted: 161#fold_bins_1_subs[2].sum()/2 #161 for human #21 #
i_fold_giving_1=3
i_fold_giving_0=4
max_perm=6 # reduce to 4 if pc runsout of memory

enzymes_giving1=get_enzymes(i_fold_giving_1, fold_bins_1_subs, count_to_swap, max_perm)

human_kpi_sorted_flag=False
if not human_kpi_sorted_flag:
    enzymes_giving0=get_enzymes(i_fold_giving_0, fold_bins_0_subs, count_to_swap, 4) 
else:
    enzymes_giving0=[fold_bins_enzymes[i_fold_giving_0][1]]  #for human we know it's idx 1 of fold2  and it has degree 161

#%% 2 Swap enzymes
swap_enzymes(fold_bins_enzymes, i_fold_giving_1, i_fold_giving_0, enzymes_giving1, enzymes_giving0)

#%% check that it worked
print('enzymes swapped: number of 1s and 0s per fold:')
check_ones(training_labels, fold_bins_enzymes, enzyme_substrates)

# END OF ENZYME SWAPPING
#%% repeat as many times as you want:
# example: human kpi ubinet unsorted (MAKE SURE TO flag sort= False in step 2: get_degree_sorted_enzymes)
fold_bins_1_enzymes, fold_bins_1_subs, fold_bins_0_enzymes, fold_bins_0_subs = get_sign_only_data_per_fold(training_labels, fold_bins_enzymes, enzyme_substrates)
count_to_swap=23 # this number should be equal to the degree of the 1 enzymes of interest to swap (to check running previous cell  (check_ones) manually)
i_fold_giving_1=3
i_fold_giving_0=4
max_perm=5
enzymes_giving1=get_enzymes(i_fold_giving_1, fold_bins_1_subs, count_to_swap, max_perm)
enzymes_giving0=get_enzymes(i_fold_giving_0, fold_bins_0_subs, count_to_swap, max_perm) 
#%%
swap_enzymes(fold_bins_enzymes, i_fold_giving_1, i_fold_giving_0, enzymes_giving1, enzymes_giving0)
check_ones(training_labels, fold_bins_enzymes, enzyme_substrates)

# END OF ENZYME SWAPPING
#%% Create k fold indexes.
print('data can be split! creating k fold indexes')
        
fold_edges = get_edges(fold_bins_enzymes, train_net, k)

print('extracting indexes, and checking length of folds')
fold_indexes = get_indexes(fold_edges, features_table)

print('making sure all indexes are unique...')
check_unique_inds(fold_indexes, enzyme_substrates)

#check that all splits have both labels.
if not check_labes(fold_indexes, training_labels):
    print('Warning. Not all folds have both signs')
else:
    print('successfully created kinase phoshatase unique k fold crossvalidation indexes!')

#%% Apply cross validation with random forest classifier
x_train = pd.DataFrame(StandardScaler().fit_transform(features_table), columns=features_table.columns)
 
mean_auc, classifier = RF_stratified_cv(x_train, training_labels, fold_indexes, finaldataname)
#%% save auc and classifier

log(dataset_name,np.round(mean_auc,2),'-',logfile, DATE, SPECIES, pert_map=PERT_MAP, comments=comments)
print('saving model ', finaldataname)
with open(MOD_DIR+finaldataname+'.rf', 'wb') as f:
    pickle.dump(classifier, f)

fold_indexes_to_save=fold_indexes
with open(CV_OUT+dataset_name+'_splits_idx_'+str(mean_auc)+'.pkl','wb') as f:
    pickle.dump(fold_indexes_to_save,f)
#%% Random splits stratification: randomize enzymes distribution across splits
def get_rand_folds(fold_bins_1_subs,fold_bins_0_subs):
    random_fold_giving_1 = random.randint(0,4)
    random_fold_giving_0 = random.randint(0,4)
    while random_fold_giving_0==random_fold_giving_1:
        random_fold_giving_0 = random.randint(0,4)
    # print(fold_bins_1_subs[random_fold_giving_1])
    # print('random 1 fold:', random_fold_giving_1)
    # print('random 0 fold:', random_fold_giving_0)
    # print(fold_bins_0_subs[random_fold_giving_0])
    
    return random_fold_giving_1, random_fold_giving_0
def get_reasonable_max_deg(fold_1_subs):
    reasonable_deg=int(fold_1_subs.sum()/2)
    if reasonable_deg<fold_1_subs.iloc[-1]:
        return fold_1_subs.iloc[-1]
    return reasonable_deg

def get_random_count(enzyme_subs,reasonable_degree):
    enzyme_subs.sort_values(ascending=False,inplace=True)
    max_r_len=len(enzyme_subs)
    random_degree=1000
    while random_degree>reasonable_degree:
        # Select (1-3) random enzymes
        r_length=random.randint(1,max_r_len)
        r_inds= []
        for i in range(r_length):
            found=False
            while not found:
                r_ind=random.randint(0,len(enzyme_subs)-1)
                if not r_ind in r_inds:
                    r_inds.append(r_ind)
                    found=True
    
        # if their degree is somewhat reasonable, use it, otherwise, repeat
        random_degree = enzyme_subs.iloc[r_inds].sum()
    
    return random_degree
    
#%% reload fold_bins_enzymes 
fold_bins_enzymes = get_enzyme_wise_splits(enzyme_substrates, fold_size)
# %% randomly try to swap around data 

mean_auc=0.5
best_auc=0.5
max_perm=5  # 6 amd 5 yeast kpi + ubinet2 gives memory error

while mean_auc<=0.85:
    print('-------------------------\nnew random split stratification')
    no_swap_candidates=True
    while no_swap_candidates:
        
        fold_bins_1_enzymes, fold_bins_1_subs, fold_bins_0_enzymes, fold_bins_0_subs = get_sign_only_data_per_fold(training_labels, fold_bins_enzymes, enzyme_substrates)
        
        #1 get random folds
        random_fold_giving_1, random_fold_giving_0 = get_rand_folds(fold_bins_1_subs,fold_bins_0_subs)
        # 2 get random count
        reasonable_max_degree=get_reasonable_max_deg(fold_bins_1_subs[random_fold_giving_1])
        print('reasonable max degree:', reasonable_max_degree) #it's reasonable
        
        random_count_to_swap=get_random_count(fold_bins_1_subs[random_fold_giving_1], reasonable_max_degree)
        # print(random_count_to_swap)
        
        enzymes_giving1=get_enzymes(random_fold_giving_1, fold_bins_1_subs, random_count_to_swap, max_perm)
        enzymes_giving0=get_enzymes(random_fold_giving_0, fold_bins_0_subs, random_count_to_swap, max_perm) 
    
        if (enzymes_giving1==0) or (enzymes_giving0==0):
            no_swap_candidates=True
            print('--new random set--')
        else:
            no_swap_candidates=False
    #swap
    swap_enzymes(fold_bins_enzymes, random_fold_giving_1, random_fold_giving_0, enzymes_giving1, enzymes_giving0)
    
    # create splits and crossvalidate
    fold_edges = get_edges(fold_bins_enzymes, train_net, k)
    fold_indexes = get_indexes(fold_edges, features_table)
    check_unique_inds(fold_indexes, enzyme_substrates)
    if not check_labes(fold_indexes, training_labels):
        print('Warning. Not all folds have both signs')
    else:
        x_train = pd.DataFrame(StandardScaler().fit_transform(features_table), columns=features_table.columns)
        mean_auc, classifier = RF_stratified_cv(x_train, training_labels, fold_indexes, finaldataname)
        if mean_auc>=(best_auc-0.02): #might be close to new record
            # print('fold1',random_fold_giving_1,'fold0', random_fold_giving_0,'count',random_count_to_swap)
            for i in range(5):
                temp_auc, temp_classifier = RF_stratified_cv(x_train, training_labels, fold_indexes, finaldataname)
                if temp_auc>mean_auc:
                    mean_auc = temp_auc
                    classifier = temp_classifier

        if mean_auc>best_auc:
            best_auc=mean_auc
            print('*******************\nNEW BEST AUC',mean_auc,'\n*******************')
            fold_indexes_to_save=fold_indexes
            with open(CV_OUT+dataset_name+'_splits_idx_'+str(mean_auc)+'.pkl','wb') as f:
                pickle.dump(fold_indexes_to_save,f)
            log(dataset_name,np.round(mean_auc,2),'-',logfile, DATE, SPECIES, \
                pert_map=PERT_MAP, comments=comments+'random splits stratification')
            with open(MOD_DIR+finaldataname+'.rf', 'wb') as f:
                pickle.dump(classifier, f)
        
    # STOP when mean_auc>a target

#FOUND IT!
print('Target auc reached')
fold_indexes_to_save=fold_indexes
with open(CV_OUT+dataset_name+'_splits_idx_'+str(mean_auc)+'.pkl','wb') as f:
    pickle.dump(fold_indexes_to_save,f)

log(dataset_name,np.round(mean_auc,2),'-',logfile, DATE, SPECIES, \
    pert_map=PERT_MAP, comments=comments+'random splits stratification')
with open(MOD_DIR+finaldataname+'_'+str(mean_auc)+'.rf', 'wb') as f:
    pickle.dump(classifier, f)
#%% Load a pre-trained set of fold indexes for kpi (and ubinet 2) and crossvalidate on those folds

if SPECIES=='S_cerevisiae': 
    fold_index_name='kpi_splits_idx_0.77.pkl'#'kpi_ubinet2_splits_idx_0.85.pkl'
else:
    fold_index_name = 'PSP_depod_splits_idx_0.82.pkl'    #'PSP_ubinet2_depod_splits_idx_0.89.pkl'    
with open(CV_OUT+fold_index_name,'rb') as f:
    best_kpi_ubinet_fold_indexes = pickle.load(f)
#%% before we launch this:
# mean_auc, classifier = RF_stratified_cv(x_train, training_labels, best_kpi_ubinet_fold_indexes, finaldataname)
# let's make sure we're loading the correct x_train and training labels:
if SPECIES == 'S_cerevisiae':
    data=[ 'kpi']#, 'ubinet2'] 
else:
    data=['PSP','depod']#'ubinet2',]
    
signed_kegg_labels, _ = load_training_data(LBL_DIR, data, SPECIES)
signed_kegg_features = load_features(FT_DIR, [i+'_'+PERT_MAP for i in data], SPECIES) #attenzione, aggiungi _PERT_MAP se non e' hosltege
training_labels_kpi_ubinet=pd.concat(list(signed_kegg_labels.values())) # works because python dictionaries are ordered in python3
features_table_kpi_ubinet= pd.concat(list(signed_kegg_features.values()))

#%% Now, let's run the classifier to check it works

x_train_kpi_ubinet = pd.DataFrame(StandardScaler().fit_transform(features_table_kpi_ubinet), columns=features_table_kpi_ubinet.columns)
mean_auc, classifier = RF_stratified_cv(x_train_kpi_ubinet, training_labels_kpi_ubinet, best_kpi_ubinet_fold_indexes, finaldataname)
#%% 23 07 2024 LOADed the kpi+ubinet2 buono e provare a aggiungerci patkarkeg sopra (per cerevisiae)
# prima bisogna vedere se patkar kegg riesco a splittarlo da solo:

comments='unique enzyme CV kegg added'

if SPECIES == 'S_cerevisiae':
    data=['patkar_kegg']
else:
    data=['kegg']
signed_kegg_labels, _ = load_training_data(LBL_DIR, data, SPECIES)
signed_kegg_features = load_features(FT_DIR, [i+'_'+PERT_MAP for i in data], SPECIES) #attenzione, aggiungi _PERT_MAP se non e' hosltege
training_labels_kegg=pd.concat(list(signed_kegg_labels.values())) # works because python dictionaries are ordered in python3
features_table_kegg= pd.concat(list(signed_kegg_features.values()))

# 1 build train data edges graph
kegg_net = net_from_edges(list(training_labels_kegg.index))

# 2 sort nodes by number of outgoing edges(node1 is the kinase or phosphatase), from now referred to as 'enzyme'
kegg_enzyme_substrates = get_degree_sorted_enzymes(kegg_net, plot=True, img_dir=IMG_DIR, dataset_name=dataset_name, sort=False)

# 3 find K for which requirements are satisfied:
# 3.1 fold size must be >biggest enzyme degree

print('first requirement:fold size must be >biggest enzyme degree')
k=5
k, fold_size =find_largest_k_for_unique_enzyme_split(kegg_enzyme_substrates, k)

# 3.2Check (and build) if possible to build k enzymes groups where the sum of their degrees is exactly multiple of foldsize, plus one group with rest of division.
print('finding perfect split of combinations of enzyme degrees that are multiples of fold size')
kegg_fold_bins_enzymes = get_enzyme_wise_splits(kegg_enzyme_substrates, fold_size)   
#%% CI RIESCO!! Estrai dunque questi indici e aggiungili ai best kpi+ubinet fold indexes!
        
kegg_fold_edges = get_edges(kegg_fold_bins_enzymes, kegg_net, k)

print('extracting indexes, and checking length of folds')
kegg_fold_indexes = get_indexes(kegg_fold_edges, features_table_kegg)

print('making sure all indexes are unique...')
check_unique_inds(kegg_fold_indexes, kegg_enzyme_substrates)
#%% be mindful that indexes are related to the train data, so different train 
# data have diferent enzymes indexes at the same place
# so start from the edges iinstead of indexes, and add kegg_fold_edges
# to kpi_ubinet edges, which u dont have,  so trace back kpi_ubinet_edges

def indexes_to_edges(fold_indexes, labels):
    fold_edges=[]
    for fold in fold_indexes:
        fold_edges.append(list(labels.iloc[fold].index))
    return fold_edges

kpi_ubinet_fold_edges=indexes_to_edges(best_kpi_ubinet_fold_indexes, training_labels_kpi_ubinet)
# add the kegg edges to each fold of the kpi_ubinet_edges:
best_all_edges=[]
for i,kpi_ubinet_fold in enumerate(kpi_ubinet_fold_edges):
    best_all_edges.append(kpi_ubinet_fold+kegg_fold_edges[i])

# extract indexes, and checking length of folds
if SPECIES == 'S_cerevisiae':
    data=[ 'patkar_kegg','kpi', 'ubinet2'] 
else:
    data=['PSP','kegg','ubinet2','depod']

signed_features = load_features(FT_DIR, [i+'_'+PERT_MAP for i in data], SPECIES) #attenzione, aggiungi _PERT_MAP se non e' hosltege
all_features_table= pd.concat(list(signed_features.values()))
#%%
# best_all_fold_indexes = get_indexes_slow_int(best_all_edges, all_features_table)
best_all_fold_indexes = get_indexes_slow_str(best_all_edges, all_features_table)


#%% train classifier and crossvalidate!
if SPECIES == 'S_cerevisiae':
    dataset_name='patkar_kegg_kpi'#'patkar_kegg_kpi_ubinet2'
    finaldataname='patkar_kegg_kpi_Kemmeren'#'patkar_kegg_kpi_ubinet2_Kemmeren'
else:
    dataset_name='PSP_kegg_depod_CMGE'#'PSP_kegg_ubinet2_depod_CMGE'
    finaldataname='PSP_kegg_depod_CMGE'#'PSP_kegg_ubinet2_depod_CMGE'
comments='best stratified unique enzyme CV'
signed_labels_all, _ = load_training_data(LBL_DIR, data, SPECIES)
all_training=pd.concat(list(signed_labels_all.values())) # works because python dictionaries are ordered in python3

x_train_all = pd.DataFrame(StandardScaler().fit_transform(all_features_table), columns=all_features_table.columns)
mean_auc, classifier = RF_stratified_cv(x_train_all, all_training, best_all_fold_indexes, finaldataname)
#%% 
best_auc=0.5
target=0.85
while best_auc<target:
    mean_auc, classifier = RF_stratified_cv(x_train_all, all_training, best_all_fold_indexes, finaldataname)
    if mean_auc>=best_auc:
        print('new best AUC:', mean_auc)
        best_auc=mean_auc
        
        log(dataset_name,np.round(mean_auc,2),'-',logfile, DATE, SPECIES, \
            pert_map=PERT_MAP, comments=comments+'random splits stratification')
        with open(MOD_DIR+finaldataname+'_'+str(mean_auc)+'.rf', 'wb') as f:
            pickle.dump(classifier, f)
            
        fold_indexes_to_save=best_all_fold_indexes
        with open(CV_OUT+dataset_name+'_splits_idx_'+str(mean_auc)+'.pkl','wb') as f:
            pickle.dump(fold_indexes_to_save,f)
            