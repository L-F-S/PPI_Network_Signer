#!/usr/bin/env python
# coding: utf-8



import os
import pandas  as pd
import numpy  as np
import matplotlib.pyplot as plt
import networkx
import collections
from time import time
from IPython.display import clear_output
from joblib import Parallel, delayed
from data_preproc import remove_dupes, add_missing_edges,  translate_ind, remove_flips,  read_alias2geneid, get_targets, graph_from_dataframe, remove_proteins_from_experiments
from signed_functions import  random_even_subsample,  create_defective_columns,propagate,score, defective_norm_matrix_from_columns, generate_similarity_matrix, create_the_features_different_knockouts, custom_k_fold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


##############################################################################
#  INPUTS
##############################################################################

SPECIES = "S_cerevisiae"


HOME_DIR  =  "G:" +os.sep+"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# HOME_DIR  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
#HOME_DIR  =  "/content/drive" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 

GENE_INFO_DIRECTORY = HOME_DIR+"07-2020_ANAT"+ os.sep +"downloads"+ os.sep +"kegg"+ os.sep
DATA_DIR=HOME_DIR+"yeast oriented and directedd network"+os.sep+'data'+os.sep
DIRECTED_DIR = HOME_DIR+"yeast oriented and directedd network"+os.sep
#FEATURES_DIR = HOME_DIR+"yeast oriented and directedd network"+os.sep+"features_creation_data"+os.sep
HOLSTEGE_DIR= DATA_DIR+'Holstege_S1Data'+os.sep
IMG_DIR=HOME_DIR+"yeast oriented and directedd network"+os.sep+'imgs'+os.sep

DATANAMES=['scD2DnewHOL','scD2DHOL', 'SchD2DHOL','SchD2DSch','SchSchHol', 'SchSchSch','SchUndHol',\
           'SchUndSch','scUndHol','scUndSch','scscHol','scscSch']

DATANAME='scnewD2DHOL'
DATASFX='.csv'
TRANSLATE_LABELS=False


# Propagation parameters:
PROPAGATE_ALPHA = 0.6
PROPAGATE_EPSILON = 1e-5
PROPAGATE_ITERATIONS = 100

if ("D2D" in DATANAME) or ('Und' in DATANAME):
    TRANSLATE_LABELS=True # True with Und and D2D as base net data, False with Sch/sc as base net data

# Initioalize dictionary of aliases
if (TRANSLATE_LABELS):
    alias_2geneid=read_alias2geneid(GENE_INFO_DIRECTORY, SPECIES)

##############################################################################
# ## Get training set:
##############################################################################

print("\n>>>>>>> getting training set:")

# from Patkar and Sharan 2018
training_label=pd.read_csv(DATA_DIR+"yeast_kegg_signs.txt", sep="\t",header=None, skiprows=1, index_col=(0,1)).dropna()
training_label=training_label[training_label[3]!='indirect effect']
training_label.drop(axis=1, labels=3, inplace=True)
training_label[2].replace({"activation":0,"inhibition":1}, inplace=True)
training_label=pd.concat([training_label,pd.read_csv(DATA_DIR+"yeast_kpi.txt", sep="\t",header=None, skiprows=1, index_col=(0,1)).dropna()])

if DATANAME.startswith('Sch'):
    training_label=pd.concat([training_label,pd.read_csv(DATA_DIR+"yeast_pdi.txt", sep="\t",header=None, skiprows=1, index_col=(0,1)).dropna()])


print('Removing flipped edges')

training_label=remove_flips(training_label)


training_label = pd.Series(data=list(training_label[2]), index=training_label.index)
training_label.name = 2 # 11/08/2021 added perche improvvisamente il translate nn fungeva
# ma niente funziona oggi qui. non toccarlo, riguardalo domani
if TRANSLATE_LABELS:
    print('translate indexes:')
    training_label = translate_ind(training_label, alias_2geneid)
    # training_label.index=pd.Series(data=list(training_label), index=trnsl_ind)

print('Removing duplicates')    

training_label=remove_dupes(training_label)

# print("------> Nuber of interactions found", training_label.shape[0],"<------")
# print(training_label.value_counts())

#%%#############################################################################
# Load PPI Propagation base network
##############################################################################
# Network shape:
# index: node-node edge tuple
# col 2: edge probability score
# col 3: directionality (0 for undirected, 1 for directed)
# col 4: (if signed) sign (0 for positive, 1 for negative)

print("\n>>>>>>> loading base network to propagte on:")
network=pd.read_csv(HOME_DIR+"07-2020_ANAT"+os.sep +\
                        "S_cerevisiaeD2Doriented.txt",sep="\t", index_col = [0,1],usecols=([0,1,2,4]), header=None)
network.rename(columns={4:3}, inplace=True)
# from Patkar and Sharan 2018, keeping Nans as undirected, and not NAns as directed

# network=pd.read_csv(HOME_DIR+os.sep+"Data"+os.sep+"S_cerevisiae.net",sep="\t", index_col = [0,1], header=None)
# network=pd.read_csv(DATA_DIR+"yeast_kegg_signs.txt", sep="\t",header=None, skiprows=1, index_col=(0,1))
# network=network[network[3]!='indirect effect']
# network=network[network[2]!='expression']
# network=network[network[2]!='binding/association']
# network=network[network[2]!='dissociation']
# network[2].replace({"activation":0,"inhibition":1}, inplace=True)
# network=pd.concat([network,pd.read_csv(DATA_DIR+"yeast_kpi.txt", sep="\t",header=None, skiprows=1, index_col=(0,1))])
# network=pd.concat([network,pd.read_csv(DATA_DIR+"yeast_ppi.txt", sep="\t",header=None, skiprows=1, index_col=(0,1))])
# # col 3 : directionality (0 if undirected, 1 if directed). all signed are assumed to be directed, the rest is left undirected
# network[3]= network[2].notna().astype(int)
# network[2]=0.8

net_type='dir' #CHANGE HERE to 'undir'
# network=pd.read_csv(HOME_DIR+os.sep+"Data"+os.sep+"S_cerevisiae.net",sep="\t", index_col = [0,1], header=None)

#graph = networkx.from_pandas_edgelist(network.reset_index(), 0, 1, 2)

graph = graph_from_dataframe(network, net_type)
genes = sorted(graph.nodes)

# ## Check that nodes in knockout experiment holstege are present in ANAT base net:
# forse dei TF non ci saranno?. If not, remove them
print('removing proteins which are not present in propagation network from experiments ')

##############################################################################
# Get knockdown experiments (knockdown +  their targets)
##############################################################################

#%%# dataset:  Reimand deletion - targets set:
# agust 2021: da cambiare qst arte qui xke usiamo il holstege data non riemand xke funge meglio
print(">>>>>> getting knockout pairs set:")
df=pd.read_csv(DATA_DIR+'yeast_reimand.txt', skiprows=1, index_col=(0,1), usecols=(0,1,2), sep='\t', header=None)
print('translate index ')

df=translate_ind(df, alias_2geneid)
# df=pd.DataFrame(data=list(df[2]), index=trnsl_ind)
# print(df.shape)


plus_targets_of_deletion, minus_targets_of_deletion=get_targets(df)
plus_targets_of_deletion, minus_targets_of_deletion = remove_proteins_from_experiments(plus_targets_of_deletion, minus_targets_of_deletion, genes)

# ## check which labels are present in base net
# and add those that are not. SHOULD BE NOTHING FOR CURRENT BASE NET (04/05/2021)
# print("Adding missing edges to base network from training data:")
# network=add_missing_edges(network, training_label, net_type)

#%%
print("\n--------------- DATA PREPROCESSED SUCCESFULLY---------------")
print('- Positive knockout experiment targets:', len(plus_targets_of_deletion),'\n- Negative knockout experiment targets:',len(minus_targets_of_deletion))
print('- Labels (0 for potive and 1 for negative signe of directed interaction):\n',len(training_label.index))
print('- Network size (for propagation): ', len(network.index),'edges, \n\tof whch directed:', network[network[3]==1].shape[0] ) # da mette a posto!! TODO

##############################################################################
# # Features creation
##############################################################################



raw_matrix = networkx.to_scipy_sparse_matrix(graph, genes, format='csc')
matrix = generate_similarity_matrix(raw_matrix) #normalized similarity matrix
num_genes     = len(genes)
gene_indexes  = dict([(gene, index) for (index, gene) in enumerate(genes)]) #all genes present in matrix
print('Preparing for propagations\n- Number of genes:',num_genes)


#%% Parallel version:

print("-------------CREATING FEATURES------------------")
start=time()
# results=Parallel(n_jobs=7)(map(delayed(create_the_features_different_knockouts), training_label.index))
results=Parallel(n_jobs=7)(delayed(create_the_features_different_knockouts)\
                           (raw_matrix, edge, gene_indexes, matrix, plus_targets_of_deletion, minus_targets_of_deletion, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON)\
                               for edge in training_label.index)

print("time passed", time()-start)
data = pd.DataFrame().from_records(results, index=0).drop(0, axis=1)   
print(data.head())

data.to_csv(DATANAME+DATASFX)
#%% LOAD FEATURES. IF you have already created features, run from here

data=pd.read_csv(DIRECTED_DIR+'out/'+DATANAME+DATASFX, sep=',', index_col=0)

#index is not a tuple but a string, but it deos not matter for the plot, but l;et s reset 
# indexes for boht data and training_labels
#12/08/2021 FOR SOME REASON RELOADING THE scD2DHOL data has less lines than the training,
# so removing excess labels from trainingin this next line: which you need to run twice lol
for index in training_label.index:
    if not str(index) in data.index:
        training_label.drop(index, inplace=True)
    
data.reset_index(inplace=True)
data.drop(columns=['0'], inplace=True)

training_label.index=(range(len(training_label)))    
data, training_label = random_even_subsample(data, training_label, 1993)
#%%
print("-------------TRAINING CLASSIFIER----------------------")


fig, classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
    mean_roc_auc=custom_k_fold(LogisticRegression(max_iter=1000),5,\
                            pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns),\
                                training_label[:], DATANAME, IMG_DIR, metric="ROC") #todo only first 500 samples

fig, classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper,\
    mean_precrec_auc=custom_k_fold(LogisticRegression(max_iter=1000),5,\
                            pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns),\
                                training_label[:], DATANAME, IMG_DIR, metric="PrecRec")
print("ROC AUC:", DATANAME, np.round(mean_roc_auc,2))
print("PrecRec AUC:", DATANAME, np.round(mean_precrec_auc,2))


# todo: check ocnnectivity of new graph with added yeasteract interactions
