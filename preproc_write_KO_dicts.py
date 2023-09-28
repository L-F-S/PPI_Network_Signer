# -*- coding: utf-8 -*-
"""
Preprocesses perturbation dataset and signed datasets to create inputs for network_signer.py
Inputs are saved in /features_and_labels

v2:
    -remove DATANAME flags
    -only use knockdown_map as feature creation signatures
    -select base network (undirected, or Directed (daniel data))
    -load and add all datasets with signing info together and add to base net
    -create features for all datasets separately
    -only AFTER, train on different datasets and decide which are worth keeping for training and validation etc.
v3: 
    -add human
    -removeddirected base network option
"""

import os
import pickle
from preproc_utils import  add_edges_from_labels,graph_from_dataframe,\
    preprocess_signed_datasets, readname2geneid, get_perturbations_map,\
         extract_knockotut_effect_pairs_from_data, write, load_training_data
##############################################################################
#  INPUTS
##############################################################################
SPECIES = "H_sapiens"#
# SPECIES =  "S_cerevisiae" #

HOME_DIR  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# HOME_DIR  =  "G:" +os.sep+"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 

DATA_DIR = HOME_DIR+'Data'+os.sep+SPECIES+os.sep
DIRECTED_DIR = HOME_DIR+"network_signing"+os.sep
OUTDIR=DIRECTED_DIR+'features_and_labels'+os.sep+SPECIES+os.sep
# Initioalize dictionary of aliases
# alias_2geneid =  readname2geneid(DATA_DIR, SPECIES)
# with open(DATA_DIR+'alias_2geneid.pkl','wb') as f:
#     pickle.dump(alias_2geneid , f)
with open( DATA_DIR+'alias_2geneid.pkl', 'rb') as f:
    alias_2geneid = pickle.load(f)

##############################################################################
# ## Get training sets:
##############################################################################

if SPECIES == 'S_cerevisiae':
    datasets = {'patkar_kegg':DATA_DIR +  "patkar_yeast_kegg_signs.txt",\
                'kpi':DATA_DIR + 'yeast_kpi.txt',\
                'kegg':DATA_DIR + "S_cerevisiae_kegg_signed_ppi.txt",\
                'ubiq':DATA_DIR +'UbiNet2E3_substrate_interactions.tsv'}
       #'p_complex':DATA_DIR + SPECIES + "cyc2008.txt"
else:
    datasets = {'kegg':DATA_DIR + "H_sapiens_kegg_signed_ppi.txt",\
                 'PSP':DATA_DIR +'Kinase_Substrate_Dataset.txt',\
                 'depod':DATA_DIR +'depod_phospho_unchecked.csv',\
                'ubinet2':DATA_DIR+'UbiNet2E3_substrate_interactions.tsv'}

signed_datasets, signed_datasets_edge_weights = preprocess_signed_datasets(DATA_DIR, datasets, SPECIES, alias_2geneid)
#todo load-training da robe diverse. da preprocess-signed. da riguardare bah
# write(OUTDIR, signed_datasets, signed_datasets_edge_weights)
 # alternatively load:
# from preproc_utils import load_training_data

# signed_datasets, signed_datasets_edge_weights = load_training_data(OUTDIR, datasets, SPECIES)
#%%
print("\n>>>>>>> loading base network to propagte on:")
graph = graph_from_dataframe(DATA_DIR, SPECIES)
#ADD edges from training labels which are not present:
if SPECIES=='H_sapiens': 
    flag_of = {'kegg':'d', 'patkar_kegg':'d','PSP':'d','depod':'d','ubinet2':'d'}
else:
    flag_of = {'kegg':'d','kpi':'d', 'ubiq':'d', 'p_complex':'u', 'patkar_kegg':'d'}

for dataname, signed_data in signed_datasets_edge_weights.items():
    graph = add_edges_from_labels(graph, signed_data, flag = flag_of[dataname])


genes = sorted(graph.nodes)
#%%# Generate perturbation maps for subsequent feature generation:

if SPECIES == "S_cerevisiae":
    filename='mutational_signatures_Holstege.cdt'
else:
    filename =  'clustered_mean_gene_expression_figs2-4.csv.gz'
#%% for human: using clustered_mean_gene_experssion data  from figures 2 and 4
import pandas as pd
perturbations_map=pd.read_csv(DATA_DIR+filename, index_col=0, \
                  skiprows=[1,2],  header=0) #from fig 2a
#columns= gene transcript
#rows = gene name
perturbations_map.drop(columns=perturbations_map.columns[0], inplace=True)
#ma strano: figura dice: rows 1973 perturbs with strong fenotipes
#cols 2319 highli variable genes
# e i numeri piu o meno ci tornano, ma invertiti
# nel data le rows sono 2321 (e sembrano le perturbs, come nella foto)
# mentre le cols sono 1973 (e sono definiti rtanscripts, come nella fot)
# pero nella foto si vede chiaramente che il lato longo sono effettivamente le cols
#quindi qlcs non va. magari semplicemente hanno girato la figura di 90'
# this is the data u need, and fuck the rest.
perturbations_map.rename(columns= lambda x : x.split('_')[3], inplace=True)
from preproc_utils import translate_axes
from collections import defaultdict
seen_id=defaultdict(list)
seengene=[]
dupedid=defaultdict(list)
dupedgene=[]
for i in perturbations_map.columns:
    if i in seengene:
        dupedgene.append(i)
        try:
            dupedid[alias_2geneid[i]].append( i)
        except:
            continue
    else:
        seengene.append(i)
        #fin qui non ci sn duplicati
    try:
        if alias_2geneid[i] in seen_id.keys():
            dupedid[alias_2geneid[i]].append(i)
        seen_id[alias_2geneid[i]].append(i)
    except:
        continue
print([(i,seen_id[i]) for i in dupedid.keys()]) # ci sono alcuni ID che sono con piu nomi
#nell index, e nel column ce ne sono 3 ripetuti lol

translate_axes(perturbations_map, alias_2geneid) 
perturbations_map=perturbations_map.loc[:,~perturbations_map.columns.duplicated()] # drop cols
perturbations_map = perturbations_map[~perturbations_map.index.duplicated(keep=False)] #drop rows

#%% reimand kemmeren perturb map;
PERT_MAP_NAME =  'reimand'#'Holstege'
filename = 'patkar_yeast_reimand.txt'
perturb_dir=DIRECTED_DIR+'perturbation maps'+os.sep
from preproc_utils import translate_axes
import networkx as nx
pert_graph = graph_from_dataframe(perturb_dir, SPECIES, net_type="dir",filename=filename)
perturbations_map=nx.to_pandas_adjacency(pert_graph).T
perturbations_map=translate_axes(perturbations_map, alias_2geneid)
#%% load perturbation map 
if SPECIES == 'S_cerevisiae' :
    threshold = 1.7
if SPECIES=='H_sapiens':
    threshold=1#1.009
plus_targets_of_deletion, minus_targets_of_deletion = extract_knockotut_effect_pairs_from_data(perturbations_map, genes, threshold=threshold)
print(len(plus_targets_of_deletion.keys()))
#%% filter: take only those with at least 3= and 3- genes:
k=1000
pl_t={}
mn_t={}
for p in plus_targets_of_deletion.keys():
    if (len(plus_targets_of_deletion[p])>=k and len(minus_targets_of_deletion[p])>=k):
            pl_t[p]=plus_targets_of_deletion[p].copy()
            mn_t[p]=minus_targets_of_deletion[p].copy()
            
print(len(pl_t.keys()),len(mn_t.keys()))
#%% are the human t>1 and human k> perturbations similar?
shared=[]
for k in pl_t.keys():
    if k in plus_targets_of_deletion.keys():
        shared.append(k)
        
print(len(shared))
f=open(OUTDIR+'shared_filtered_perturbations.txt','w')
f.write(' '.join([str(i) for i in shared]))
f.close()

#%%
PERT_MAP_NAME='ADPBH'+'_1'
with open(OUTDIR+'plus_targets_'+PERT_MAP_NAME+'.pkl', 'wb') as f:
    pickle.dump(plus_targets_of_deletion, f)
with open(OUTDIR+'minus_targets_'+PERT_MAP_NAME+'.pkl','wb') as f:
    pickle.dump(minus_targets_of_deletion, f)
#%%
print("\n--------------- DATA PREPROCESSED SUCCESFULLY---------------")
print('- Positive knockout experiment targets:', len(plus_targets_of_deletion),'\n- Negative knockout experiment targets:',len(minus_targets_of_deletion))
print('- Labels (0 for potive and 1 for negative signe of directed interaction):\n', {name: data.value_counts() for (name, data) in signed_datasets.items()})
print('- Network size (for propagation): ','edges:', len(graph.edges), 'nodes:', len(graph.nodes)) # da mette a posto!! #TODO

