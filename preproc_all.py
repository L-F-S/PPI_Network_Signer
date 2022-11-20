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
    - major refactoring
    
"""

import os
import pickle
from preproc_utils import  add_edges_from_labels,graph_from_dataframe,\
    preprocess_signed_datasets, readname2geneid, get_perturbations_map,\
         extract_knockotut_effect_pairs_from_data, write

##############################################################################
#  INPUTS
##############################################################################
SPECIES = "H_sapiens"#
SPECIES =  "S_cerevisiae" #

HOME_DIR  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# HOME_DIR  =  "G:" +os.sep+"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
#HOME_DIR  =  "/content/drive" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 

DATA_DIR = HOME_DIR+'Data'+os.sep
SPECIES_DATA_DIR = DATA_DIR+SPECIES+os.sep
DIRECTED_DIR = HOME_DIR+"network_signing"+os.sep
OUTDIR=DIRECTED_DIR+'features_and_labels'+os.sep+SPECIES+os.sep

# Propagation parameters:
PROPAGATE_ALPHA = 0.6
PROPAGATE_EPSILON = 1e-5
PROPAGATE_ITERATIONS = 100

# Initioalize dictionary of aliases
alias_2geneid =  readname2geneid(SPECIES_DATA_DIR, SPECIES)

##############################################################################
# ## Get training sets:
##############################################################################


if SPECIES == 'S_cerevisiae':
    datasets = ['kegg_kpi', 'ubiq', 'p_complex']
else:
    datasets = {'kegg':SPECIES_DATA_DIR + SPECIES + "_kegg_signed_ppi.txt",\
                 'PSP':DATA_DIR +'Kinase_Substrate_Dataset.txt',\
                'ubinet2':DATA_DIR+'UbiNet2E3_substrate_interactions.tsv'}

signed_datasets, signed_datasets_edge_weights = preprocess_signed_datasets(DATA_DIR, datasets, SPECIES, alias_2geneid)
write(OUTDIR, signed_datasets, signed_datasets_edge_weights)

#%%
print("\n>>>>>>> loading base network to propagte on:")

graph = graph_from_dataframe(SPECIES_DATA_DIR, SPECIES)
#ADD edges from training labels which are not present:
if SPECIES=='H_sapiens': 
    flag_of = {'kegg':'d','PSP':'d','depod':'d','ubinet2':'d'}
else:
    flag_of = {'kegg_kpi':'d', 'ubiq':'d', 'p_complex':'u'}

for dataname, signed_data in signed_datasets_edge_weights.items():
    graph = add_edges_from_labels(graph, signed_data, flag = flag_of[dataname])

    
genes = sorted(graph.nodes)
# Generate perturbation maps for subsequent feature generation:

perturbations_map = get_perturbations_map(SPECIES_DATA_DIR, alias_2geneid, SPECIES)
print('shape:',perturbations_map.shape)
plus_targets_of_deletion, minus_targets_of_deletion = extract_knockotut_effect_pairs_from_data(perturbations_map, genes)
with open(OUTDIR+'plus_targets.pkl', 'wb') as f:
    pickle.dump(plus_targets_of_deletion, f)
with open(OUTDIR+'minus_targets.pkl','wb') as f:
    pickle.dump(minus_targets_of_deletion, f)

print("\n--------------- DATA PREPROCESSED SUCCESFULLY---------------")
print('- Positive knockout experiment targets:', len(plus_targets_of_deletion),'\n- Negative knockout experiment targets:',len(minus_targets_of_deletion))
print('- Labels (0 for potive and 1 for negative signe of directed interaction):\n', {name: data.value_counts() for (name, data) in signed_datasets.items()})
print('- Network size (for propagation): ','edges:', len(graph.edges), 'nodes:', len(graph.nodes)) # da mette a posto!! #TODO

