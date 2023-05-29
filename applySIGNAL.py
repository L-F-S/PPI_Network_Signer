# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

utility script for KT pairs reconstuction validation 

Predicts SIGNER scores for all base network.
output: SIGNAL score for every  PPI in the network.
input: Species, training data, deletion signature data name, edges features table (created before with preprocess _all)
"""


import os

import pandas  as pd
from datetime import date
from preproc_utils import load_training_data ,graph_from_dataframe, add_edges_from_labels
from train_and_vis3_5 import  load_features
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import argparse



##############################################################################
#  INPUTS
##############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('datasets', type=str, nargs='*',
                    help='dataset names:  options (and default) for S_cerevisiae: [\'kegg\',\'kpi\', \'ubiq\'],\
                        options (anmd default) for H_sapiens: [\'kegg\',\'PSP\',\'depod\',\'ubinet2\']')
parser.add_argument('-s', dest='SPECIES', type=str, nargs='?', default='S_cerevisiae',
                    help='species: [\'H_sapiens\', \'S_cerevisiae\']\ndefault: S_cerevisiae')
parser.add_argument('-o', dest='OUTDIR', type=str, nargs='?', default=None,
                    help='Optional, output directory')
parser.add_argument('-p', dest='perturbation_filename', type=str, nargs='?', default='Holstege',
                    help='default: \'Holstege\' , other options: [\'reimand\', ADPBH, CMGE..]')
parser.add_argument('-i', dest='feature_table_path', type=str, nargs='?', default=None,
                    help='Input directory')
parser.add_argument('-f', dest='feature_table_name', type=str, nargs='?', default=None,
                    help='Features table name')

args = parser.parse_args()

DATE=str(date.today())
SPECIES=args.SPECIES
if len(args.datasets) == 0:
    datasets = ['kegg','kpi', 'ubiq'] if SPECIES == 'S_cerevisiae' else ['kegg','PSP','depod','ubinet2']
else:
    datasets = args.datasets
dataset_name = '_'.join(datasets)
pert_map=args.perturbation_filename  

# # Servier directories:
# HOME_DIR  =  '/home/bnet/lorenzos/signed/signedv3/'
# INPUT_DIR=HOME_DIR+"input"+os.sep+SPECIES+os.sep
# OUTDIR=HOME_DIR+'output'+os.sep+SPECIES+os.sep+'validation_out'
# FEATURESDIR=OUTDIR
# BASENETDIR=INPUT_DIR

# Local directories
HOME_DIR  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
HOME_DIR  =  "G:" +os.sep+"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
SIGNAL_DIR = HOME_DIR+"network_signing"+os.sep
os.chdir(SIGNAL_DIR)
TRAINING_DIR=SIGNAL_DIR+'features_and_labels'+os.sep+SPECIES+os.sep
if not args.OUTDIR:
    OUTDIR=SIGNAL_DIR+'output'+os.sep+SPECIES+os.sep
else:
    OUTDIR=args.OUTDIR

MODEL_DIR=SIGNAL_DIR+'models_and_predictions'+os.sep+SPECIES+os.sep

feature_table_path=args.feature_table_path  # HOME_DIR+'collab norway'+os.sep+'SIGNAL pacitaxel target sign reconstruction'\+os.sep+ '2 SIGNAL score networks'+os.sep+'SIGNAL_features'+os.sep
feature_table_name=args.feature_table_name  #'4088.edges_'+pert_map+'.ft.pkl'
with open( feature_table_path+feature_table_name, 'rb') as f:
    features_table = pickle.load(f)

   # TODO: da metterlo solo se si applya a tutto 
# CHUNKNAMES=['netedgesI','netedgesII','netedgesIII','netedgesIV','netedgesV','netedgesVI']
# #print('Load edges (indexes) and features (columns) for base net')
# netind={}
# for chunk in CHUNKNAMES:
#     with open(INPUT_DIR+chunk,'rb') as f:
#         netind[chunk]=pickle.load(f)
        #%%

def predictSIGNAL(test_features_table, train=False):
    
    if train:
        print('Training classifier')
        training_labels, training_labels_weights = load_training_data(TRAINING_DIR, datasets, SPECIES)
        training_features_table = load_features(TRAINING_DIR, [i+'_'+pert_map for i in datasets], SPECIES)
        training_labels=pd.concat(list(training_labels.values())) # works because python dictionaries are ordered in python3
        # training_labels=training_labels.iloc[1:] #temp solo x ADPBH_1 e 1000 (vedi log 30/01/2023)
        training_features_table= pd.concat(list(training_features_table.values()))
        classifier=RandomForestClassifier() 
        classifier.fit(StandardScaler().fit_transform(training_features_table), training_labels) 
    
    else:
        with open(MODEL_DIR+dataset_name+'_'+pert_map+'.rf','rb') as f:
            classifier=pickle.load(f)
    print('caclulating SIGNAL scores')
    predictions = classifier.predict_proba(StandardScaler().fit_transform(test_features_table))
    predictions = pd.DataFrame(data=predictions, index=test_features_table.index, columns=['+','-'])
    return predictions['-']
#%% predict SIGNAL score for given features table

print('apply SIGNAL')
SIGNALscores=predictSIGNAL(features_table, train=False)
#%%
print('Exporting SIGNAL data')
SIGNALscores.to_csv(OUTDIR+os.sep+feature_table_name+'.sgnl', sep=' ', header=False)


#%% PREDICT ALL EDGES for different KT pairs
# N_JOBS=64
# from joblib import Parallel, delayed

# packed_results = Parallel(n_jobs=N_JOBS)(delayed(function)\
#                             (source,)\
#                                 for source in list(targets_of_deletion.keys()))