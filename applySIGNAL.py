# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

utility script for KT pairs reconstuction validation 

Predicts SIGNER scores for all base network.
output: SIGNAL score for every  PPI in the network.
input: Species, training data, deletion signature data name,
edges features table (created before with preprocess _all)
"""


import os
import pandas  as pd
from preproc_utils import load_training_data
from train_and_vis import  load_features
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import argparse
from glob_vars import SPECIES, TRAIN_DATA, PERT_MAP, HOME_DIR, LBL_DIR,\
    EDGES_DIR, FT_DIR, SIGNAL_DIR, MOD_DIR


def predictSIGNAL(test_features_table, training_model_name, MOD_DIR, train=False):
    'test_features_table: pd.DataFrame'
    print(len(test_features_table.index),test_features_table.index[0],test_features_table.index[-1])
    
    if train:
        print('Training classifier')
        training_labels, training_labels_weights = load_training_data(LBL_DIR, TRAIN_DATA, SPECIES)
        training_features_table = load_features(FT_DIR, [i+'_'+PERT_MAP for i in TRAIN_DATA], SPECIES)
        training_labels=pd.concat(list(training_labels.values())) # works because python dictionaries are ordered in python3
        # training_labels=training_labels.iloc[1:] #temp solo x ADPBH_1 e 1000 (vedi log 30/01/2023)
        training_features_table= pd.concat(list(training_features_table.values()))
        classifier=RandomForestClassifier() 
        classifier.fit(StandardScaler().fit_transform(training_features_table), training_labels) 
    
    else:
        with open(MOD_DIR+training_model_name,'rb') as f:
            classifier=pickle.load(f)
    print('caclulating SIGNAL scores')
    predictions = classifier.predict_proba(StandardScaler().fit_transform(test_features_table))
    predictions = pd.DataFrame(data=predictions, index=test_features_table.index, columns=['+','-'])
    return predictions['-']
#%%
def main(SPECIES, TRAIN_DATA, PERT_MAP, HOME_DIR, LBL_DIR,\
    EDGES_DIR, FT_DIR, SIGNAL_DIR, MOD_DIR, w=False):
    ##############################################################################
    #  INPUTS
    ##############################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('TRAIN_DATA', type=str, nargs='*',
                        help='training dataset names:  options (and default) for S_cerevisiae: [\'kegg\',\'kpi\', \'ubiq\'],\
                            options (anmd default) for H_sapiens: [\'kegg\',\'PSP\',\'depod\',\'ubinet2\']')
    parser.add_argument('-s', dest='SPECIES', type=str, nargs='?', default=None,
                        help='species: [\'H_sapiens\', \'S_cerevisiae\']\ndefault: S_cerevisiae')
    parser.add_argument('-p', dest='PERT_MAP', type=str, nargs='?', default=None,
                        help='default: \'Kemmeren\' , other options: [\'reimand\', ADPBH, CMGE..]')
    parser.add_argument('-f', dest='feature_table_name', type=str, nargs='?', default=None,
                        help='Features table name')
    parser.add_argument('-ld', dest='LBL_DIR', type=str, nargs='?', default=None,
                        help='Optional, directory of training labels')
    parser.add_argument('-ed', dest='EDGES_DIR', type=str, nargs='?', default=None,
                        help='Optional, directory of .edges file (with which edges to make features for)')
    parser.add_argument('-nd', dest='NET_DIR', type=str, nargs='?', default=None,
                        help='Optional, directory of network file (file must be a pickled networkx.DiGraph object)')
    parser.add_argument('-kd', dest='KO_DIR', type=str, nargs='?', default=None,
                        help='Optional, directory of deletion signatures files (files must be a pickled dictionary)')
    parser.add_argument('-fd', dest='FT_DIR', type=str, nargs='?', default=None,
                        help='Optional, directory of SIGNAL features (input for applySIGNAL)')
    parser.add_argument('-md', dest='MOD_DIR', type=str, nargs='?', default=None,
                        help='Optional, directory of trained models')
    parser.add_argument('-sd', dest='SIGNAL_DIR', type=str, nargs='?', default=None,
                        help='Optional, directory of output SIGNAL scores')
    
    args = parser.parse_args()
    
    TRAIN_DATA = TRAIN_DATA if not args.TRAIN_DATA else args.TRAIN_DATA
    
    SPECIES= SPECIES if not args.SPECIES else args.SPECIES
    PERT_MAP=PERT_MAP if not args.PERT_MAP else args.PERT_MAP
    
    # input dirs:
    HOME_DIR = HOME_DIR
    os.chdir(HOME_DIR)
    LBL_DIR = LBL_DIR if not args.LBL_DIR else args.LBL_DIR
    EDGES_DIR = EDGES_DIR if not args.EDGES_DIR else args.EDGES_DIR
    
    # output dirs
    FT_DIR= FT_DIR if not args.FT_DIR else args.FT_DIR
    SIGNAL_DIR=SIGNAL_DIR if not args.SIGNAL_DIR else args.SIGNAL_DIR
    MOD_DIR=MOD_DIR if not args.MOD_DIR else args.MOD_DIR
    
    if not args.feature_table_name:
        to_concat=[]
        for data in TRAIN_DATA:
            print(data)
            feature_table_name=data+'_'+PERT_MAP+'.ft' 
            with open( FT_DIR+feature_table_name, 'rb') as f:
                tmp_features_table = pickle.load(f)
            to_concat.append(tmp_features_table)
        features_table = pd.concat(to_concat)
    else:
        feature_table_name=args.feature_table_name
        with open( FT_DIR+feature_table_name, 'rb') as f:
            features_table = pickle.load(f)
    
    
    # CHUNKNAMES=['netedgesI','netedgesII','netedgesIII','netedgesIV','netedgesV','netedgesVI']
    # #print('Load edges (indexes) and features (columns) for base net')
    # netind={}
    # for chunk in CHUNKNAMES:
    #     with open(INPUT_DIR+chunk,'rb') as f:
    #         netind[chunk]=pickle.load(f)
    
    # predict SIGNAL score for given features table
    tmp = '_'.join(TRAIN_DATA)
    training_model_name=tmp+'_'+PERT_MAP+'.rf'
    SIGNALscores=predictSIGNAL(features_table, training_model_name, MOD_DIR, train=False)
    print(SIGNALscores)
    #
    print('Exporting SIGNAL data')
    SIGNALscore_filename=feature_table_name.rstrip('.ft')
    SIGNALscores.to_csv(SIGNAL_DIR+os.sep+SIGNALscore_filename+'.sgnl', sep=' ', header=False)
    return
    


#%%
if __name__ == '__main__':
    main(SPECIES, TRAIN_DATA, PERT_MAP, HOME_DIR, LBL_DIR,\
        EDGES_DIR, FT_DIR, SIGNAL_DIR, MOD_DIR)
    
#%% PREDICT ALL EDGES for different KT pairs
# N_JOBS=64
# from joblib import Parallel, delayed

# packed_results = Parallel(n_jobs=N_JOBS)(delayed(function)\
#                             (source,)\
#                                 for source in list(targets_of_deletion.keys()))
