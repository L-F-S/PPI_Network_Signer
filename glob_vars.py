# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 11:27:58 2023

@author: L-F-S
"""
import os
SPECIES =  "S_cerevisiae" # H_sapiens

# Perturbation map name
PERT_MAP = 'Holstege'     #'CMGE'

# Training dataset names
TRAIN_DATA = ['patkar_kegg','kpi', 'ubiq'] if SPECIES == 'S_cerevisiae' else ['kegg','PSP','depod','ubinet2']

# PATHS
HOME_DIR = os.path.dirname(os.path.abspath(__file__))+os.sep
# HOME_DIR  =  '/home/bnet/lorenzos/signed/signedv3/'
# DRIVE_PATH  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# DRIVE_PATH=  'G:'+os.sep+'Il Mio Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep
# HOME_DIR  =  DRIVE_PATH+'network_signing'+os.sep
# sys.path.append(HOME_DIR)

# input dirs:
LBL_DIR = HOME_DIR+'input'+os.sep+SPECIES+os.sep +'labels'+os.sep 
EDGES_DIR = HOME_DIR+'input'+os.sep+SPECIES+os.sep +'edges'+os.sep 
NET_DIR = HOME_DIR+'input'+os.sep+SPECIES+os.sep +'network'+os.sep 
PRT_DIR = HOME_DIR+'input'+os.sep+SPECIES+os.sep +'perturbation_signatures'+os.sep
DICT_DIR = HOME_DIR+'input'+os.sep+SPECIES+os.sep +'dictionaries'+os.sep

# output dirs
FT_DIR=HOME_DIR+'output'+os.sep+SPECIES+os.sep+os.sep+'features'+os.sep
SIGNAL_DIR=HOME_DIR+'output'+os.sep+SPECIES+os.sep+os.sep+'predictions'+os.sep 
MOD_DIR=HOME_DIR+'output'+os.sep+SPECIES+os.sep+os.sep+'models'+os.sep
IMG_DIR=HOME_DIR+os.sep+'imgs'+os.sep+'v4'+os.sep+SPECIES+os.sep

#File names:
NET_FILE=SPECIES+'_'+ '_'.join(TRAIN_DATA)+'.nx' # Base network filename
GENE_MAP_NAME = 'alias_2geneid.pkl'              # Gene name : id dictionary filename

# Propagation parameters:
    # These are parameters for the propagation algorithms in score_edges.py
PROPAGATE_ALPHA = 0.6
PROPAGATE_EPSILON = 1e-5
PROPAGATE_ITERATIONS = 100

# Multi-threading parameters:
    # for SIGNAL_ft_gen_PARALLEL.py
N_JOBS = 8
