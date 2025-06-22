# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 11:27:58 2023

@author: L-F-S

Set the vriables in this file before a pipeline run
"""
import os
import pandas as pd

######################
# SPECIES
######################
# SPECIES =  'H_sapiens' # 
SPECIES =  "S_cerevisiae" #


#######################
# Filenames and Paths
#######################

# Perturbation map name
# PERT_MAP = 'Reimand'
PERT_MAP = 'Kemmeren' if SPECIES == 'S_cerevisiae' else 'CMGE'

# Training dataset names
TRAIN_DATA = [ 'patkar_kegg', 'kpi', 'ubinet2'] if SPECIES == 'S_cerevisiae' else ['PSP','depod','kegg','ubinet2']#'patkar_kegg']#patkar_kegg,
# TRAIN_DATA = ['PSP','depod','ubinet2']    

def get_train_data_names(species):
    return ['patkar_kegg','kpi', 'ubinet2'] if species == 'S_cerevisiae' else ['kegg','PSP','ubinet2', 'depod']

# PATHS
HOME_DIR = os.path.dirname(os.path.abspath(__file__))+os.sep
# HOME_DIR  =  '/home/bnet/lorenzos/signed/signedv3/'
# DRIVE_PATH  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# DRIVE_PATH=  'G:'+os.sep+'Il Mio Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep
# HOME_DIR  =  DRIVE_PATH+'network_signing'+os.sep
# sys.path.append(HOME_DIR)

# input dirs:
LBL_DIR = HOME_DIR+'input'+os.sep+SPECIES+os.sep +'labels'+os.sep 
def get_lbl_dir(species):
    return HOME_DIR+'input'+os.sep+species+os.sep +'labels'+os.sep 

EDGES_DIR = HOME_DIR+'input'+os.sep+SPECIES+os.sep +'edges'+os.sep 
NET_DIR = HOME_DIR+'input'+os.sep+SPECIES+os.sep +'network'+os.sep
PRT_DIR = HOME_DIR+'input'+os.sep+SPECIES+os.sep +'perturbation_signatures'+os.sep
DICT_DIR = HOME_DIR+'input'+os.sep+SPECIES+os.sep +'dictionaries'+os.sep

# output dirs
FT_DIR=HOME_DIR+'output'+os.sep+SPECIES+os.sep+'features'+os.sep
SIGNAL_DIR=HOME_DIR+'output'+os.sep+SPECIES+os.sep+'predictions'+os.sep 
MOD_DIR=HOME_DIR+'output'+os.sep+SPECIES+os.sep+'models'+os.sep
IMG_DIR=HOME_DIR+os.sep+'imgs'+os.sep+'v4'+os.sep+SPECIES+os.sep

# Bio insights:
BIO_DIR=HOME_DIR+os.sep+'functional_enrichment'+os.sep
ENRICHMENT_OUT=BIO_DIR+'output'+os.sep+'enrichments'+os.sep


def get_img_dir(species):
    return HOME_DIR+os.sep+'imgs'+os.sep+'v4'+os.sep+species+os.sep

#File names:
NET_FILE=SPECIES+'_'+ '_'.join(TRAIN_DATA)+'.nx' # Base network filename
DICT_FILE = 'alias_2geneid.pkl'              # gene map filename. Gene name : id dictionary filename

# Paths initializer
# To be used if SPECIES is declared inside script in use

def init_all(SPECIES):
    home_dir = os.path.dirname(os.path.abspath(__file__))+os.sep
    train_data = ['patkar_kegg','kpi', 'ubiq'] if SPECIES == 'S_cerevisiae'\
        else ['kegg','PSP','depod','ubinet2']
    lbl_dir = home_dir+'input'+os.sep+SPECIES+os.sep +'labels'+os.sep 
    edges_dir = home_dir+'input'+os.sep+SPECIES+os.sep +'edges'+os.sep 
    net_dir = home_dir+'input'+os.sep+SPECIES+os.sep +'network'+os.sep 
    prt_dir = home_dir+'input'+os.sep+SPECIES+os.sep +'perturbation_signatures'+os.sep
    dict_dir = home_dir+'input'+os.sep+SPECIES+os.sep +'dictionaries'+os.sep

    # output dirs
    ft_dir=home_dir+'output'+os.sep+SPECIES+os.sep+os.sep+'features'+os.sep
    signal_dir=home_dir+'output'+os.sep+SPECIES+os.sep+os.sep+'predictions'+os.sep 
    mod_dir=home_dir+'output'+os.sep+SPECIES+os.sep+os.sep+'models'+os.sep
    img_dir=home_dir+os.sep+'imgs'+os.sep+'v4'+os.sep+SPECIES+os.sep
    net_file=SPECIES+'_'+ '_'.join(train_data)+'.nx' # Base network filename
    return home_dir, train_data, lbl_dir, edges_dir, net_dir, prt_dir, dict_dir, ft_dir,\
        signal_dir, mod_dir, img_dir, net_file
    
    

###############################################################################
# Propagation parameters:
###############################################################################

PROPAGATE_ALPHA = 0.8
PROPAGATE_EPSILON = 1e-5
PROPAGATE_ITERATIONS = 100

# Multi-threading parameters:
    # for SIGNAL_ft_gen_PARALLEL.py
N_JOBS = 4


###############################################################################
# CROSS VALIDATION VARIABLES
###############################################################################

# CV dirs
CV_HOME = HOME_DIR+'Validations'+os.sep+'Crossvalidations'+os.sep
CV_OUT = CV_HOME+os.sep+'out'+os.sep
CV_LOG = 'crossvalidations_out_log.txt'


###############################################################################
# SIGNAL PHENOTYPE PREDICTION MODEL PARAMETERS
###############################################################################

nSPS = 100
tau = 0.5
epsilon = 0.0

###############################################################################
#KO REC VALIDATION VARIABLES
###############################################################################
KO_VAL_HOME  = HOME_DIR+'Validations'+os.sep+'phenotype_reconstruction_model'+os.sep
KO_VAL_IN = KO_VAL_HOME + 'input'+os.sep
KO_VAL_OUT = KO_VAL_HOME + 'output'+os.sep


# ANCHORS AND TERMINALS

def get_anchors():
    f=open(KO_VAL_IN+'telomere_machinery_genes.txt')
    lines=[x.strip() for x in f.readlines()]
    anchors=lines
    f.close()
    return anchors

def get_terms(termfile=None):
    if not termfile:
        termfile='TLMphenotypesnodupes'
    TLMphenotypesdf=pd.read_csv(KO_VAL_IN+termfile+'.txt', header=0, sep='\t')
    terms=list(TLMphenotypesdf[TLMphenotypesdf.columns[0]].values)
    return terms

################################################################################
# ANAT paths
################################################################################
ANAT_VAL_HOME = HOME_DIR+'other'+os.sep+'ANAT_validation_pipeline'+os.sep
ANAT_VAL_INPUT=ANAT_VAL_HOME+'1 ANAT Build networks'+os.sep+'input'+os.sep
ANAT_VAL_SIGNAL_OUT = ANAT_VAL_HOME+os.sep+'2 SIGNAL score networks'+os.sep+'SIGNAL_output'+os.sep
TLM_VAL_OUT = ANAT_VAL_HOME+os.sep+'TLM_chain_rule_output'+os.sep
ANAT_EXPERIMENTAL_INPUT_DIR=ANAT_VAL_HOME+os.sep+'3 score networks and compare to experiments'+os.sep+'experimental_inputs'+os.sep


###############################################################################
# Functional enrichment paths
###############################################################################
BIO_HOME=HOME_DIR+os.sep+'functional_enrichment'+os.sep
BIO_DATA=HOME_DIR+os.sep+'functional_enrichment'+os.sep+'data'+os.sep
BIO_OUT=HOME_DIR+os.sep+'functional_enrichment'+os.sep+'output'+os.sep
