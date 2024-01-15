# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:45:41 2024

@author:L-F-S
"""
import os
import pickle
from preproc_utils import readname2geneid
from glob_vars import SPECIES, DICT_DIR

RAW_DATA_DIR =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep +'Data'+os.sep+SPECIES+os.sep

alias_2geneid =  readname2geneid(RAW_DATA_DIR, SPECIES)
# write dictionary
with open(DICT_DIR+'alias_2geneid.pkl','wb') as f:
    pickle.dump(alias_2geneid, f)
