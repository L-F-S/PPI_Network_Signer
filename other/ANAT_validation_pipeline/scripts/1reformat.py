# -*- coding: utf-8 -*-
"""
Reformats ANAT network output

python reformat.py ANAT_output_name

creates a .txt edge file
"""

import pickle
import collections
import itertools
import lxml.etree as etree
import os
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from random import seed, sample
from datetime import date
import sys



SPECIES = "S_cerevisiae"
# SPECIES='H_sapiens'
DATE=str(date.today().strftime("%d/%m/%Y"))
HOME_DIR='G:'+os.sep+'Il Mio Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep
HOME_DIR='G:'+os.sep+'My Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep
MAIN_DATA_DIR = HOME_DIR+'Data'+os.sep+SPECIES+os.sep
os.chdir(HOME_DIR+'network_signing'+os.sep+'SIGNAL Validations target sign reconstruction'\
         +os.sep+'ANAT_validation_pipeline'+os.sep)

with open( MAIN_DATA_DIR+'alias_2geneid.pkl', 'rb') as f:
    alias_2geneid = pickle.load(f)


if len(sys.argv)==1:
    FILENAME='TLM_all_KOterms2'#'TUBB_v_all'
else:
    FILENAME=sys.argv[1]
os.chdir('1 ANAT Build networks')

columns = ['ID1', 'ID2']

# Read the CSV file and select the desired columns and rows
df = pd.read_csv('ANAT output'+os.sep+FILENAME+'.csv', usecols=columns)

# Convert the selected columns to a list of tuples
edges = list(df.values)

print(edges)

# Write the list of tuples to the file
with open('ANAT output'+os.sep+FILENAME+'.txt', 'w') as f:
    for edge in edges:
        f.write(' '.join([str(alias_2geneid[protein]) for protein in edge]) + '\n')

print("File saved successfully.")
