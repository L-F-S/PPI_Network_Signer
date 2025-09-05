# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 12:05:53 2023

@author: L-F-S
"""
from glob_vars import SPECIES, LBL_DIR, TRAIN_DATA, SPECIES
from preproc_utils import load_training_data
import argparse
print(SPECIES)
signed_datasets_labels, signed_datasets_edge_weights = load_training_data(LBL_DIR, TRAIN_DATA, SPECIES)


print('- Labels (0 for positive and 1 for negative signe of directed interaction):')
for name, label in signed_datasets_labels.items():
    print('\t',name,':')
    for i in label.value_counts().index:
        print('\t\t'+str(i)+'\t'+str(label.value_counts()[i]))
