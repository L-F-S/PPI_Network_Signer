# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 12:05:53 2023

@author: los4
"""
from glob_vars import SPECIES, LBL_DIR, TRAIN_DATA
from preproc_utils import load_training_data
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-s', dest='SPECIES', type=str, nargs='?', default='S_cerevisiae',
                    help='species: [\'H_sapiens\', \'S_cerevisiae\']\ndefault: S_cerevisiae')


args = parser.parse_args()

SPECIES= SPECIES if not args.SPECIES else args.SPECIES
print(SPECIES)
signed_datasets_labels, signed_datasets_edge_weights = load_training_data(LBL_DIR, TRAIN_DATA, SPECIES)


print('- Labels (0 for positive and 1 for negative signe of directed interaction):')
for name, label in signed_datasets_labels.items():
    print('\t',name,':')
    for i in label.value_counts().index:
        print('\t\t'+str(i)+'\t'+str(label.value_counts()[i]))