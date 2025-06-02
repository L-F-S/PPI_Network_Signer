# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 12:05:53 2023

@author: L-F-S

Creates horizontal stacked barplots of signed data (paper fig. 2)
"""
from glob_vars import get_lbl_dir, get_train_data_names, get_img_dir
from preproc_utils import load_training_data
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = 'times new roman'

a=0.9
fig, ax = plt.subplots(1,2, figsize=(15*a,5*a))
for SPECIES in ['S_cerevisiae', 'H_sapiens']:
    signed_datasets_labels, signed_datasets_edge_weights = load_training_data(get_lbl_dir(SPECIES), get_train_data_names(SPECIES), SPECIES)
    
    # 1 get the signed daata
    print('- Labels (0 for positive and 1 for negative signe of directed interaction):')
    pos_stack = {}
    neg_stack = {}
    for name, label in signed_datasets_labels.items():
        print('\t',name,':')
        if 'patkar' in name:
            name=name.split('_')[1]
        for i in label.value_counts().index:
            print('\t\t'+str(i)+'\t'+str(label.value_counts()[i]))
            try:
                label.value_counts()[0]
                pos_stack[name] = label.value_counts()[0]
            except:
                pos_stack[name] = 0
            try:
                label.value_counts()[1]
                neg_stack[name] = label.value_counts()[1]
            except:
                neg_stack[name] = 0
    # Plot horizontal bar chart
    
    # signed_datasets_labels['kpi'].value_counts().plot(kind='bar', stacked=True)
    
    datasets = list(signed_datasets_labels.keys())
    
    signs = ['+','-']
      
    
    
    
    # ax = fig.add_subplot(111)
    cumsum=np.array([0,0])
    
    i = 0 if SPECIES=='S_cerevisiae' else 1
    print(i)
    for name, num in neg_stack.items():
        print(name, num)
        ax[i].barh(['-','+'], [num, pos_stack[name]], label=name, left=cumsum)
        cumsum= cumsum+[num, pos_stack[name]]
    ax[i].set_title(SPECIES)
    ax[i].tick_params(axis='y', labelsize=20)
    ax[i].legend()
plt.savefig(get_img_dir(SPECIES)+'fig2_signed_data.pdf')
