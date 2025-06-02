# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
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

SPECIES = "S_cerevisiae"
# SPECIES='H_sapiens'
DATE=str(date.today().strftime("%d/%m/%Y"))
HOME_DIR='G:'+os.sep+'Il Mio Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep
HOME_DIR='G:'+os.sep+'My Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep
MAIN_DATA_DIR = HOME_DIR+'Data'+os.sep+SPECIES+os.sep
#%% fig 2 ma non 
# KPI KEGG+KPI KEGGG+UBINET KEGG+KPI+UBINET
plt.rcParams["font.family"] = "times new roman"

dati_y={'KPI':0.77,'KEGG+KPI':0.71, 'KEGG+UbiNet2':0.81, 'KPI+UbiNet2':0.85, 'KEGG+KPI+UbiNet2':0.79}#'KEGG Patkar 2018 best':0.81,'KPI Patkar 2018 best':0.85}
 
dati_h={'KPI':0.88,'KEGG+KPI':0.71,\
        'KEGG+UbiNet2':0.87,\
        'KPI+UbiNet2':0.91,\
        'KEGG+KPI+UbiNet2':0.81}

color_of={'KPI':'b','KEGG':'b','KEGG+KPI':'b','KEGG+UbiNet2':'b',\
      'KEGG+KPI+UbiNet2':'b','KPI+UbiNet2':'b','KEGG Patkar 2018 best':'r','KPI Patkar 2018 best':'r',\
         'KEGG Patkar 2018 Holstege':'b','KPI Patkar 2018 Holstege':'b',\
        'KEGG Patkar 2018 Reimand':'b','KPI Patkar 2018 Reimand':'b',\
         'KEGG Holstege':'r','KPI Holstege':'r',\
             'KEGG Reimand':'r','KPIReimand':'r','PSP+Depod':'b',\
        'KEGG+PSP+Depod':'b','KEGG+PSP+Depod+Ubinet2':'b','PSP+Depod+Ubinet2':'b'}
scale=2
fig, ax = plt.subplots(1,2, figsize=np.array([15,5])*scale)

for i, (name,dati_to_plot) in enumerate([ ('yeast',dati_y), ('human', dati_h)]):
    ax[i].bar(dati_to_plot.keys(),dati_to_plot.values(),color=[color_of[data] for data in dati_to_plot.keys() ])
    ax[i].tick_params(axis='x', labelsize=12*scale, rotation=20)
    ax[i].tick_params(axis='y', labelsize=15*scale)
    ax[i].set_ylim(0,1)
    
ax[0].set_title('(a) Yeast', size=20*scale)
ax[1].set_title('(b) Human', size=20*scale)
#%% fig 3
# dati_to_plot={'KEGG Patkar 2018 best':0.81,'KPI Patkar 2018 best':0.85}
# labels=['Patkar2018','SIGNAL']
# plt.bar(dati_to_plot.keys(),dati_to_plot.values(),color=[color_of[data] for data in dati_to_plot.keys() ], label=labels )
# plt.ylim((0,1))
# plt.legend()
# plt.ylabel('AUC')
# plt.grid()
#%%
# dati_signs_plus = { 'KPI':893,'KEGG':350}
# dati_signs_minus ={'KPI':179,'KEGG':145}
dati_signs_yeast ={'KEGG':[350,145],\
             'KPI':[893,179],\
             'Ubinet2':[0,211]}
dati_signs_human = {'KEGG':[109,208],\
                    'KPI':[2506,923],\
                    #'PSP':[2506,0],\
                    'UbiNet2':[0,1272],\
                #'Depod':[0,923]
    }
    
scale=2
fig, ax = plt.subplots(1,2, figsize=np.array([15,5])*scale)

dati_signs=dati_signs_yeast
ax[1].barh(['+','_'],dati_signs['KEGG'], left = 0,  label='KEGG')
for i, (name, data) in enumerate(dati_signs.items()):
    if i == 0:
        continue    
    ax[1].barh(['+','_'],data, left = dati_signs[list(dati_signs.keys())[i-1]], label = name)
ax[1].set_title('(b) Yeast', size=20*scale)
ax[0].barh(['+','_'],dati_signs_human['KEGG'], left = 0,  label='KEGG')
for i, (name, data) in enumerate(dati_signs_human.items()):
    if i == 0:
        continue    
    ax[0].barh(['+','_'],data, left = dati_signs_human[list(dati_signs_human.keys())[i-1]], label = name)
ax[0].set_title('(a) Human', size=20*scale)
#ax[0].set_xticks(np.arange(0,3000,500),fontsize=40*scale)
ax[0].tick_params(axis='x', labelsize=15*scale)
ax[0].tick_params(axis='y', labelsize=20*scale)
ax[1].tick_params(axis='both', labelsize=15*scale)
ax[1].tick_params(axis='y', labelsize=20*scale)
# ax[0].legend(fontsize=12*scale, loc=(1*scale,0.33333*scale))
handles, labels = ax[0].get_legend_handles_labels() 
fig.legend(handles, labels, loc=(0.85,0.65),fontsize=12*scale)
