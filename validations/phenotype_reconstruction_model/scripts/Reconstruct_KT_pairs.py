# -*- coding: utf-8 -*-
"""

@author: L-F-S

Predict sign of leftout knockout/target pairs
    To run with the output of PARALLELgenerate_SP_scores.py

Reconstruct TLM phenotype 
    To run with the output of generate_SP_scores_TLM_inputs.py
"""

import os
import sys
HOME_DIR='G:'+os.sep+'Il Mio Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep+'network_signing'+os.sep+'SIGNAL_release'+os.sep
HOME_DIR='G:'+os.sep+'My Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep+'network_signing'+os.sep+'SIGNAL_release'+os.sep
sys.path.append(HOME_DIR)
import pandas  as pd
import numpy  as np
from datetime import date
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from glob_vars import SPECIES, PERT_MAP, TRAIN_DATA, HOME_DIR, LBL_DIR,\
    EDGES_DIR, FT_DIR, SIGNAL_DIR, MOD_DIR, KO_VAL_HOME,\
        KO_VAL_OUT, IMG_DIR, tau, nSPS, epsilon
from path_sign_reconstruction import predict_path_sign, \
    predict_path_sign_TLM_paths, calc_AUC
 
##############################################################################
#  INPUTS
##############################################################################

# Servier directories:
# HOME_DIR  =  '/home/of/SIGNAL'
# INPUT_DIR=HOME_DIR+"input"+os.sep+SPECIES+os.sep
# OUTDIR=HOME_DIR+'output'+os.sep+SPECIES+os.sep
# FEATURESDIR=OUTDIR
# BASENETDIR=INPUT_DIR

dataset_name = '_'.join(TRAIN_DATA)

print('tau = ', tau)
print('epsilon = ', epsilon)

#%% KT DATA:

TSIGN='MINUS'
with open(KO_VAL_OUT+TSIGN+'_SP_edges_per_source_target_pair_'+dataset_name+'_'+PERT_MAP+'.pkl', 'rb') as f:
    MINUSTSP=pickle.load(f)
TSIGN='PLUS'
with open(KO_VAL_OUT+TSIGN+'_SP_edges_per_source_target_pair_'+dataset_name+'_'+PERT_MAP+'.pkl', 'rb') as f:
    PLUSTSP=pickle.load(f)

#%% calculate signal:

Y_true, Y_pred,  = predict_path_sign(PLUSTSP, MINUSTSP, tau, n_SPs=nSPS, e=epsilon)
fprs, tprs, s_thresholds, AUC = calc_AUC(Y_true, Y_pred, True)

print(auc(fprs, tprs), 'e',epsilon, 'nSPS',nSPS)
#%% same function as in ipynotebook of data exploration
def heatmap_from_dataframe(dataframe, figname='', savefig=False):
    heatmap = dataframe.unstack().values

    # Get the y-axis labels from the index levels
    y_ticks = sorted(dataframe.unstack().index)

    # Get the x-axis labels from the index levels
    x_ticks = sorted(dataframe.unstack().columns)

    # Create the heatmap plot
    return x_ticks, y_ticks, heatmap

def build_figure_true_vs_pred(x_ticks_t, y_ticks_t, heatmap_t,\
                              x_ticks_p, y_ticks_p, heatmap_p,\
                                  figname='',savefig=False, titles=None):
    fig, ax = plt.subplots(nrows=1, ncols=2,\
                           figsize=(heatmap_t.shape[1]* 0.3 * 2 + 10 ,\
                                    heatmap_t.shape[0]*0.3+3))
    # Create heatmaps with consistent color scale
    
    
    im1 = ax[0].imshow(heatmap_t, cmap='coolwarm')
    im2 = ax[1].imshow(heatmap_p, cmap='coolwarm')

    # Set titles
    if titles:
        ax[0].set_title(titles[0], fontsize=14)
        ax[1].set_title(titles[1], fontsize=14)
    else:
        ax[0].set_title('True Values', fontsize=14)
        ax[1].set_title('Predicted Values', fontsize=14)
    
    # Set ticks and labels
    ax[0].set_xticks(np.arange(len(x_ticks_t)))
    ax[0].set_xticklabels(x_ticks_t, rotation=45, ha='right')
    ax[0].set_yticks(np.arange(len(y_ticks_t)))
    ax[0].set_yticklabels(y_ticks_t)
    
    ax[1].set_xticks(np.arange(len(x_ticks_p)))
    ax[1].set_xticklabels(x_ticks_p, rotation=45, ha='right')
    ax[1].set_yticks(np.arange(len(y_ticks_p)))
    ax[1].set_yticklabels(y_ticks_p)
    plt.subplots_adjust(bottom=0.15, left=0.1)
    # Improve layout
    plt.tight_layout()
    
    if savefig:
        fig.savefig(f"{IMG_DIR}{figname}.pdf", bbox_inches='tight', dpi=300)
    plt.show()
    
    return
#%%
Y_pred_ts=pd.Series(data=np.where(Y_pred>=0.5,1,0) , index=Y_pred.index)
#%%
x_ticks_t, y_ticks_t, heatmap_t = heatmap_from_dataframe(Y_true)
x_ticks_p, y_ticks_p, heatmap_p = heatmap_from_dataframe(Y_pred_ts)
build_figure_true_vs_pred(x_ticks_t, y_ticks_t, heatmap_t,\
                              x_ticks_p, y_ticks_p, heatmap_p,\
                                  figname='testone',savefig=True)
