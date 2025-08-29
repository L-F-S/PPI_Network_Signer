# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 19:07:47 2025

@author: L-F-S
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from glob_vars import tau, epsilon, nSPS

def signal_threshold(p=tau, e=epsilon):
    return p+e, p-e

def score_negative_paths(paths,t):
    Sminus=0
    Splus=0
    for path in paths:
        if path.size>1:
            #transform into a (0,1) array based on threshold (1= negative edge)
            neg_edges_p= np.where(path[:,1]>=t, 1, 0) 
            
            if sum(neg_edges_p)%2==1:
                Sminus+=1
            else:
                Splus+=1

    return round(Sminus/(Sminus+Splus),3)
    
def predict_path_sign(PLUSTSP, MINUSTSP,t=tau, n_SPs=nSPS, e=epsilon):
    ''' taking into account reversal of signs'''
    Y_true = []
    Y_true_ind = []
    SplusT = defaultdict(dict)
    SminusT = defaultdict(dict)

    minus_th, _ = signal_threshold(t, e)    

    # S score for negative paths for positive KT pairs
    for KO, SPSdict in PLUSTSP.items():
        for T, SPS in SPSdict.items():
            pair=(KO,T)
            if (len(SPS)>n_SPs): #change here which pathways to use
                SplusT[pair] = score_negative_paths(SPS,minus_th)
                Y_true+=[1] # 0 is + T, 1 is -T
                Y_true_ind.append(pair)
    
    # S score for negative paths for positive KT pairs
    for KO, SPSdict in MINUSTSP.items():
        for T, SPS in SPSdict.items():
            pair=(KO,T)
            if len(SPS)>n_SPs: #change here which pathways to use
                SminusT[pair] = score_negative_paths(SPS,minus_th)
                Y_true+=[0] # 0 is + T, 1 is -T
                Y_true_ind.append(pair)
    SplusT=pd.Series(SplusT)
    SminusT=pd.Series(SminusT)
    
    return pd.Series(Y_true, index=pd.MultiIndex.from_tuples(Y_true_ind)), pd.concat([SplusT,SminusT])


def score_negative_paths_list(paths,t):
    '''calclates negscore for list of lists of 
    shortest paths'''
    
    Sminus=0
    Splus=0
    for path in paths:
        if path.size>1:
            # print('p',path)
            #transform into a (0,1) array based on threshold (1= negative edge)
            neg_edges_p= np.where(path>=t, 1, 0) 
            # print('n', neg_edges_p)
            if sum(neg_edges_p)%2==1:
                Sminus+=1
            else:
                Splus+=1
                # print('Sminus',Sminus,'Splus',Splus)
    return round(Sminus/(Sminus+Splus),3)


def predict_path_sign_TLM_paths(PLUSTSP, MINUSTSP,t=tau, n_SPs=nSPS, e=epsilon):
    ''' taking into account reversal of signs
    Input is a dictionary of dictionaries of SPs of Signal scores 
    (instead of wd array of (1-SIGNAL, SIGNAL) for KT data )
    WARNING: signs are inverted wrt KO validation,
    because KO actually are supposed to show the inverse of the sign
    (gene upregulated with KO = downregulated in standard condition ), 
    since this function is copied from calcROC,
    I kept the signs unchanged, so you currently need to input the - with the + and vice versa.
    '''
    Y_true = []
    Y_true_ind = []
    SplusT = defaultdict(dict)
    SminusT = defaultdict(dict)

    minus_th, _ = signal_threshold(t, e)    

    # S score for negative paths for positive KT pairs
    for KO, SPSdict in PLUSTSP.items():
        for T, SPS in SPSdict.items():
            pair=(KO,T)
            if len(SPS)>n_SPs:
                SplusT[pair] = score_negative_paths_list(SPS,minus_th)
                Y_true+=[1] # 0 is + T, 1 is -T
                Y_true_ind.append(pair)
    
    # S score for negative paths for positive KT pairs
    for KO, SPSdict in MINUSTSP.items():
        for T, SPS in SPSdict.items():
            pair=(KO,T)
            if len(SPS)>n_SPs:
                SminusT[pair] = score_negative_paths_list(SPS,minus_th)
                Y_true+=[0] # 0 is + T, 1 is -T
                Y_true_ind.append(pair)

    SplusT=pd.Series(SplusT)
    SminusT=pd.Series(SminusT)
    print('\t\t+',len(SplusT), '-', len(SminusT), '% of 1 in Y_true', sum(Y_true)/len(Y_true))
    return pd.Series(Y_true, index=pd.MultiIndex.from_tuples(Y_true_ind)), pd.concat([SplusT,SminusT])


def calc_AUC(Y_true, Y_pred, precrec=False):
    if precrec:
        precs, recs, s_thresholds = precision_recall_curve(Y_true, Y_pred)
        precs=precs[::-1]  #they are reversed for some reason
        recs=recs[::-1]
        aucvalue=auc(recs, precs)
        return recs, precs, s_thresholds, aucvalue
    
    else:
        fprs, tprs, s_thresholds = roc_curve(Y_true, Y_pred)
        aucvalue=auc(fprs, tprs)
        return fprs, tprs, s_thresholds, aucvalue
