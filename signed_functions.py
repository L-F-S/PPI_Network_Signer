# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:33:35 2021

@author: los4
"""

import numpy  as np
import matplotlib.pyplot as plt
import scipy
import math
from time import time

###############################################################################
#  GENERATE FEATURES
###############################################################################


# Simple propagation:
    
    
def generate_similarity_matrix(matrix):
    """Normalize a similarity matrix
    by number of outgoing edges from node.
    this normalization works for  both
    directed and undirected graphs"""

    sum_vector = matrix.sum(axis=1).A1 # sum over columns, then get a flattened array
    sum_vector[sum_vector == 0] = 1  #substitute 0s with ones , to avoid division by 0 later
    norm_matrix = scipy.sparse.diags(1/sum_vector) # create a diagonal matrix (all 0s except diagonal) with 1/sum vector
    matrix = matrix * norm_matrix #multiply the two. (effectively u are dividing each column by the sum of the column)

    return  matrix #19/04, cambiato da PROPAGATE_ALPHA * matrix a solo matrix, xke lo moltiplico dopo.


def propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, seeds, matrix, gene_indexes, num_genes):
    '''propagate from seeds inside matrix
    gene indexes is a dictinary mapping gene names to their indexes in matrix'''
    F_t = np.zeros(num_genes)
    F_t[[gene_indexes[seed] for seed in seeds if seed in gene_indexes]] = 1
    Y = (1 - PROPAGATE_ALPHA) * F_t

    for _ in range(PROPAGATE_ITERATIONS):
        F_t_1 = F_t
        F_t = PROPAGATE_ALPHA * matrix.dot(F_t_1) + Y #19/04 aggiunto propagate alpha qui invece che nella funzione sopra

        if math.sqrt(scipy.linalg.norm(F_t_1 - F_t)) < PROPAGATE_EPSILON:
            break
    return F_t


# Edge-defective propagation:
    

def extract_and_normalize_column_of(sparse_matrix, ind1, ind2):
    col1=sparse_matrix[:,ind1].copy()
    col2=sparse_matrix[:,ind2].copy()
    col1[ind2]=0
    col2[ind1]=0
    
    return col1/col1.sum() if col1.sum() !=0 else 1, col2/col2.sum() if col2.sum() !=0 else 1


def create_defective_columns(non_normalized_sparse_matrix, edge, gene_indexes):
    """takes a column-normalzied sparse matrix, normalized with A D^-1
    and removes one interaction from it, but keeps the normalization, effectlively
    re-normalizing two columns.
    inputs: scipy sparse matrix (csr), tuple, dictionary"""
    ind1=gene_indexes[edge[0]]
    ind2=gene_indexes[edge[1]]
    if ind1>ind2:
        temp=ind2
        ind2=ind1
        ind1=temp
    column1, column2=extract_and_normalize_column_of(non_normalized_sparse_matrix, ind1, ind2)
    return column1, column2, ind1, ind2


def defective_norm_matrix_from_columns(normalized_sparse_matrix, col1, col2, ind1, ind2):
    """takes a column-normalzied sparse matrix, normalized with A D^-1
    and removes one interaction from it, but keeps the normalization, effectlively
    re-normalizing two columns."""
    def_matrix=normalized_sparse_matrix.copy()
    def_matrix[:,ind1]=col1
    def_matrix[:,ind2]=col2
    return def_matrix


def score(prop_edge, prop_noedge,positive_foldchange_genes_indexes, negative_foldchange_genes_indexes=None, edge=None):
    # current score: Summ the differences of propagation scores (between normal 
    # vs defective network) of all positive foldchanges divided
    #  by sum of difference of propoagation scores of all negatively affected genes with a normalization
# normalizzare attraverso il sum(edge positive)/n(edge positive) *n/edge negative
    # e un modo brutto di scrivere AVg(Delta(P=+))/ Avg(delta(P-))
    if negative_foldchange_genes_indexes:
        try:
            norm = len(prop_edge[negative_foldchange_genes_indexes])*(sum(prop_edge[positive_foldchange_genes_indexes] - prop_noedge[positive_foldchange_genes_indexes]))/len(prop_edge[positive_foldchange_genes_indexes])
            return sum(prop_edge[negative_foldchange_genes_indexes] - prop_noedge[negative_foldchange_genes_indexes])/norm
        except:
            raise ValueError('Empty list of gene targets',positive_foldchange_genes_indexes, edge)

    else:
        try:
            return sum(prop_edge[positive_foldchange_genes_indexes]-prop_noedge[positive_foldchange_genes_indexes])/len(positive_foldchange_genes_indexes)
        except:
            raise ValueError('Empty list of gene targets',positive_foldchange_genes_indexes, edge)
     
            
# def create_the_features(raw_matrix, edge):
#     ''' TODO: CURRENTY UNUSED'''
#     col1, col2, ind1, ind2 = create_defective_columns(raw_matrix, edge, gene_indexes)
#     defective_normalized_matrix = defective_norm_matrix_from_columns(matrix, col1, col2,ind1, ind2 )
#     score_of=[edge]
    
#     for n, source in enumerate(plus_targets_of_deletion.keys()):
#         if n<5000:  #stop after a while
            
#             prop_edge_fwd = propagate({source},matrix,gene_indexes,num_genes)          
#             # create defective matrix and run defective propagation
#             prop_noedge_fwd = propagate({source}, defective_normalized_matrix,gene_indexes,num_genes) # num_genes is the same.
#             score_of.append(score(prop_edge_fwd, prop_noedge_fwd,[gene_indexes[x] for x in plus_targets_of_deletion[source]],[gene_indexes[x] for x in minus_targets_of_deletion[source]])) # [gene_indexes[x] for x in prefeatures_decreased_log_ratio[source]] is substituting node names with their relative index in the list

#     return score_of


def create_the_features_different_knockouts(raw_matrix, edge, gene_indexes, matrix, plus_targets_of_deletion, minus_targets_of_deletion, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON):
    
    col1, col2, ind1, ind2 = create_defective_columns(raw_matrix, edge, gene_indexes)
    defective_normalized_matrix = defective_norm_matrix_from_columns(matrix, col1, col2,ind1, ind2 )
    score_of=[edge]
    
    for n, source in enumerate(plus_targets_of_deletion.keys()):
        if n<5000:  #stop after a while
            
            prop_edge_fwd = propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, {source},matrix,gene_indexes,num_genes)          
            # create defective matrix and run defective propagation:
            prop_noedge_fwd = propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON,{source}, defective_normalized_matrix,gene_indexes,num_genes) # num_genes is the same.
            score_of.append(score(prop_edge_fwd, prop_noedge_fwd,[gene_indexes[x] for x in plus_targets_of_deletion[source]],edge=edge)) # [gene_indexes[x] for x in pplus_targets_of_deletio] is substituting node names with their relative index in the list


    for n,source in enumerate(minus_targets_of_deletion.keys()):
        if n<5000: #stop after a while

            prop_edge_fwd = propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, {source},matrix,gene_indexes,num_genes) 
            # create defective matrix and run defective propagation:
            prop_noedge_fwd = propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, {source}, defective_normalized_matrix, gene_indexes,num_genes) # num_genes is the same.            
            score_of.append(score(prop_edge_fwd, prop_noedge_fwd,[gene_indexes[x] for x in minus_targets_of_deletion[source]], edge=source)) 

    return score_of


###############################################################################
# TRAIN CLASSIFIER
###############################################################################

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve,  auc
from random import seed, sample


XLABEL_OF={"PrecRec":"Recall (TP/P)","ROC":"False Positive Rate"}
YLABEL_OF={"PrecRec":"Precision (TP/PredictedP)","ROC":"True Positive Rate"}
TITLE_OF ={"PrecRec":"Precision vs Recall","ROC":"ROC curve"}
COLOR_OF ={"PrecRec":"magenta","ROC":"cyan"}

def random_even_subsample(x,y, Seed):
    '''x is a pd.DataFrame (cam aslo be a ps. Series), y is a pd.Series
    x and y must have the same index. y must have binary values [0,1]
    returns reduced x and reduced y'''
    
    tot_0=y[y==0].shape[0]
    tot_1=y[y==1].shape[0]
    
    if tot_0>tot_1:
        lower, higher = 1, 0
    elif tot_0<tot_1:
        lower, higher = 0, 1
    else:
        return x, y

    tot_lower=y[y==lower].shape[0]
    tot_higher=y[y==higher].shape[0]
    indexes=y[y==higher].index
    
    seed(Seed) # set random sample seed, to make experiment repeatable
    rand_indexes_to_drop = sample(list(indexes), k = tot_higher - tot_lower)

    x=x.drop(rand_indexes_to_drop)
    y=y.drop(rand_indexes_to_drop)
    
    return x, y

def custom_k_fold(classifier, n_folds, x_train, y_train, dataname, IMG_DIR, metric="ROC", plot=True):
     

    interp_0_of={"PrecRec":1.0,"ROC":0.0}
    tprs = []  # list of np.arrays
    aucs = []  # list of floats
    mean_fpr = np.linspace(0, 1, 100)  # create a line to add points in (x axis of graph), for false psoitve rate

    cv=StratifiedKFold(n_splits=n_folds)   #splits TRAINING data into n folds.
    if plot:
        fig, ax =plt.subplots()
    for i, (train, test) in enumerate(cv.split(np.array(x_train), np.array(y_train))):
        
        # Train classifier with subset (updates every fold)
        classifier.fit(x_train.iloc[train], y_train.iloc[train])         
        # create metric curve (ROC or PrecRec). Needs X and Y for current fold's 
        # test set to calculate tpr and fpr (or other metrics) at different thresholds
        # this is if you use RandomForest instead of LogisticRegression
        y_predicted=classifier.predict_proba(x_train.iloc[test])[:,1:] # indexing this way because of output for predict_prob
            
        if metric=="PrecRec":
            tpr, fpr, thresholds = precision_recall_curve(y_train.iloc[test], y_predicted) # not actually fpr and tpr in this case, they are tprs=precision, fpr = recall
            tpr=tpr[::-1]  #they are reversed for some reason
            fpr=fpr[::-1]
        elif metric=="ROC":
            fpr, tpr, thresholds = roc_curve(y_train.iloc[test], y_predicted)
        roc_auc= auc(fpr, tpr) #float, the AUC
        aucs.append(roc_auc)   # list of all aucs, called from a plot_roc_curve's method
        
        #interpolate line of true positive rates. We interpolate cos every fold will output a 
        # tprs (or other metric) with different lengths, so in order to have same lengths to take the mean later
        interp_tpr = np.interp(mean_fpr, fpr, tpr) #linear interpolation between points. 
        # To evaluate on coordinates: mean_fpr. Points to use: x data points: viz.fpr; y datapoints = viz.tpr
        # this is so that we can plot for every mean_fpr point (i.e. for every point on the x axis), the (sort of) corresponding 
        # point on the y axis (i.e. the tpr, or Recall, or other metric)
        interp_tpr[0] = interp_0_of[metric] #0.0 for ROC, 1.0 for PrecRec
        tprs.append(interp_tpr) #the list of y points to take the mean on
        
        # plot the curve (one for every fold)
        if plot:
            ax.plot(fpr, tpr, lw=1, label ="k {}".format(i+1), alpha=0.3)
            ax.set_xlabel(XLABEL_OF[metric])
            ax.set_ylabel(YLABEL_OF[metric])


    ########
    #  Plot x y bisector:
    if metric == "ROC":
        if plot:
            ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray',
                label='bisector', alpha=.8)

    # plot the mean ROC, taking the mean of all other ROCs
    mean_tpr = np.mean(tprs, axis=0) #takes point-wise mean for all tpr lines
    #mean_tpr[-1] = 1.0  
    mean_auc = auc(mean_fpr, mean_tpr)   #remember mean_fpr is just the x axis.(this is why we interpolated the y axes)
    std_auc = np.std(aucs)
    if plot:
        ax.plot(mean_fpr, mean_tpr, color=COLOR_OF[metric],
            label=r'Mean {} (AUC = {} $\pm$ {})'.format(metric, np.round(mean_auc,2), np.round(std_auc,2)),
            lw=2)

    # Plot one standard deviation above and below the mean
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    if plot:
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=COLOR_OF[metric], alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

        # Add title, legend, set xlim and ylim
        ax.set( xlim=[0, 1.1], ylim=[0, 1],
                title=TITLE_OF[metric]+" for classifier = "+str(classifier))
        ax.legend(loc=(1.1, 0))
        
    fig_name=dataname+"."+metric
    plt.savefig(IMG_DIR+fig_name+'2.png',  bbox_inches='tight')

    return fig, classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper, mean_auc
