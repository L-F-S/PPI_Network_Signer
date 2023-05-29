# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:33:35 2021

@author: los4
"""

import numpy  as np
import scipy
import scipy.linalg
import math
import warnings
warnings.filterwarnings("ignore")

###############################################################################
#  GENERATE FEATURES
###############################################################################


# Simple propagation:
    
    
def generate_similarity_matrix(matrix):
    """Normalize a similarity matrix
    by number of outgoing edges from node.
    this normalization works for  both
    directed and undirected graphs
    inputs:
        matrix: Scipy.sparse matrix (NOT numpy array)"""
# 05 04 2022 rimosso .A1 dalla riga sotto xke e una nparray e non piu una  np.matrix e quindi non gli serve fare il flattening.
    sum_vector = matrix.sum(axis=0).A1 # sum over rows (along columns), then get a flattened array
    sum_vector[sum_vector == 0] = 1  #substitute 0s with ones , to avoid division by 0 later
    norm_matrix = scipy.sparse.diags(1/sum_vector) # create a diagonal matrix (all 0s except diagonal) with 1/sum vector
    matrix = matrix * norm_matrix #multiply the two. (effectively u are dividing each column by the sum of the column). WARNING: * is the matrix multiplication op[erator for scipy sparse matrices. For numpy arrays, it is element-wise product. MAke sure that matrix is a scipy.sparse matrix

    return  matrix, sum_vector 


def propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, seeds, matrix, gene_indexes, num_genes):
    '''propagate from seeds inside matrix
    gene indexes is a dictinary mapping gene names to their indexes in matrix'''
    
    # matrix = np.array(matrix)
    F_t = np.zeros(num_genes)
    F_t[[gene_indexes[seed] for seed in seeds if seed in gene_indexes]] = 1
    Y = (1 - PROPAGATE_ALPHA) * F_t

    for _ in range(PROPAGATE_ITERATIONS):
        F_t_1 = F_t  #05 045 2022v add transpose perche adesso matrix e  una np.matrix e bo se no mi da errore
        F_t =  PROPAGATE_ALPHA*matrix.dot(F_t_1) + Y #105 094 2022, rimossa la mia modifica in cui mettevo PROPAGATE_ALPHA* matrix qui (e lhor rimesso nel return della funzione sopra) perchedava errore se matrix e' np.matrix, e non scipy.sparse. cazzo. CAZZO. LOL
        
        if math.sqrt(scipy.linalg.norm(F_t_1 - F_t)) < PROPAGATE_EPSILON:
            break
    return F_t

# Edge-defective propagation:


def extract_and_normalize_column_of(matrix, ind1, ind2):
    #05 04 added intexing [0] to return perche non e piu sparse amtrix ma numpy.matrix
    col1=matrix[:,ind1].copy()
    col2=matrix[:,ind2].copy()
    col1[ind2]=0
    col2[ind1]=0
    
    return col1/col1.sum() if col1.sum() !=0 else 1, col2/col2.sum() if col2.sum() !=0 else 1


def create_defective_columns(non_normalized_sparse_matrix, edge, gene_indexes):
    """
    takes a column-normalzied sparse matrix, normalized with A D^-1
    and removes one interaction from it, but keeps the normalization,
    effectlively re-normalizing two columns.
    inputs: scipy sparse matrix (csr), tuple, dictionary.
    """
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


def score(prop_all, prop_noedge,positive_foldchange_genes_indexes, negative_foldchange_genes_indexes=None, edge=None):
    # current score: Summ the differences of propagation scores (between normal 
    # vs defective network) of all positive foldchanges divided
    #  by sum of difference of propoagation scores of all negatively affected genes with a normalization
# normalizzare attraverso il sum(edge positive)/n(edge positive) *n/edge negative
    # e un modo brutto di scrivere AVg(Delta(P=+))/ Avg(delta(P-))
    '''returns a float'''
    
    if negative_foldchange_genes_indexes: #todo 01/09/2021 currently deprecated
        try:
            norm = len(prop_all[negative_foldchange_genes_indexes])*(sum(prop_all[positive_foldchange_genes_indexes] - prop_noedge[positive_foldchange_genes_indexes]))/len(prop_all[positive_foldchange_genes_indexes])
            return sum(prop_all[negative_foldchange_genes_indexes] - prop_noedge[negative_foldchange_genes_indexes])/norm
        except:
            raise ValueError('Empty list of gene targets',positive_foldchange_genes_indexes, edge)

    else: # actually using this. positive_foldchange_genes_indexes can actually be negative, depending on what you input
        try:
            return sum(prop_all[positive_foldchange_genes_indexes]-prop_noedge[positive_foldchange_genes_indexes])/len(positive_foldchange_genes_indexes)
        except:
            raise ValueError('Empty list of gene targets',positive_foldchange_genes_indexes, edge)


def create_the_features_different_knockouts(raw_matrix, edge, gene_indexes, matrix, plus_targets_of_deletion, minus_targets_of_deletion, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON):
    '''fills one row of features for one sample (edge) for the features matrix to feed to the classifier.
    
    raw_matrix: the base network matrix
    matrix: normalized similarity matrix by number of outgoing edges
    edge: tuple of ints. ints are the nodes IDs. the training edge with known orientation. Also, the edge to be removed to create each feature
    gene_indexes: dict of geneid:index in the genes list (outside this function)
    plus/minus_targets_of_deletion: dict. {knockoutgene:list of target positive(?negative) genes}
    num genes: int. number of genes
    PROPAGATE_x: ints. propagation parameters
    returns:
        knockout_names:  list [0,1,knockout1+,knockout2+...knockoutn+,
                               knockout1-,...knockoutn-]
        score_of: list  [edge1ID,edge2ID, score1,....score2n]
    '''
    col1, col2, ind1, ind2 = create_defective_columns(raw_matrix, edge, gene_indexes)
    defective_normalized_matrix = defective_norm_matrix_from_columns(matrix, col1, col2,ind1, ind2 )

    score_of=[edge[0], edge[1]]
    knockout_names = [0,1]
    
    for n, source in enumerate(plus_targets_of_deletion.keys()):
            prop_edge_fwd = propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, {source},matrix,gene_indexes,num_genes)  
            # create defective matrix and run defective propagation:
            prop_noedge_fwd = propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON,{source}, defective_normalized_matrix,gene_indexes,num_genes) # num_genes is the same.
            score_of.append(score(prop_edge_fwd, prop_noedge_fwd,[gene_indexes[x] for x in plus_targets_of_deletion[source]],edge=edge)) # [gene_indexes[x] for x in pplus_targets_of_deletio] is substituting node names with their relative index in the list
            knockout_names.append(str(source)+'+')
            

    for n,source in enumerate(minus_targets_of_deletion.keys()): #01/03/2022 TODO prop_edge_fwd e prop_noedge_fwd in questo ciclo non serve farli, per come sono plus?minus_targets of deletion ora, perche' hanno le stesse keys per ora, xke prendo solo knockouts che hanno sia plus che minus targets dentro, quindi per ora questa propagazione e' ridondante.

            prop_edge_fwd = propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, {source},matrix,gene_indexes,num_genes) 
            # create defective matrix and run defective propagation:
            prop_noedge_fwd = propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, {source}, defective_normalized_matrix, gene_indexes,num_genes) # num_genes is the same.            
            score_of.append(score(prop_edge_fwd, prop_noedge_fwd,[gene_indexes[x] for x in minus_targets_of_deletion[source]], edge=source)) 
            knockout_names.append(str(source)+'-')

    return knockout_names, score_of 

def create_the_features_different_knockouts_iterative(raw_matrix, edges, gene_indexes, matrix, plus_targets_of_deletion, minus_targets_of_deletion, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON):
    '''fills one row of features for one sample (edge) for the features matrix to feed to the classifier.
    
    raw_matrix: the base network matrix
    matrix: normalized similarity matrix by number of outgoing edges
    edge: tuple of ints. ints are the nodes IDs. the training edge with known orientation. Also, the edge to be removed to create each feature
    gene_indexes: dict of geneid:index in the genes list (outside this function)
    plus/minus_targets_of_deletion: dict. {knockoutgene:list of target positive(?negative) genes}
    num genes: int. number of genes
    PROPAGATE_x: ints. propagation parameters
    returns:
        knockout_names:  list [0,1,knockout1+,knockout2+...knockoutn+,
                               knockout1-,...knockoutn-]
        score_of: list  [edge1ID,edge2ID, score1,....score2n]
    '''
    scores_of=[]
    knockout_names_of=[]
    for edge in edges:
        # col1, col2, ind1, ind2 = create_defective_columns(raw_matrix, edge, gene_indexes)
        defective_normalized_matrix = matrix#defective_norm_matrix_from_columns(matrix, col1, col2,ind1, ind2 )
    
        score_of=[edge[0], edge[1]]
        knockout_names = [0,1]
        
        for n, source in enumerate(list(plus_targets_of_deletion.keys())):
                # print(n)
                prop_edge_fwd = propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, {source},matrix,gene_indexes,num_genes)  
                # create defective matrix and run defective propagation:
                prop_noedge_fwd = propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON,{source}, defective_normalized_matrix,gene_indexes,num_genes) # num_genes is the same.
                score_of.append(score(prop_edge_fwd, prop_noedge_fwd,[gene_indexes[x] for x in plus_targets_of_deletion[source]],edge=edge)) # [gene_indexes[x] for x in pplus_targets_of_deletio] is substituting node names with their relative index in the list
                knockout_names.append(str(source)+'+')
                
    
        for n,source in enumerate(list(minus_targets_of_deletion.keys())): #01/03/2022 TODO prop_edge_fwd e prop_noedge_fwd in questo ciclo non serve farli, per come sono plus?minus_targets of deletion ora, perche' hanno le stesse keys per ora, xke prendo solo knockouts che hanno sia plus che minus targets dentro, quindi per ora questa propagazione e' ridondante.
    
                prop_edge_fwd = propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, {source},matrix,gene_indexes,num_genes) 
                # create defective matrix and run defective propagation:
                prop_noedge_fwd = propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, {source}, defective_normalized_matrix, gene_indexes,num_genes) # num_genes is the same.            
                score_of.append(score(prop_edge_fwd, prop_noedge_fwd,[gene_indexes[x] for x in minus_targets_of_deletion[source]], edge=source)) 
                knockout_names.append(str(source)+'-')
    
        knockout_names_of.append(knockout_names)
        scores_of.append(score_of)
    return knockout_names_of, scores_of



#04 - 04 - 2022  :
    # cambiato metodo di creazione defective matrix ma e' molto piu lento nn so xke
    #quindi nisba


def create_defective_matrix2(normalized_sparse_matrix, raw_matrix_column_sum_1, raw_matrix_column_sum_2, ind_1, ind_2, raw_weight_1, raw_weight_2):
    '''
    Column sum: int, s
    '''
    def_matrix=normalized_sparse_matrix.copy()
    
    # set edge weight to 0 (works for both directed and undirected graphs)
    def_matrix[ind_2,ind_1] = 0
    def_matrix[ind_1,ind_2] = 0

    def_matrix[:,ind_1] = def_matrix[:,ind_1]*(raw_matrix_column_sum_1/(raw_matrix_column_sum_1-raw_weight_1)) 
    def_matrix[:,ind_2] = def_matrix[:,ind_2]*(raw_matrix_column_sum_2/(raw_matrix_column_sum_2-raw_weight_2))
    
    return def_matrix

def get_indexes_and_columns2(raw_matrix, gene_indexes, raw_col_sums, edge):
    '''
    note: raw_matrix[ind2,ind1] takes the weight of the edge, from the column 
    of ind1, so given our nomenclature, it will be edge_weight_1
    '''
    
    ind1=gene_indexes[edge[0]]
    ind2=gene_indexes[edge[1]]
    return ind1, ind2, raw_matrix[ind2,ind1], raw_matrix[ind1, ind2], raw_col_sums[ind1], raw_col_sums[ind2]

def create_the_features_different_knockouts2(raw_matrix, edge, gene_indexes, matrix, plus_targets_of_deletion, minus_targets_of_deletion, num_genes, raw_col_sums, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON):
    #04 04 2022 NON e' piu  veloce...
    '''
    fills one row of features for one sample (edge) for the features matrix to feed to the classifier.
    raw_matrix: the base network matrix
    matrix: normalized similarity matrix by number of outgoing edges
    edge: tuple of ints. ints are the nodes IDs. the training edge with known orientation. Also, the edge to be removed to create each feature
    gene_indexes: dict of geneid:index in the genes list (outside this function)
    plus/minus_targets_of_deletion: dict. {knockoutgene:list of target positive(?negative) genes}
    raw_col_sums: list of sum over non normalized columns (no 0 values)
    num genes: int. number of genes
    PROPAGATE_x: ints. propagation parameters
    '''

    tic1 = time.time()
    ind_1, ind_2, edge_weight_1, edge_weight_2, col_sum1, col_sum2 = get_indexes_and_columns2(raw_matrix, gene_indexes, raw_col_sums, edge)
    tic2 = time.time()
    defective_normalized_matrix = create_defective_matrix2(matrix, col_sum1, col_sum2, ind_1, ind_2, edge_weight_1, edge_weight_2)
    tic3 = time.time()
    

    
    score_of=[edge[0], edge[1]]

    for n, source in enumerate(plus_targets_of_deletion.keys()):
        if n<5000:  # small numbers to not iterate over all knockouts
            
            prop_edge_fwd = propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, {source},matrix,gene_indexes,num_genes)          
            # create defective matrix and run defective propagation:
            prop_noedge_fwd = propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON,{source}, defective_normalized_matrix,gene_indexes,num_genes) # num_genes is the same.
            score_of.append(score(prop_edge_fwd, prop_noedge_fwd,[gene_indexes[x] for x in plus_targets_of_deletion[source]],edge=edge)) # [gene_indexes[x] for x in pplus_targets_of_deletio] is substituting node names with their relative index in the list


    for n,source in enumerate(minus_targets_of_deletion.keys()): #01/03/2022 TODO prop_edge_fwd e prop_noedge_fwd in questo ciclo non serve farli, per come sono plus?minus_targets of deletion ora, perche' hanno le stesse keys per ora, xke prendo solo knockouts che hanno sia plus che minus targets dentro, quindi per ora questa propagazione e' ridondante.
        if n<5000: # small numbers to not iterate over all knockouts

            prop_edge_fwd = propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, {source},matrix,gene_indexes,num_genes) 
            # create defective matrix and run defective propagation:
            prop_noedge_fwd = propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, {source}, defective_normalized_matrix, gene_indexes,num_genes) # num_genes is the same.            
            score_of.append(score(prop_edge_fwd, prop_noedge_fwd,[gene_indexes[x] for x in minus_targets_of_deletion[source]], edge=source)) 

    return score_of #1/3/2022 al momentp e' una lista con [edge1ID,edge2ID, score1,....score472]
