# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:33:35 2021

@author: L-F-S

###############################################################################
#                  SIGNAL feature creation algorithm
###############################################################################

"""

import numpy  as np
import scipy
import scipy.linalg
import math
import warnings
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")




# Simple propagation:
    
def generate_similarity_matrix(matrix):
    """
    AD^-1 matrix normalization.
    Normalize an adjacency matrix by number of outgoing edges from node.
    Works for  both directed and undirected graphs.
    Inputs:
        matrix: scipy.sparse._csr.csr_matrix"""

    if not type(matrix)==scipy.sparse._csr.csr_matrix:
        raise ValueError('Wrong matrix type:', type(matrix), 'Convert to scipy.sparse._csr.csr_matrix.')

    # Create inverse diagonal matrix of inverse row sums
    sum_vector = matrix.sum(axis=0).A1 
    sum_vector[sum_vector == 0] = 1  
    norm_matrix = scipy.sparse.diags(1/sum_vector) 

    # multiply raw matrix with diagonal matrix.
    # WARNING: * is the matrix multiplication operator for scipy sparse matrices. 
    # For numpy arrays, it is element-wise product: Make sure that matrix is a scipy.sparse matrix
    matrix = matrix * norm_matrix 

    return  matrix, sum_vector 


def propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, seeds, matrix, gene_indexes, num_genes):
    '''Propagate from seeds inside matrix
    gene indexes is a dictinary mapping gene names to their indexes in matrix'''
    
    # matrix = np.array(matrix)
    P_t = np.zeros(num_genes)
    P_t[[gene_indexes[seed] for seed in seeds if seed in gene_indexes]] = 1
    Y = (1 - PROPAGATE_ALPHA) * P_t

    for _ in range(PROPAGATE_ITERATIONS):
        P_t_1 = P_t  #05 045 2022v add transpose perche adesso matrix e  una np.matrix e bo se no mi da errore
        P_t =  PROPAGATE_ALPHA*matrix.dot(P_t_1) + Y #105 094 2022, rimossa la mia modifica in cui mettevo PROPAGATE_ALPHA* matrix qui (e lhor rimesso nel return della funzione sopra) perchedava errore se matrix e' np.matrix, e non scipy.sparse.
        
        if math.sqrt(scipy.linalg.norm(P_t_1 - P_t)) < PROPAGATE_EPSILON:
            break
    return P_t

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
        col1, col2, ind1, ind2 = create_defective_columns(raw_matrix, edge, gene_indexes)
        defective_normalized_matrix = defective_norm_matrix_from_columns(matrix, col1, col2,ind1, ind2 )
    
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



#%% 26 09 2024 optimized iterative feature creation

def defective_matrix_wrapper(raw_matrix, matrix, edge, gene_indexes):
  """Builds the defective PPI matrix for a given row of the feature matrix.
  edge: tuple of geneids of the given edge, representing the feature matrix row"""
  
  col1, col2, ind1, ind2 = create_defective_columns(raw_matrix, edge, gene_indexes)
  defective_normalized_matrix = defective_norm_matrix_from_columns(matrix, col1, col2,ind1, ind2 )

  return defective_normalized_matrix

def S_score(defective_normalized_matrix, prop_edge_fwd, edge, source, plus_targets_of_deletion, minus_targets_of_deletion, gene_indexes, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON):
  """Combines row and column scores to get the element score.
  
  defective_normalized_matrix: row-wise defective normalized matrix. Previously calculated because it is invariant across rows

  prop_edge_fwd: column-wise array of propagation scores from knockout K. Previously calculated because it is invariant across columns

  source: knockout K gene id

  """
  prop_noedge_fwd = propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON,{source}, defective_normalized_matrix, gene_indexes,num_genes)

  score_of_cell_plus = score(prop_edge_fwd, prop_noedge_fwd,[gene_indexes[x] for x in plus_targets_of_deletion[source]],edge=edge)
  score_of_cell_minus = score(prop_edge_fwd, prop_noedge_fwd,[gene_indexes[x] for x in minus_targets_of_deletion[source]],edge=edge)

  return score_of_cell_plus, score_of_cell_minus

def create_the_features_different_knockouts_iterative_optimized(edges, plus_targets_of_deletion, minus_targets_of_deletion, raw_matrix, matrix, gene_indexes, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON):
  """Fills a 2d feature matrix of SIGNAL features. does less steps than the old create_the_features_different_knockouts_iterative"""

  # Calculate row and column scores 
  row_names=edges
  col_names=list(plus_targets_of_deletion.keys())+list(minus_targets_of_deletion.keys()) #they are the same keys repeated
  row_defective_matrices = np.array([defective_matrix_wrapper(raw_matrix, matrix, i, gene_indexes) for i in row_names])
  col_propagation_scores = np.array([propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, {j}, matrix, gene_indexes, num_genes) for j in col_names])
  # Create an empty matrix
  features_matrix = np.zeros((len(row_names), len(col_names)), dtype=np.float32)

  # Fill the matrix element-wise using broadcasting and the combine_scores function
  for i, edge in enumerate(edges):
    # fill 2 column indexes at a time
    for j, source in enumerate(col_names[:int(len(col_names)/2)]):
      features_matrix[i, [j,j+int(len(col_names)/2)]] = S_score(row_defective_matrices[i], col_propagation_scores[j], edge, source, plus_targets_of_deletion, minus_targets_of_deletion, gene_indexes, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON)
      
  # Transfer the filled matrix back to the CPU as a NumPy array
  return features_matrix

def create_the_features_different_knockouts_parallel_optimized(N_JOBS, edges, plus_targets_of_deletion, minus_targets_of_deletion, raw_matrix, matrix, gene_indexes, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON):
  """Fills a 2d feature matrix of SIGNAL features. does less steps than the old create_the_features_different_knockouts_iterative"""

  # Calculate row and column scores on the GPU
  row_names=edges
  col_names=list(plus_targets_of_deletion.keys())+list(minus_targets_of_deletion.keys()) #they are the same keys repeated
  row_defective_matrices = np.array([defective_matrix_wrapper(raw_matrix, matrix, i, gene_indexes) for i in row_names])
  col_propagation_scores = np.array([propagate(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, {j}, matrix, gene_indexes, num_genes) for j in col_names])


  # Fill the matrix element-wise using broadcasting and the combine_scores function

  def fill_row(i,edge,row_defective_matrices, col_propagation_scores, col_names, plus_targets_of_deletion, minus_targets_of_deletion, gene_indexes, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON):
    row_plus, row_minus= [], []
    for j, source in enumerate(col_names[:int(len(col_names)/2)]):
      score_of_cell_plus, score_of_cell_minus = S_score(row_defective_matrices[i], col_propagation_scores[j], edge, source, plus_targets_of_deletion, minus_targets_of_deletion, gene_indexes, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON)
      row_plus.append(score_of_cell_plus)
      row_minus.append(score_of_cell_minus)
    return row_plus+ row_minus
  

  features_matrix = Parallel(n_jobs=N_JOBS)(delayed(fill_row)\
                                (i,edge,row_defective_matrices, col_propagation_scores, col_names, plus_targets_of_deletion, minus_targets_of_deletion, gene_indexes, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON)\
                                    for i, edge in enumerate(edges))
  
      
  # Transfer the filled matrix back to the CPU as a NumPy array
  return features_matrix

#### cupy GPU optimized calculations
# import cupy as cp

# def propagate_gpu(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, seeds, matrix, gene_indexes, num_genes):
#     '''Propagate from seeds inside matrix
#     gene indexes is a dictinary mapping gene names to their indexes in matrix
#     matrix must be a cupyx.scipy.sparse._csr.csr_matrix'''

#     if not type(matrix) == cp.sparse._csr.csr_matrix:
#         return ('ERROR! matrix is not a cp.sparse.csr.csr_matrix', type(matrix))
    
#     # matrix = np.array(matrix)
#     F_t = cp.zeros(num_genes)
#     F_t[[gene_indexes[seed] for seed in seeds if seed in gene_indexes]] = 1
#     Y = (1 - PROPAGATE_ALPHA) * F_t

#     for _ in range(PROPAGATE_ITERATIONS):
#         F_t_1 = F_t 
#         F_t =  PROPAGATE_ALPHA*matrix.dot(F_t_1) + Y
        
#         if math.sqrt(cp.linalg.norm(F_t_1 - F_t)) < PROPAGATE_EPSILON:
#             break
#     return F_t

# def S_score_gpu(defective_normalized_matrix, prop_edge_fwd, edge, source, plus_targets_of_deletion, minus_targets_of_deletion, gene_indexes, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON):
#   """Combines row and column scores to get the element score.
  
#   defective_normalized_matrix: row-wise defective normalized matrix. Previously calculated because it is invariant across rows

#   prop_edge_fwd: column-wise array of propagation scores from knockout K. Previously calculated because it is invariant across columns

#   source: knockout K gene id

#   """
#   prop_noedge_fwd = propagate_gpu(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON,{source}, defective_normalized_matrix, gene_indexes,num_genes)

#   score_of_cell_plus = score(prop_edge_fwd, prop_noedge_fwd,[gene_indexes[x] for x in plus_targets_of_deletion[source]],edge=edge)
#   score_of_cell_minus = score(prop_edge_fwd, prop_noedge_fwd,[gene_indexes[x] for x in minus_targets_of_deletion[source]],edge=edge)

#   return score_of_cell_plus, score_of_cell_minus

# def create_the_features_different_knockouts_optimized_gpu(edges, plus_targets_of_deletion, minus_targets_of_deletion, raw_matrix, matrix, gene_indexes, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON):
#   """Fills a 2d feature matrix of SIGNAL features. does less steps than the old create_the_features_different_knockouts_iterative
  
#   raw_matrix and matrix must be cupy cp.sparse._csr.csr_matrix"""

#   # Calculate row and column scores on the GPU
#   row_names=edges
#   col_names=list(plus_targets_of_deletion.keys())+list(minus_targets_of_deletion.keys()) #they are the same keys repeated
#   row_defective_matrices = [defective_matrix_wrapper(raw_matrix, matrix, i, gene_indexes) for i in row_names] # cupy cannot hold a list of sparse matrices object, so we keep it as a list
#   col_propagation_scores = [propagate_gpu(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, {j}, matrix, gene_indexes, num_genes) for j in col_names]
#   # Create an empty matrix on the GPU
#   features_matrix = cp.zeros((len(row_names), len(col_names)), dtype=np.float32)

#   # Fill the matrix element-wise using broadcasting and the combine_scores function
#   for i, edge in enumerate(edges):
#     # fill 2 column indexes at a time
#     for j, source in enumerate(col_names[:int(len(col_names)/2)]):
#       features_matrix[i, [j,j+int(len(col_names)/2)]] = S_score_gpu(row_defective_matrices[i], col_propagation_scores[j], edge, source, plus_targets_of_deletion, minus_targets_of_deletion, gene_indexes, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON)
      
#   # Transfer the filled matrix back to the CPU as a NumPy array
#   return cp.asnumpy(features_matrix)

# def create_the_features_different_knockouts_parallel_optimized_gpu(N_JOBS, edges, plus_targets_of_deletion, minus_targets_of_deletion, raw_matrix, matrix, gene_indexes, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON):
#   """Fills a 2d feature matrix of SIGNAL features. does less steps than the old create_the_features_different_knockouts_iterative"""

#   # Calculate row and column scores on the GPU
#   row_names=edges
#   col_names=list(plus_targets_of_deletion.keys())+list(minus_targets_of_deletion.keys()) #they are the same keys repeated
#   row_defective_matrices = [defective_matrix_wrapper(raw_matrix, matrix, i, gene_indexes) for i in row_names] # cupy cannot hold a list of sparse matrices object, so we keep it as a list
#   col_propagation_scores = [propagate_gpu(PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON, {j}, matrix, gene_indexes, num_genes) for j in col_names]

#   # Fill the matrix element-wise using broadcasting and the combine_scores function

#   def fill_row(i,edge,row_defective_matrices, col_propagation_scores, col_names, plus_targets_of_deletion, minus_targets_of_deletion, gene_indexes, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON):
#     row_plus, row_minus= [], []
#     for j, source in enumerate(col_names[:int(len(col_names)/2)]):
#       score_of_cell_plus, score_of_cell_minus = S_score_gpu(row_defective_matrices[i], col_propagation_scores[j], edge, source, plus_targets_of_deletion, minus_targets_of_deletion, gene_indexes, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON)
#       row_plus.append(cp.asnumpy(score_of_cell_plus))
#       row_minus.append(cp.asnumpy(score_of_cell_minus))
#     return row_plus+ row_minus
  

#   features_matrix = Parallel(n_jobs=N_JOBS)(delayed(fill_row)\
#                                 (i,edge,row_defective_matrices, col_propagation_scores, col_names, plus_targets_of_deletion, minus_targets_of_deletion, gene_indexes, num_genes, PROPAGATE_ALPHA, PROPAGATE_ITERATIONS, PROPAGATE_EPSILON)\
#                                     for i, edge in enumerate(edges))
  
      
#   return features_matrix


