# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 12:57:58 2025

@author: Lorenzo Federico Signorini

"""

import requests
import gzip
import csv
from collections import defaultdict
import sys
import os
import pandas as pd
import numpy as np
import pickle
import gseapy as gs
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
import  matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#%%
DRIVE_PATH='G:'+os.sep+'Il Mio Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep

# DRIVE_PATH='G:'+os.sep+'My Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep
sys.path.append(DRIVE_PATH+'network_signing'+os.sep)
from glob_vars import SPECIES, TRAIN_DATA, PERT_MAP, HOME_DIR, LBL_DIR,\
    EDGES_DIR, FT_DIR, SIGNAL_DIR, MOD_DIR,ENRICHMENT_OUT, DICT_FILE, DICT_DIR,\
    BIO_DIR

with open(DICT_DIR+DICT_FILE,'rb') as f:
    alias_2geneid=pickle.load(f)
#%% Load data
net_filename = 'signed_base_net_patkar_kegg_kpi_ubinet2_Kemmeren.sgnl'
data = pd.read_csv(SIGNAL_DIR+net_filename, sep=' ', header=None, names=['Gene1', 'Gene2', 'Signal'])

all_genes = set(data.Gene1).union(set(data.Gene2))

train_data=[]
for dataset in TRAIN_DATA:
    with open(EDGES_DIR+dataset+'.edges','rb') as f:
        edges=pickle.load(f) #list of edge names tuples
        train_data+=edges
train_genes=set(list(zip(*train_data))[0]+list(zip(*train_data))[1])
train_genes =[str(gene) for gene in train_genes]

#%% functions

def split_signed_data(data, e):
    # Separate positive and negative interactions
    def signal_threshold(epsilon, p=0.5):
        return p+epsilon, p-epsilon
    minus_th, plus_th = signal_threshold(e)
    
    positive_interactions = data[data['Signal'] < plus_th]
    negative_interactions = data[data['Signal'] > minus_th]
    
    return positive_interactions, negative_interactions


def split_data_fulldf(data, e, drop_common=False, write=False):
    '''usde for wilcoxon rank sums test'''
    # Separate positive and negative interactions
    def signal_threshold(epsilon, p=0.5):
        return p+epsilon, p-epsilon
    minus_th, plus_th = signal_threshold(e)
    
    positive_interactions = data[data['Signal'] < plus_th]
    negative_interactions = data[data['Signal'] > minus_th]
    
    return pd.concat([positive_interactions,negative_interactions])

def split_data(data, e, drop_common=False, write=False, v=True):
    ''' Separate positive and negative interactions '''
    def signal_threshold(epsilon, p=0.5):
        return p+epsilon, p-epsilon
    minus_th, plus_th = signal_threshold(e)
    
    positive_interactions = data[data['Signal'] < plus_th]
    negative_interactions = data[data['Signal'] >= minus_th]
    if v:
        print(f"Number of positive interactions: {len(positive_interactions)}")
        print(f"Number of negative interactions: {len(negative_interactions)}")
        
    # Extract unique genes from positive and negative interactions
    positive_genes = set(positive_interactions['Gene1']).union(set(positive_interactions['Gene2']))
    negative_genes = set(negative_interactions['Gene1']).union(set(negative_interactions['Gene2']))
    
    common_genes=negative_genes.intersection(positive_genes)
    if v:
        print(f"Number of common genes: {len(common_genes)}")
    if drop_common:
        positive_genes = positive_genes - common_genes
        negative_genes = negative_genes - common_genes
        
    positive_genes = [str(x) for x in list(positive_genes)]
    negative_genes = [str(x) for x in  list(negative_genes)]
    
    negative_interactions = set(zip(negative_interactions['Gene1'], negative_interactions['Gene2']))
    positive_interactions = set(zip(positive_interactions['Gene1'], positive_interactions['Gene2']))

    if write:
        f=open(ENRICHMENT_OUT+'positive_genes.txt', 'w')
        f.write('\n'.join([str(x) for x in positive_genes]))
        f.close()
        f=open(ENRICHMENT_OUT+'negative_genes.txt', 'w')
        f.write('\n'.join(([str(x) for x in negative_genes])))
        f.close()

    if v:
        print(f"Number of positive genes: {len(positive_genes)}")
        print(f"Number of negative genes: {len(negative_genes)}")
        
    return positive_interactions, negative_interactions, positive_genes, negative_genes

###############################################################################
### GO and Kegg functions
################################################################################

def parse_GAF(gaf_file):
    '''
    Parse GAF file to extract useful 
    gene-to-GO mappings
    '''
    gene2go = defaultdict(set)
    go_to_gene = defaultdict(list)
    go_to_gene_name = defaultdict(list)
    id_to_name= {}
    
    
    with gzip.open(gaf_file, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row[0].startswith("!"):  # Skip comments
                gene = row[2]  # SGD gene name
                go_term = row[4]  # GO term ID
                try:
                    gene_id = str(alias_2geneid[gene])
                    gene2go[gene_id].add(go_term)
                    go_to_gene[go_term].append(gene_id)
                    go_to_gene_name[go_term].append(gene)
                    id_to_name[gene_id]=gene
                except:
                    continue
    return gene2go, go_to_gene, go_to_gene_name, id_to_name

def get_scer_go_terms(gaf_file):
    """
    Retrieves all GO terms associated with Saccharomyces cerevisiae from the GO OBO file 
    and filters them using the S. cerevisiae GAF file.

    Input:
    - gaf_file:   (str)  Path to the GAF file containing GO annotations.

    Output:
    - go_to_names: (dict) Dictionary mapping GO terms to their names.
    """
    
    url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
    response = requests.get(url)
    go_to_names = {}

    if response.status_code == 200:
        lines = response.text.split('\n')
        current_term = None
        term_name = None

        for line in lines:
            if line.startswith('[Term]'):
                current_term = None
                term_name = None
            elif line.startswith('id: GO:'):
                current_term = line.split('id: ')[1]
            elif line.startswith('name:'):
                term_name = line.split('name: ')[1]
            elif current_term and term_name:
                go_to_names[current_term] = term_name

    else:
        print("Failed to retrieve GO terms")
        return {}

    # Filter GO terms for S. cerevisiae
    valid_go_terms = set()

    with gzip.open(gaf_file, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            if line[0].startswith('!'):
                continue  # Skip headers
            organism_id = line[0]  # Column 1 = Taxon ID
            go_term = line[4]  # Column 5 = GO Term

            # Keep only terms associated with S. cerevisiae (NCBI Taxon ID: 559292)
            if organism_id == "SGD" and go_term in go_to_names:
                valid_go_terms.add(go_term)

    # Return only valid GO terms mapped to S. cerevisiae
    return {go: go_to_names[go] for go in valid_go_terms}


# Select signaling-related GO terms
def get_signal_transduction_terms():
    url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
    response = requests.get(url)
    terms = []
    go_to_names = {}
    
    if response.status_code == 200:
        lines = response.text.split('\n')
        current_term = None
        term_name = None
        term_def = None
        in_biological_process = False

        for line in lines:
            if line.startswith('[Term]'):
                current_term = None
                term_name = None
                term_def = None
                in_biological_process = False
            elif line.startswith('id: GO:'):
                current_term = line.split('id: ')[1]
            elif line.startswith('name:'):
                term_name = line.split('name: ')[1]
            elif line.startswith('namespace:'):
                if line.split('namespace: ')[1] == 'biological_process':
                    in_biological_process = True
            elif line.startswith('def:'):
                term_def = line.split('def: ')[1]
                if in_biological_process:
                    if 'signal' in term_def:
                        terms.append(current_term)
                        go_to_names[current_term]=term_name
                        current_term = None
                        continue
                    elif 'signal' in term_name:
                        terms.append(current_term)
                        go_to_names[current_term]=term_name
                        current_term = None
                    else:
                        continue
    else:
        print("Failed to retrieve GO terms")
    return terms, go_to_names

def filter_GO_dictionaries(go_to_gene, go_to_gene_name, lower, upper):
    '''
    Filter terms  by only  using terms with size size in the intreval
    [lower, upper]
    '''
    go_to_gene_filtered = {}
    go_to_gene_name_filtered = {}
        
    for go_term in go_to_gene.keys():
        if len(go_to_gene[go_term])>=lower and len(go_to_gene[go_term])<=upper:
            go_to_gene_filtered[go_term]=go_to_gene[go_term]
            go_to_gene_name_filtered[go_term]=go_to_gene_name[go_term]
    return go_to_gene_filtered, go_to_gene_name_filtered


#############################################################
#  Post processing loading saving
#################################################################

def filter_significant(data, PvalCol='adj_p_value'):
    data=data[data[PvalCol]<=0.05]
    return data.sort_values(by=PvalCol)

def saver(data, filename):
    data.to_csv(ENRICHMENT_OUT+filename+'.tsv', index=None, sep='\t')
    
def loader(filename):
    data=pd.read_csv(ENRICHMENT_OUT+filename+'.tsv', sep='\t')
    return data

############################################################################
# GENE ENRICHMENT FUNCTIONS
############################################################################

def HG_test_enrichment(target_set, go_to_gene, M, term_id_to_name=None,\
                       multiple_test_correction=True):
    
    go_enrichment = []
    
    for go_term, genes_in_term in go_to_gene.items():
        # n = Number of genes_in_term annotated with this GO term
        n = len(genes_in_term)
        # X = Observed genes_in_term in our target set (=drawn successes)
        X = sum(1 for gene in genes_in_term if gene in target_set)
        # Experiment genes (number of draws)
        N = len(target_set)
        
        p_value = hypergeom.sf(X-1, M, n, N)
        
        if term_id_to_name:
            go_enrichment.append([go_term, term_id_to_name[go_term], p_value]) 
        else:
            go_enrichment.append([go_term, go_term, p_value]) 
    
    result=pd.DataFrame(go_enrichment, columns=['GO_ID', 'name', 'P-value'])
    
    if multiple_test_correction:
        _, result['adj_p_value'], _, _ = multipletests(result['P-value'], method='bonferroni') # "fdr_bh" 'bonferroni'
    return result


def gseapy_enrichment_wrap(gene_list, gene_sets, background_genes=None):
    neg_enr = gs.enrich(gene_list=gene_list,
         gene_sets=gene_sets,
         background=background_genes, # or None or "hsapiens_gene_ensembl", or int, or text file, or a list of genes
         outdir=None,
         verbose=False)
    
    res_neg=neg_enr.results
    # add GO name:
    res_neg['name']=res_neg['Term'].apply(lambda x : go_to_name[x])
    return res_neg
############################################################################
# VISUALIZATION FUNCTIONS:
############################################################################

def build_name(string, e, drop_common=False, GO='all'):
    '''Concatenate enrichment parameter strings
    into the figure name'''
    string+='_e'+str(e)
    string+=GO
    string+='unique' if drop_common else string
    return string


#%% Step 1: Download & Parse Yeast GO Annotations (SGD GAF file)
# gaf_url = "https://current.geneontology.org/annotations/sgd.gaf.gz"
gaf_file = "sgd.gaf.gz"

# # Download file if needed
# response = requests.get(gaf_url)
# with open(gaf_file, "wb") as f:
#     f.write(response.content)
        
gene2go, go_to_gene, go_to_gene_name, id_to_name = parse_GAF(gaf_file)

#%% Get names of go terms (requires downnloading obo file)
# go_to_name=get_scer_go_terms(gaf_file)
# with open(BIO_DIR+'data'+os.sep+'go_to_name.pkl', "wb") as f:
#    pickle.dump(go_to_name, f)

with open(BIO_DIR+'data'+os.sep+'go_to_name.pkl', "rb") as f:
   go_to_name=pickle.load( f)
name_to_go={x:y for y,x in go_to_name.items()}

print(len(go_to_gene[name_to_go['nucleus']]))
#%% Filter terms  by only  using terms with given size  

# print min and max GO term size:
# maxgo=0
# mingo=999999999
# for term, genes in go_to_gene.items():
#     if len(genes)>maxgo:
#         maxgo=len(genes)
#     if len(genes)<mingo:
#         mingo=len(genes)
# print('min', mingo,'max', maxgo)

ll=10
ul=100

go_to_gene_f, go_to_gene_name_f = filter_GO_dictionaries(go_to_gene, go_to_gene_name, ll, ul)        

##
def print_term(res_neg, term_of_interest='peroxi', pvalcol='Adjusted P-value'):
    interesting_data=pd.DataFrame(columns=['name','adj_p_value'])
    for i, row in res_neg.iterrows():
        if term_of_interest in row['name']:
            interesting_data.loc[i]=[row['name'], row[pvalcol]]
    series=interesting_data.set_index('name')['adj_p_value']
    
    return series.sort_values()
#%% Gene enrichment
e=0.01
print('e',e,', lower limit',ll,', upper limit',ul)

_, _,  _, negative_genes = split_data(data, e, v=True)

# remove training genes
negative_genes = [gene for gene in negative_genes if gene not in set(train_genes)]


# Perform over representation analysis via hypergeometric test
res_neg = HG_test_enrichment(negative_genes, go_to_gene_f,\
            M=len(all_genes), term_id_to_name=go_to_name, multiple_test_correction=True)
#gseapy version:
# res_neg=gseapy_enrichment_wrap(negative_genes, go_to_gene_f, all_genes)

# print 10 best terms:
res_neg['-log10Pval']=res_neg['adj_p_value'].apply(lambda x: -np.log10(x))
print( res_neg.sort_values(by='adj_p_value')[[ 'name','GO_ID','-log10Pval']].head(10))
print( res_neg.sort_values(by='adj_p_value')[[ 'name','GO_ID','adj_p_value']].head(10))
#%% check which genes are in a given term
with open(DICT_DIR+'geneid_2symbol.pkl','rb') as f:
    geneid_2symbol=pickle.load(f)
    
current_term='protein import into peroxisome matrix, docking'

set([ geneid_2symbol[int(x)] for x in go_to_gene[name_to_go[current_term]]])

