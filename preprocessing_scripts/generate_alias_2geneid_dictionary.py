# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: L-F-S

Builds a dictionary of gene names and gene id for the selected organism
from NCBI's GENE_INFO database.

usage:

For H sapiens set:

SPECIES='Homo_sapiens'

to download gene_info file set:
    
download=True
"""
import pickle
import os
import pandas as pd
import urllib.request
# os.chdir('../')
# print(os.listdir())
from glob_vars import SPECIES, DICT_DIR

###############################################################################
# Inputs
###############################################################################
print(SPECIES)
if SPECIES == "S_cerevisiae":
    NCBI_ftp_path = 'https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Fungi/Saccharomyces_cerevisiae'

if SPECIES =='H_sapiens':
    NCBI_ftp_path='https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens'



# set to True to download NCBI's gene_info file
download=True

f=open(DICT_DIR+'alias_2geneid.pkl','rb')
f.close()
#%% 

###############################################################################
# Download gene_info file from NCBI
###############################################################################


def download_gene_info(NCBI_ftp_path, output_path):
    print('downloading gene_info file')
    urllib.request.urlretrieve(NCBI_ftp_path+'.gene_info.gz', output_path+SPECIES+'.gene_info.gz')
    print('done')
    return

if download:
    download_gene_info(NCBI_ftp_path, DICT_DIR)


#%% Build alias_2geneid dictionary

def read_alias2geneid(GENE_INFO_DIRECTORY, species, alias_column='LocusTag',\
                      final_column='GeneID'):
    '''LocusTag, GeneID, Symbol'''
    print('adding', alias_column)
    geneinfo_filename = GENE_INFO_DIRECTORY+ species + ".gene_info.gz"
    geneinfo = pd.read_table(geneinfo_filename, \
                             usecols=[final_column,alias_column])
    
    if len(geneinfo.columns) == 1:
        geneinfo["GeneID_str"] = geneinfo[final_column].astype(str)
    if not alias_column == 'Synonyms':
       return {series[alias_column]: series[final_column]\
               for ind, series in geneinfo.iterrows()}
    else:
       geneinfo = geneinfo[geneinfo['Synonyms']!= '-']
       temp = {str(series[alias_column]) :series[final_column]\
               for ind, series in geneinfo.iterrows()}
       return { key: value for keys, value in temp.items()\
               for key in keys.split('|') }


# Construct alias_2geineid dictionary with all possible aliases
    
def readname2geneid(GENE_INFO_DIRECTORY, SPECIES):
    print('reading Gene_info file')
    synonyms_2geneid = read_alias2geneid(GENE_INFO_DIRECTORY, SPECIES,\
                                         'Synonyms','GeneID')
    symbol_2geneid=read_alias2geneid(GENE_INFO_DIRECTORY, SPECIES,\
                                     'Symbol','GeneID')
    alias_2geneid=read_alias2geneid(GENE_INFO_DIRECTORY, SPECIES)
    geneid_2geneid=read_alias2geneid(GENE_INFO_DIRECTORY, SPECIES,\
                                     'GeneID','GeneID')
    strgeneid_2geneid = {str(key):value\
                         for (key,value) in geneid_2geneid.items()}
    
    symbol_2geneid.update(synonyms_2geneid)
    symbol_2geneid.update(alias_2geneid)
    symbol_2geneid.update(geneid_2geneid)
    symbol_2geneid.update(strgeneid_2geneid)
    return symbol_2geneid


#%% read all gene_info columns
alias_2geneid =  readname2geneid(DICT_DIR, SPECIES)

#%% Write pickled dictionary
filename = DICT_DIR+'alias_2geneid.pkl'
with open(filename, 'wb') as f:
    pickle.dump(alias_2geneid, f)

#%% Optional: Write tsv file
# def write_tsv(alias_dictionary, tsv_directory):
#     print('writing tsv file')
#     alias_2geneid_df=\
#         pd.Series(alias_dictionary, name='gene_id').to_frame().reset_index().\
#             rename(columns={'index':'gene_name'})
#     alias_2geneid_df.to_csv(tsv_directory+'test_alias_2geneid.tsv',\
#                             sep='\t',index=None,header=True)
#     return

# write_tsv(alias_2geneid, DICT_DIR)
#%% To load dictionary to python,simply run:
# with open(filename, 'rb') as f:
#     alias_2geneid=pickle.load(f)   

