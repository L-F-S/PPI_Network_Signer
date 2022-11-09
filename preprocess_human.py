# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""

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
SPECIES='H_sapiens'
DATE=str(date.today.strftime("%d/%m/%Y"))
HOMEDIR='G:'+os.sep+'Il Mio Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep
HOMEDIR='G:'+os.sep+'My Drive'+os.sep+'SECRET-ITN'+os.sep+'Projects'+os.sep
DATADIR=HOMEDIR+os.sep+'Data'+os.sep

def read_alias2geneid(GENE_INFO_DIRECTORY, species, alias_column='LocusTag', final_column='GeneID'):
    '''LocusTag, GeneID, Symbol'''
    geneinfo_filename = GENE_INFO_DIRECTORY+ species + ".gene_info"
    geneinfo = pd.read_table(geneinfo_filename, usecols=[final_column,alias_column])
    
    if len(geneinfo.columns) == 1:
        geneinfo["GeneID_str"] = geneinfo[final_column].astype(str)
    if not alias_column == 'Synonyms':
       return {series[alias_column]: series[final_column] for ind, series in geneinfo.iterrows()}
    else:
       geneinfo = geneinfo[geneinfo['Synonyms']!= '-']
       temp = {str(series[alias_column]) :series[final_column] for ind, series in geneinfo.iterrows()}
       return { key: value for keys, value in temp.items() for key in keys.split('|') }

def readname2geneid(GENE_INFO_DIRECTORY, SPECIES):
    synonyms_2geneid = read_alias2geneid(GENE_INFO_DIRECTORY, SPECIES, 'Synonyms','GeneID')
    symbol_2geneid=read_alias2geneid(GENE_INFO_DIRECTORY, SPECIES, 'Symbol','GeneID')
    alias_2geneid=read_alias2geneid(GENE_INFO_DIRECTORY, SPECIES)
    
    symbol_2geneid.update(synonyms_2geneid)
    symbol_2geneid.update(alias_2geneid)
    return symbol_2geneid

alias_2geneid =  readname2geneid(DATADIR, SPECIES)

network=pd.read_csv(DATADIR +SPECIES+'.net',sep="\t", header=None)
graph = nx.from_pandas_edgelist(network.reset_index(), 0, 1, 2)
