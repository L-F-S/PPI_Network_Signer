# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 17:48:16 2025

@author: los4
"""

#%%% explore peroxisome 

import sys
import os
import pandas as pd
import networkx as nx
import numpy as np
import pickle
import matplotlib.pyplot as plt
import  matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
#%%
DRIVE_PATH='home\dir'

sys.path.append(DRIVE_PATH+'network_signing'+os.sep)
from glob_vars import SPECIES, TRAIN_DATA, PERT_MAP, HOME_DIR, LBL_DIR,\
    EDGES_DIR, FT_DIR, SIGNAL_DIR, MOD_DIR,ENRICHMENT_OUT, DICT_FILE, DICT_DIR,\
    BIO_DIR, BIO_OUT
sys.path.append(BIO_DIR+os.sep+'modules')
from functional_enrichment_5_gene_based_enrichment import gene2go, data, all_genes, go_to_name, alias_2geneid,\
    go_to_gene, go_to_gene_name, split_data, train_genes, HG_test_enrichment
with open(DICT_DIR+'geneid_2symbol.pkl','rb') as f:
    geneid_2symbol=pickle.load(f)

#%%

net = nx.from_pandas_edgelist(data.reset_index(), "Gene1", "Gene2", ["Signal"])
genes=list(net.nodes)
#%%
def build_adjacent_net(PPI_net, genes):
    '''Returns subnetwork of all adjacent genes to list of genes'''
    dfs=[]
    all_edges_of_net=[]
    for gene in genes:
        gene=alias_2geneid[gene]
        print(gene)
        signals_for_gene = []
        edges_for_gene = []
        for neighbor in PPI_net[gene]:
            signals_for_gene.append(PPI_net[gene][neighbor]['Signal'])
            edges_for_gene.append((gene, neighbor))
            all_edges_of_net.append((gene, neighbor))
        print(len(edges_for_gene))
    
        # condition of high signal
        adjacent_net=PPI_net.edge_subgraph(edges_for_gene).copy()
        
        # back to df
        df_sub = pd.DataFrame(adjacent_net.edges(data=True))
        df_sub.columns = ["Gene1", "Gene2", "Signal"]
        df_sub["Signal"] = df_sub["Signal"].apply(lambda x: x["Signal"])
        dfs.append(df_sub)
    PPI_df=pd.concat(dfs)
    graph=PPI_net.edge_subgraph(all_edges_of_net).copy()
    return PPI_df, graph
    
def get_edge_attributes(signal, e=0.01):
    """Determine edge color and width based on signal strength."""
    color = 'red' if signal > 0.5+e else 'darkblue'
    width = signal * 5  # Scale edge width for visibility
    return color, width

def draw_SIGNAL_network(G, genes_list, e=0.01, figname='peroxisome_SIGNAL_net', savefig=False):
    """Draw the network with edge colors and thickness based on 'Signal'.
    Edges must have a 'Signal' attribute 0<Signal<1"""
    
    
    def init_pos(genes_list):
        pos={}
        if len(genes_list)==2:
            pos = {
            genes_list[0]: (-1, 0),  # Center of cluster 1
            genes_list[1]: (1, 0), # Center of cluster 2
            }
        elif len(genes_list)==3:
            pos = {
            genes_list[0] : (-1, 0),  # Center of cluster 1
            genes_list[1] : (1, 0), # Center of cluster 2
            genes_list[2] : (0, 1)   # Center of cluster 3
            }
        else:
            print('implement pos dictionary for longer lists')
            return None
        return pos

    pos = init_pos(genes_list)
    
    if not pos:
        pos = nx.fruchterman_reingold_layout(G)  # Compute layout for visualization
    else:
        pos.update(nx.fruchterman_reingold_layout(G, pos=pos, fixed=pos.keys() , k=0.3))
    
    edges = G.edges(data=True)
    edge_colors = []
    edge_widths = []
    
    for u, v, edge_data in edges:
        color, width = get_edge_attributes(edge_data['Signal'], e)
        edge_colors.append(color)
        edge_widths.append(width)
        

    
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, \
            edge_color=edge_colors,edgecolors='black',\
            width=edge_widths, node_color='lightblue', node_size=600,\
                linewidths=1, font_size=7)
    if not savefig:
        plt.show()
    else:
        plt.savefig(BIO_OUT+'SIGNAL_graphs'+os.sep+figname, bbox_inches='tight', format='pdf')
#%%
subnet_name = 'peroxisome_PTS_receptors_adjacent' # Mitochondtrion_receptors_adjacent
genes_list=['PEX5','PEX7']   #  ['TOM20', 'TOM22'] # 
e=0.00

df_subnet, subnet = build_adjacent_net(net, genes_list)
# init single key dictionary as input for enrichment function
subnet_dict = {subnet_name: [str(gene) for gene in list(subnet.nodes)]}
#%% Check for enrichment of negative genes in sub network 

_, _,  _, negative_genes = split_data(data, e, v=True)

negative_genes = [gene for gene in negative_genes if gene not in set(train_genes)]


# Perform over representation analysis via hypergeometric test
res_neg = HG_test_enrichment(negative_genes, subnet_dict,\
            M=len(all_genes), multiple_test_correction=True)
print(res_neg)
#%% Plot subnetwork

_, subnet = build_adjacent_net(net, genes_list)

graph_with_gene_names = nx.relabel_nodes(subnet, lambda x: geneid_2symbol[x], copy=True)

draw_SIGNAL_network(graph_with_gene_names, genes_list, figname='mitochondrion_receptors_SIGNAL_net', e=e, savefig=False)


