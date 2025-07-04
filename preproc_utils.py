# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 13:51:18 2021

@author: L-F-S
"""
import pandas as pd
import collections
import networkx
import itertools
import os
import anndata
from datetime import date
from collections import defaultdict

###############################################################################
# READ, FILTER AND PREPARE DATA
###############################################################################
 
def read_alias2geneid(GENE_INFO_DIRECTORY, species, alias_column='LocusTag', final_column='GeneID'):
    '''LocusTag, GeneID, Symbol'''
    geneinfo_filename = GENE_INFO_DIRECTORY+ species + ".gene_info"
    geneinfo = pd.read_table(geneinfo_filename, usecols=[final_column,alias_column])
    print(alias_column)
    if len(geneinfo.columns) == 1:
        geneinfo["GeneID_str"] = geneinfo[final_column].astype(str)
    if not (alias_column == 'Synonyms' or alias_column == 'dbXrefs'):
       return {series[alias_column]: series[final_column] for ind, series in geneinfo.iterrows()}
    elif alias_column == 'Synonyms':
       geneinfo = geneinfo[geneinfo['Synonyms']!= '-']
       temp = {str(series[alias_column]) :series[final_column] for ind, series in geneinfo.iterrows()}
       return { key: value for keys, value in temp.items() for key in keys.split('|') }
    elif alias_column == 'dbXrefs':
        geneinfo = geneinfo[geneinfo['dbXrefs']!= '-']
        temp = {str(series[alias_column]) :series[final_column] for ind, series in geneinfo.iterrows()}
        return { keys.split('Ensembl:')[-1]: value for keys, value in temp.items()}

    
def readname2geneid(GENE_INFO_DIRECTORY, SPECIES):
    synonyms_2geneid = read_alias2geneid(GENE_INFO_DIRECTORY, SPECIES, 'Synonyms','GeneID')
    symbol_2geneid=read_alias2geneid(GENE_INFO_DIRECTORY, SPECIES, 'Symbol','GeneID')
    alias_2geneid=read_alias2geneid(GENE_INFO_DIRECTORY, SPECIES)
    geneid_2geneid=read_alias2geneid(GENE_INFO_DIRECTORY, SPECIES,'GeneID','GeneID')
    ensembl_2geneid=read_alias2geneid(GENE_INFO_DIRECTORY, SPECIES,'dbXrefs','GeneID')
    strgeneid_2geneid = {str(key):value for (key,value) in geneid_2geneid.items()}
    
    symbol_2geneid.update(synonyms_2geneid)
    symbol_2geneid.update(alias_2geneid)
    symbol_2geneid.update(geneid_2geneid)
    symbol_2geneid.update(strgeneid_2geneid)
    symbol_2geneid.update(ensembl_2geneid)
    return symbol_2geneid
    # names_2geneid=synonyms_2geneid | symbol_2geneid # dictionary concatenation operand. works with python 3.9+
    # return names_2geneid | alias_2geneid 
def get_PSP_data(filename,  species, alias_2geneid):
    data = pd.read_csv(filename, sep='\t', skiprows=[0,1,2], header=None, index_col=(1,4), encoding_errors='ignore')
    if not species =='H_sapiens':
        raise ValueError('data only avaliable for human')
    data=data[data[8]=='human']
    data=data[data[3]=='human']
    data.rename_axis(index={1:0,4:1}, inplace=True)
    labels = pd.Series(data=0, index=data.index)
    labels.name=2
    labels = translate_multi_ind(labels, alias_2geneid)
    #'Removing duplicates'
    labels=remove_dupes(labels)
    weights = pd.Series(data=0.8, index=labels.index)
    weights.name=2
    return labels, weights

def get_kegg_data(filename,  species, alias_2geneid):
    label=pd.read_csv(filename, sep= "\t", header=None, index_col=(0,1)) 
    label=label[label[3]!='indirect effect']
    label=label[label[2]!='indirect effect']
    label=label[label[2]!='state change']
    label=label[label[2]!='repression'] #transcription factors
    label[2].replace({"activation":0,"inhibition":1, 'binding/association':0,\
                      'expression':0, 'dissociation':1, 'phosphorylation': 0, \
                          'ubiquitination':1}, inplace=True)
        
    label.drop(axis=1, labels=3, inplace=True)
    label = pd.Series(data=list(label[2]), index=label.index)
    label.name = 2 
        #'translating indexes:'
    label = translate_multi_ind(label, alias_2geneid)
    #'Removing duplicates'
    label=remove_dupes(label)
    edge_weights=pd.Series(data= 0.8, index=label.index)
    edge_weights.name=2
    return label, edge_weights


def get_data(filename,  species, alias_2geneid):
    label=pd.read_csv(filename, sep= "\t", header=0 ) 
    label_index=list(zip(label[label.columns[0]], label[label.columns[1]]))
    label = pd.Series(data=list(label[label.columns[2]]), index=label_index)
    label.name = 2 
    #'Removing duplicates'
    label=remove_dupes(label)
    edge_weights=pd.Series(data= 0.8, index=label.index)
    edge_weights.name=2
    return label, edge_weights


def get_ubiquitin_data(filename,  species, alias_2geneid):
    print("\n>>>>>>> getting ubiquitin dataset:")

    ubiq_labels = pd.read_csv(filename, sep="\t", usecols=['E3_ID','SUB_ID','E3_ORGANISM','SUB_ORGANISM'])
    if species == 'S_cerevisiae':
        sp_label = 'Saccharomyces cerevisiae'
    if species == 'H_sapiens':
        sp_label = 'Homo sapiens'
    ubiq_labels = ubiq_labels[ubiq_labels['E3_ORGANISM'].str.startswith(sp_label)]
    ubiq_labels = ubiq_labels[ubiq_labels['SUB_ORGANISM'].str.startswith(sp_label)].reset_index(drop=True)
    print(ubiq_labels.shape, 'initial interactions')
    ubiq_labels = ubiq_labels[ubiq_labels['E3_ID']!='-']
    def alias_of(dic, x):
        return  int(alias_2geneid[x]) if x in dic.keys() else None
    ubiq_labels['SUB_ID'] = ubiq_labels['SUB_ID'].apply(lambda x : alias_of(alias_2geneid, x.split('_')[0]))
    ubiq_labels['E3_ID'] = ubiq_labels['E3_ID'].apply(lambda x : alias_of(alias_2geneid, x.split('_')[0]))
    ubiq_labels.dropna(inplace=True)
    subid = [ int (x) for i,x in ubiq_labels['SUB_ID'].iteritems()]
    e3 = [int (x) for i,x in ubiq_labels['E3_ID'].iteritems()]
    ubiq_labels.index=list(zip(e3, subid))
    ubiq_labels.drop(columns=['E3_ID','SUB_ID','E3_ORGANISM','SUB_ORGANISM'], inplace=True)
    ubiq_labels[2]=1.0
    ubiq_labels=remove_dupes(ubiq_labels)
    ubiq_edge_weights = pd.Series(0.8, index=ubiq_labels.index)
    ubiq_edge_weights.name=2
    print(ubiq_labels.shape, 'final interactions')
    return ubiq_labels[2], ubiq_edge_weights

def get_protein_complexes_data(filename, alias_2geneid):
    '''
    Load dataset of protein complexes and use them as undirected positive interacctions.
    every undirected interaction is represented as two directed interactions in opposite directions
    '''
    print("\n>>>>>>> getting protein complex dataset:")
    complexes_label = pd.read_csv(filename, sep="\t",)
   
    complexesgroupby = complexes_label.groupby('Complex').apply(
        lambda x : list(itertools.combinations(x['Name'], 2)))
    complexes = [(alias_2geneid[x[0]], alias_2geneid[x[1]]) for sublist in list(complexesgroupby) for x in sublist]
    complexes2 = [(alias_2geneid[x[1]], alias_2geneid[x[0]]) for sublist in list(complexesgroupby) for x in sublist]
    complexes += complexes2
    complexes_label = pd.Series(0.0, index = complexes)
    complexes_weights = pd.Series(0.8, index = complexes)
    complexes_label.name = 2 
    complexes_weights.name = 2
    #('Removing duplicates')    
    complexes_label=remove_dupes(complexes_label)
    complexes_weights=complexes_weights.loc[complexes_label.index]
    return complexes_label, complexes_weights

def get_yeast_kpi(filename, species, alias_2geneid):
    label=pd.read_csv(filename, sep="\t",header=None, skiprows=1, index_col=(0,1)).dropna()
    edge_weights=pd.Series(0.6, index=label.index)
    label = pd.Series(data=list(label[2]), index=label.index)
    label.name = 2 
    edge_weights.name=2
    label = translate_multi_ind(label, alias_2geneid)
    edge_weights = translate_multi_ind(edge_weights, alias_2geneid)
    #'Removing duplicates'
    label=remove_dupes(label)
    edge_weights=edge_weights.loc[label.index]
    return label, edge_weights


def preprocess_yeast_signed_datasets(data_dir, alias_2geneid,datasets, SPECIES): #TODO lost kegg_patkar somewjere
    labels_of = {}
    weights_of = {}
    for dataset_name, filename in datasets.items():
        if 'kegg' in dataset_name:
            labels_of[dataset_name], weights_of[dataset_name] = kegg_Kpi_labels, kegg_Kpi_edge_weights = get_kegg_data(filename, SPECIES, alias_2geneid) 
        if dataset_name == 'kpi':
            labels_of[dataset_name], weights_of[dataset_name] = kegg_Kpi_labels, kegg_Kpi_edge_weights = get_yeast_kpi(filename, SPECIES, alias_2geneid) 
        if dataset_name == 'p_complex':
            labels_of[dataset_name], weights_of[dataset_name] = pcomp_labels, pcomp_edge_weights = get_protein_complexes_data(filename,alias_2geneid)
        if dataset_name == 'ubiq':
            labels_of[dataset_name], weights_of[dataset_name] = ubiq_labels, ubiq_edge_weights = get_ubiquitin_data(filename, SPECIES, alias_2geneid)
    return labels_of, weights_of

def preprocess_human_signed_datasets(datanames,species,  alias_2geneid): #TODO change from load to preprocess, and make a load to load preprocessed data
    '''
    INPUTS:
            datanames: dict of {dataname: absolute_path] strings
            species: str species name
            alias_2geneid: dictionary of geneID mappings
    OUTPUT:
       pd.Series,
       pd.Series
    '''
    signed_datasets = {}
    signed_datasets_edge_weights = {}
    for dataset, filename in datanames.items():
        if dataset == 'ubinet2':
           labels, edge_weights = get_ubiquitin_data(filename,  species, alias_2geneid)
        elif dataset == 'PSP':
            labels, edge_weights = get_PSP_data(filename,  species, alias_2geneid)
        elif dataset == 'kegg':
            labels, edge_weights = get_kegg_data(filename,  species, alias_2geneid)
        else:
            labels, edge_weights = get_data(filename,  species, alias_2geneid)
            
        signed_datasets[dataset] = labels
        signed_datasets_edge_weights[dataset] = edge_weights
    return signed_datasets, signed_datasets_edge_weights

def preprocess_signed_datasets(data_dir, datanames,species,  alias_2geneid):
    if species == 'H_sapiens':
        signed_datasets, signed_datasets_edge_weights = preprocess_human_signed_datasets(datanames, species, alias_2geneid)
    if species == 'S_cerevisiae':
        signed_datasets, signed_datasets_edge_weights = preprocess_yeast_signed_datasets(data_dir, alias_2geneid,datanames, species)
    return signed_datasets, signed_datasets_edge_weights

def get_targets(df):
    plus_targets_of_deletion=collections.defaultdict(list)
    minus_targets_of_deletion=collections.defaultdict(list)

    for i, series in df.iterrows():
        if float(series) >0:
            plus_targets_of_deletion[i[0]].append(i[1])
        else:
            minus_targets_of_deletion[i[0]].append(i[1])

    return plus_targets_of_deletion, minus_targets_of_deletion


def translate_multi_ind(data, alias_2geneid, v=True):
    '''data can be a pd.Series or pd.DataFrame, with tuple MultiIndex'''
    index = []
    not_found=[]
    data=data.reset_index()
    
    #11/08 addition
    for i, values in data.iterrows():
        ind=(values[0],values[1])
        try:
            index.append((alias_2geneid[ind[0]], alias_2geneid[ind[1]]))
        except:
            not_found.append((ind[0], ind[1]))
            data.drop(i, axis=0, inplace=True)
    # for ind in data.index: older code (not working after 11/08 ith some multi indexes)
    #     try:
    #         index.append((alias_2geneid[ind[0]], alias_2geneid[ind[1]]))
    #     except:
    #         not_found.append((ind[0], ind[1]))
    #         data.drop([ind], inplace=True)
    if v:
        print(len(not_found), 'indexes not found in dictionary')
    # datanew=pd.Series(data[2]) # 11/08 added
    # datanew.index=data[[0,1]]
    data=pd.Series(data[2])
    data.index=index
    return data

def translate_axes(perturbations_map, alias_2geneid, axis='all'):
    """
    translates column names and row names into entrez geneIDs
    input:
        perturbations_map: pd.DataFrame
        alias_2geneid: dictionary
        axis: str ['row', 'col', 'all']
    output:
        perturbations_map with translated intexes
    """
    
    if axis=='row' or axis=='all':
        index_map={}
        nr=0
        rows_to_drop=[]
        for oldname in perturbations_map.index:
            try:
                index_map[oldname]=alias_2geneid[oldname]
            except:
                nr+=1
                rows_to_drop.append(oldname)
        print('removing',nr,'target transcripts not fonud in gene_info dictionary')
        perturbations_map.drop(rows_to_drop, inplace=True)
        perturbations_map.rename(index=index_map, inplace=True)
    
    if axis=='col' or axis=='all':
        column_map = {}
        nc=0
        cols_to_drop=[]
        for oldname in list(perturbations_map.columns):
            try:
                column_map[oldname] = alias_2geneid[oldname]
            except:
                nc+=1
                cols_to_drop.append(oldname)
        print('removing',nc,'knockdown/perturbed genes not fonud in gene_info dictionary')
        perturbations_map.drop(cols_to_drop, axis=1, inplace=True)
        perturbations_map.rename(columns=column_map, inplace=True)
    return perturbations_map

def load_training_data(outdir, datanames, species):
    signed_datasets = {}
    signed_datasets_edge_weights = {}
    for dataname in datanames:
        labels = pd.read_csv(outdir + dataname+".lbl.tsv", header=None, sep='\t', index_col=[0,1]).squeeze()
        weights = pd.read_csv(outdir + dataname+".w8.tsv", header=None, sep='\t',index_col=[0,1]).squeeze() #squeeze argument to turn one column dataframe into series
        signed_datasets[dataname] = labels
        signed_datasets_edge_weights[dataname] = weights
    return signed_datasets, signed_datasets_edge_weights


def get_perturbations_map(data_dir, alias_2geneid,species, filename, translate=True):
    """
    output:
        perturbations_map: pd.DataFrame column names: entrez geneIDs of 
            knockout/downregulated/mutated genes  
            index names: entrez geneID of affected genes.
            column: expression/fold change for that mutation
            row: expression/fold change of given gene across all mutations
    """
    print(">>>>>> getting knockout pairs set:")
    if species == "S_cerevisiae":
        print(filename)
        perturbations_map=pd.read_csv(data_dir+filename,sep='\t',header=0, skiprows=[0,1,3,4,5,6], index_col=1) 
        perturbations_map.drop(perturbations_map.columns[range(6)],axis=1,inplace=True)
        print('initial experiments before translation: ', perturbations_map.shape[1])
        print('initial affected genes before translation: ', perturbations_map.shape[0])
    
    if species == "H_sapiens":
        perturbations_map=pd.read_csv(data_dir+filename, index_col=0, \
                          skiprows=[1,2],  header=0) #from fig 2a
        #columns= gene transcript
        #rows = gene name
        perturbations_map.drop(columns=perturbations_map.columns[0], inplace=True)
        perturbations_map.rename(columns= lambda x : x.split('_')[3], inplace=True)
        seen_id=defaultdict(list)
        seengene=[]
        dupedid=defaultdict(list)
        dupedgene=[]
        for i in perturbations_map.columns:
            if i in seengene:
                dupedgene.append(i)
                try:
                    dupedid[alias_2geneid[i]].append( i)
                except:
                    continue
            else:
                seengene.append(i)
                #fin qui non ci sn duplicati
            try:
                if alias_2geneid[i] in seen_id.keys():
                    dupedid[alias_2geneid[i]].append(i)
                seen_id[alias_2geneid[i]].append(i)
            except:
                continue
        print([(i,seen_id[i]) for i in dupedid.keys()]) # ci sono alcuni ID che sono con piu nomi
        #nell index, e nel column ce ne sono 3 ripetuti lol

        translate_axes(perturbations_map, alias_2geneid) 
        perturbations_map=perturbations_map.loc[:,~perturbations_map.columns.duplicated()] # drop cols
        perturbations_map = perturbations_map[~perturbations_map.index.duplicated(keep=False)] #drop rows
                
    if translate==True:
        perturbations_map=translate_axes(perturbations_map, alias_2geneid) 
    print('Perturbation experiments: ', perturbations_map.shape[1])
    print('Affected genes: ', perturbations_map.shape[0])

    return perturbations_map 

def extract_knockotut_effect_pairs_from_data(perturbation_map, genes, threshold=1.7):
    # See original data refs. for threshold value
# todo, refactor: this must be faster cos with human it's super slow. todo. threshold for human??
    plus_targets_of_deletion={}
    minus_targets_of_deletion={}
    perturb_map = perturbation_map.copy()
    
    print('removing genes not present in base network:')
    perturb_map.drop(columns=set(perturb_map.columns).difference(set(genes)), inplace=True)
    perturb_map.drop(index=set(perturb_map.index).difference(set(genes)), inplace=True)
    print(perturb_map.shape)
    print('filtering out expression threshold:')
    for source  in perturb_map.columns:
        list_of_plus_targets=perturb_map[source][(perturb_map[source]>threshold)].index
        list_of_minus_targets=perturb_map[source][(perturb_map[source]<-threshold)].index # the gene names of genes with threshold value for that particular experiment (source)
        if (len(list_of_plus_targets)>0) and (len(list_of_minus_targets)>0):
            plus_targets_of_deletion[source]=list(list_of_plus_targets)
            minus_targets_of_deletion[source]=list(list_of_minus_targets)
    return plus_targets_of_deletion, minus_targets_of_deletion



def read_network_from_file(network_dir, species, filename):
    """
    input:
        network_dir: str
        species: str ['S_cerevisiae', 'H_sapiens']
    output:
        pd.DataFrame: index: tuple, columns: confidence, directed
    """
    if len(filename)==0:
        filename=network_dir+os.sep+species+'.net'
    else:
        filename=network_dir+os.sep+filename
    return pd.read_csv(filename,sep="\t", index_col = [0,1],usecols=([0,1,2,3]), header=None)

        
def graph_from_dataframe(SPECIES_DATA_DIR, SPECIES, net_type="undir", filename=''):
    """
    input: 
        dataframe: pd.DataFrame
        net_type: str ['dir', 'undir'] current version only uses undirected networks
    output:
        networkx.graph
    v3: treating the input as undirected graph. For the directed case, see v2
    """
    dataframe=read_network_from_file(SPECIES_DATA_DIR, SPECIES, filename)
    if net_type=="undir":
        return networkx.from_pandas_edgelist(dataframe.reset_index().rename(columns={2:'weight'}), 0,1,'weight')

    elif net_type=="dir":
        graph = networkx.DiGraph()
        for (source, target, confidence, directed) in dataframe.reset_index().itertuples(False):
            graph.add_edge(source, target, weight=confidence)
            if directed == 0:
                graph.add_edge(target, source, weight=confidence)
        return graph

    else:
        raise ValueError("Incorrect net type: "+str(net_type))

def converttod2ddirected(network, threshold):
    network_dictionary = network.to_dict()
    new_network= []
    assigned_edges=set()
    for i, (edge, fwdprob) in enumerate(network_dictionary[3].items()):
        flipped_edge = (edge[1], edge[0])
        revprob =  network_dictionary[3][flipped_edge]    
        if not edge in assigned_edges:
            # choose thresholding method 
            if threshold == 0.1:
                if  (revprob!=0) and (fwdprob/revprob > 1+threshold):
                    new_network.append([edge[0], edge[1], network_dictionary[2][edge], 1])
                    assigned_edges.add(edge)
                    assigned_edges.add(flipped_edge)
                elif (fwdprob!=0) and (revprob/fwdprob > 1+threshold):
                    new_network.append([edge[1], edge[0], network_dictionary[2][flipped_edge], 1])
                    assigned_edges.add(edge)                
                    assigned_edges.add(flipped_edge)
                else:
                    new_network.append([edge[0], edge[1], network_dictionary[2][edge], 0])
                    assigned_edges.add(edge)
                    assigned_edges.add(flipped_edge)
                        
                        
            else:
                if (fwdprob > threshold) and (revprob<0.5):
                    new_network.append([edge[0], edge[1], network_dictionary[2][edge], 1])
                    assigned_edges.add(edge)
                    assigned_edges.add(flipped_edge)
                elif (revprob > threshold) and (fwdprob<0.5):
                    new_network.append([edge[1], edge[0], network_dictionary[2][flipped_edge], 1])
                    assigned_edges.add(edge)
                    assigned_edges.add(flipped_edge)
                else:
                    new_network.append([edge[0], edge[1], network_dictionary[2][edge], 0]) # and the other remains as 0 
                    assigned_edges.add(edge)
                    assigned_edges.add(flipped_edge)
    new_network= pd.DataFrame(new_network)
    new_network.index=pd.MultiIndex.from_frame(new_network[[0,1]])
    new_network.drop(columns=[0,1], inplace=True)
    return new_network

    
def add_edges_from_labels(graph, edge_weights, flag='d', v=True):
    '''
    Adds missing edges to network from labels. Edges can be either directed or undirected.
    If edges are added as directed, their opposite direction counterparts will be removed
    if they exist in the base network.
    Inputs:
        graph: nx.Graph() or nx.DiGraph(). The base interaction network
        edge_weights: pd.Series. indexes are a tuple of protein ids, and the value is the edge weight
        flag: str: ['d','u'] treat edges as directed ('d') or undirected ('u')
    Output:
        directedgraph: nx.DiGraph()
    '''
    directedgraph=graph.to_directed() # if already directed (nx.DiGraph object), returns a (deep) copy   
    edges=directedgraph.edges()
    n=0
    c=0
    a=0
    u=0
    w=[]
    for edge in edge_weights.index:
        if flag == 'd':
            if edge in edges:
                if (edge[1],edge[0]) in edges: #both directions are present. let's remove the wrong one!
                    if not directedgraph[edge[1]][edge[0]]['weight']==0:
                        directedgraph[edge[1]][edge[0]]['weight']=0
                        u+=1 # oriented previously unoriented edge
                    else:
                        c+=1 # only present in correct orientation
                else: #only one present. correct orientation
                    c+=1
            else:
                if not (edge[1],edge[0]) in edges: #add 'edge' as a directed edge to this network
                    directedgraph.add_edge(edge[0],edge[1], weight=edge_weights[edge])
                    a+=1
                else:  
                    if not directedgraph[edge[1]][edge[0]]['weight'] == 0: #oriented edges in basenet with claimed orientation opposite to known training edges' orientation
                        n+=1
                    directedgraph[edge[1]][edge[0]]['weight'] == 0
                    directedgraph.add_edge(edge[0],edge[1], weight=edge_weights[edge])
        elif flag=='u':
            if not (edge[0],edge[1]) in edges:
                a+=1
                directedgraph.add_edge(edge[0],edge[1], weight=0.8)
            else:
                c+=1
                w.append(directedgraph[edge[0]][edge[1]]['weight'])
                directedgraph[edge[0]][edge[1]]['weight']=max(directedgraph[edge[0]][edge[1]]['weight'], 0.8) #udpate weight of current edge to 0.8 if lower than that,
            # directedgraph.add_edge(edge[1],edge[0], weight=0.8)
    if v:
        print(n, 'edges in base network with claimed direction opposite to known training data edges orientation')
        print(c, 'edges in base network with claimed orientation equal to known training edges data orientation')
        print(a, 'directed edges added to base network')
        print(u, 'edges in base network withut orientation which have been oriented from training data')
        print('total:',a+n+c+u)
    return directedgraph

def remove_training_edges_if_not_in_base_net(label, edges_of_base_net):
    for edge in label.index:
        if not edge in edges_of_base_net:
            label.drop(edge, inplace=True)          
    return label


def remove_dupes(data): #TODO dont use this actually removes both copies of dupes
    seen = {}
    dupes = []
    
    for x in data.index:
        if x not in seen:
            seen[x] = 1
        else:
            if seen[x] == 1:
                dupes.append(x)
            seen[x] += 1
    return data.drop(dupes)
def write2(OUTDIR, labels_of, edge_weights_of):
    '''labels_of and edge_weights_of:
                dictionary of {'datasetname' : pd.Series}'''
    for  dataname, labels in labels_of.items():
        labels.to_csv(OUTDIR+dataname+'.lbl.tsv', sep='\t', header=None)
        edge_weights_of[dataname].to_csv(OUTDIR+dataname+'.w8.tsv', sep='\t', header=None)

def write(OUTDIR, labels_of, edge_weights_of):
    '''labels_of and edge_weights_of:
                dictionary of {'datasetname' : pd.Series}'''
    for  dataname, labels in labels_of.items():
        labels=labels.reset_index()
        labels[[0,1]]=pd.DataFrame(labels['index'].tolist(), index=labels.index)  
        labels.drop(columns='index')
        labels=labels[[0,1,2]]
        labels.to_csv(OUTDIR+dataname+'.lbl.tsv', sep='\t', header=None, index=False)
        
        weight = edge_weights_of[dataname]
        weight=weight.reset_index()
        weight[[0,1]]=pd.DataFrame(weight['index'].tolist(), index=weight.index)  
        weight.drop(columns='index')
        weight=weight[[0,1,2]]
        weight.to_csv(OUTDIR+dataname+'.w8.tsv', sep='\t', header=None, index=False)

        
