# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 13:51:18 2021

@author: los4
"""
import pandas as pd
import collections
import networkx
import itertools
import os
import anndata

###############################################################################
# READ, FILTER AND PREPARE DATA
###############################################################################
 
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
    # names_2geneid=synonyms_2geneid | symbol_2geneid # dictionary concatenation operand. works with python 3.9+
    # return names_2geneid | alias_2geneid 

def get_ubiquitin_data(data_dir, alias_2geneid):
    print("\n>>>>>>> getting ubiquitin dataset:")

    ubiq_labels = pd.read_csv(data_dir+"UbiNet2E3_substrate_interactions.tsv", sep="\t", usecols=['E3_ID','SUB_ID','E3_ORGANISM','SUB_ORGANISM'])
    ubiq_labels = ubiq_labels[ubiq_labels['E3_ORGANISM'].str.startswith('Saccharomyces')]
    ubiq_labels = ubiq_labels[ubiq_labels['SUB_ORGANISM'].str.startswith('Saccharomyces')].reset_index(drop=True)
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

def get_protein_complexes_data(data_dir, alias_2geneid):
    '''
    Load dataset of protein complexes and use them as undirected positive interacctions.
    every undirected interaction is represented as two directed interactions in opposite directions
    '''
    print("\n>>>>>>> getting protein complex dataset:")
    complexes_label = pd.read_csv(data_dir+"cyc2008.txt", sep="\t",)
   
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

def get_kegg_Kpi(data_dir, species, alias_2geneid): #TODO magari dividilo in due cosi printa tutto in due, e tutto diventa diviso in due, ogni dataset e' separato. qst vuol dire anche che aggiungi una flag quando li mergi e fai un remove dupes generale, in cui c''e 'una gerarchi di flags per decidere quale tenere (kegg->ubiq->kpi->pcomplex).
    print("\n>>>>>>> getting kegg and kpi sets:")
    # PPI data generated with extract_kegg_interactions.py
    label=pd.read_csv(data_dir + species + "_kegg_signed_ppi.txt", sep= "\t", header=None , skiprows=1, index_col=(0,1)) 
    label=label[label[3]!='indirect effect']
    label=label[label[2]!='indirect effect']
    label=label[label[2]!='repression'] #transcription factors
    label[2].replace({"activation":0,"inhibition":1, 'binding/association':0, 'expression':0, 'dissociation':1}, inplace=True)
        
    label.drop(axis=1, labels=3, inplace=True)
    label[3] = 0.8
     # 24 07 TODO aggiunto ste due righe per contare quanti sono i signs da qui
    label_kegg = pd.Series(data=list(label[2]), index=label.index)
    label_kegg.name = 2 
    if species == 'S_cerevisiae': #todo temporary, translate scerevisiae in the source
        label_kegg = translate_multi_ind(label_kegg, alias_2geneid)
    print('KEGG lables', label_kegg.value_counts())
    ## from Patkar and Sharan 2018
    if species == 'S_cerevisiae': #todo temporary
        print('training edges from kpi:', len(label))
        label_kpi=pd.read_csv(data_dir+"yeast_kpi.txt", sep="\t",header=None, skiprows=1, index_col=(0,1)).dropna()
        label_kpi[3] = 0.6
        # 24 07 TODO aggiunto ste due righe per contare quanti sono i signs da qui
        label_kpi_s = pd.Series(data=list(label_kpi[2]), index=label_kpi.index)
        label_kpi_s.name = 2 
        label_kpi_s = translate_multi_ind(label_kpi_s, alias_2geneid)
        print('kpi lables', label_kpi_s.value_counts())
        ##
        label=pd.concat([label,label_kpi])
    edge_weights=pd.Series(data=list(label[3]), index=label.index)
    label = pd.Series(data=list(label[2]), index=label.index)
    label.name = 2 
    edge_weights.name=2
    if species == 'S_cerevisiae': #todo temporary, translate scerevisiae in the source
        #'translating indexes:'
        label = translate_multi_ind(label, alias_2geneid)
        edge_weights = translate_multi_ind(edge_weights, alias_2geneid)
    #'Removing duplicates'
    label=remove_dupes(label)
    edge_weights=edge_weights.loc[label.index]
    return label, edge_weights

def wrapper_get_training_data(data_dir, alias_2geneid,datasets):
    labels_of = {}
    weights_of = {}
    for dataset_name in datasets:
        if dataset_name == 'kegg_kpi':
            labels_of[dataset_name], weights_of[dataset_name] = kegg_Kpi_labels, kegg_Kpi_edge_weights = get_kegg_Kpi(data_dir,alias_2geneid) 
        if dataset_name == 'p_complex':
            labels_of[dataset_name], weights_of[dataset_name] = pcomp_labels, pcomp_edge_weights = get_protein_complexes_data(data_dir,alias_2geneid)
        if dataset_name == 'ubiq':
            labels_of[dataset_name], weights_of[dataset_name] = ubiq_labels, ubiq_edge_weights = get_ubiquitin_data(data_dir, alias_2geneid)
    return labels_of, weights_of

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
    translates column names and row names into entrex geneIDs
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

def get_perturbations_map(data_dir, alias_2geneid,species):
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
        perturbations_map=pd.read_csv(data_dir+'mutational_signatures_Holstege.cdt',sep='\t',header=0, skiprows=[0,1,3,4,5,6], index_col=1) #'index_col = 1 does a better job' #yes, keep row 2...
        perturbations_map.drop(perturbations_map.columns[range(6)],axis=1,inplace=True)
        print('initial experiments before translation: ', perturbations_map.shape[1])
        print('initial affected genes before translation: ', perturbations_map.shape[0])
        #02 TODO 13/11 remove transcription factors from knockouts (see mail with roded as to why)
        # pensavo di averlo fatto ma sto codice non fa nulla
        # def read_transcription_annotations_from_GFF(annotation_file_dir, aspect='F'):
        #     functional_gene_dict = collections.defaultdict(list)
        #     with open(annotation_file_dir, 'r') as f:
        #         for line in f:
        #             if line[0] != '#':
        #                 if 'transcription factor' in line:
        #                     functional_gene_dict[line[2]].append(1)
        #     return functional_gene_dict
    
        # asd=read_transcription_annotations_from_GFF(data_dir+'yeastannotation.gff')
        # n=0
    
        # for knockout_gene in perturbations_map.columns:
        #     if knockout_gene in asd.keys():
        #         n+=1
    
    if species == "H_sapiens":
        anndataobject=anndata.read_h5ad(data_dir+"K562_gwps_normalized_bulk_01.h5ad")
        #ann.X is the heatmap
        #ann.obs is metadata on rows (y axis)
        # ann. var is metadata on columns (x axis)
        # https://gwps.wi.mit.edu/: In the anndata format, the .var annotation details genes while the .obs annotation details single-cells/pseudobulk populations.'
        data=anndataobject.X
        colnames=anndataobject.var.gene_name
        rownames=[name.split('_')[1] for name in list(anndataobject.obs.index)]
        
        perturbations_map = pd.DataFrame(data, columns=colnames, index=rownames)
        perturbations_map.drop('non-targeting', inplace=True) # removes rows of non-targeting sgRNA counts, used for batch normalization

    perturbations_map=translate_axes(perturbations_map, alias_2geneid)
    print('Knockdown?perturbation experiments: ', perturbations_map.shape[1])
    print('Affected genes: ', perturbations_map.shape[0])

    return perturbations_map 

def extract_knockotut_effect_pairs_from_data(holst, genes, threshold=1.7): # See ref. for threshold value

    plus_targets_of_deletion={}
    minus_targets_of_deletion={}
    print('filter out expression threshold')
    for source  in holst.columns:
        list_of_plus_targets=holst[source][(holst[source]>threshold)].index
        list_of_minus_targets=holst[source][(holst[source]<-threshold)].index # the gene names of genes with threshold value for that particular experiment (source)
        if (len(list_of_plus_targets)>0) and (len(list_of_minus_targets)>0):
            plus_targets_of_deletion[source]=list(list_of_plus_targets)
            minus_targets_of_deletion[source]=list(list_of_minus_targets)
    plus_targets_of_deletion, minus_targets_of_deletion = remove_proteins_from_experiments(plus_targets_of_deletion, minus_targets_of_deletion, genes)   
    return plus_targets_of_deletion, minus_targets_of_deletion



def read_network_from_file(network_dir, species):
    """
    input:
        network_dir: str
        species: str ['S_cerevisiae', 'H_sapiens']
    output:
        pd.DataFrame: index: tuple, columns: confidence, directed
    """
    filename=network_dir+os.sep+species+'.net'
    return pd.read_csv(filename,sep="\t", index_col = [0,1],usecols=([0,1,2,3]), header=None)

        
def graph_from_dataframe(dataframe, net_type="undir"):
    """
    input: 
        dataframe: pd.DataFrame
        net_type: str ['dir', 'undir'] current version only uses undirected networks
    output:
        networkx.graph
    v3: treating the input as undirected graph. For the directed case, see v2
    """
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

    
def add_edges_from_labels(graph, edge_weights, flag):
    '''
    adds missing edges to network from labels.
    If edges come from Kegg and KPi or ubiquitin datasets, 
    will be added as directed because they are directed by definition. 
    Moreover, in this case, their opposite direction counterprts will be removed.
    If edges come from protein_complex validation set, will be added as undirected
    Inputs:
    graph: nx.Graph() or nx.DiGraph(). The base interaction network
    edge_weights: pd.Series. indexes are a tuple of protein ids, and the alue is the edge weight
    flag: str: 'kegg_kpi','protein_complex' or 'ubiquitin'
    '''
    directedgraph=graph.to_directed() # if already directed (nx.DiGraph object), returns a (deep) copy   
    edges=directedgraph.edges()
    n=0
    c=0
    a=0
    u=0
    w=[]
    for edge in edge_weights.index:
        if flag == 'kegg_kpi' or flag == 'ubiq':
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
                    if not directedgraph[edge[1]][edge[0]]['weight'] == 0: #oriented edges in basenet with claimed orientation equal to known training edges' orientation
                        n+=1
                    directedgraph[edge[1]][edge[0]]['weight'] == 0
                    directedgraph.add_edge(edge[0],edge[1], weight=edge_weights[edge])
        elif flag=='p_complex':
            if not (edge[0],edge[1]) in edges:
                a+=1
                directedgraph.add_edge(edge[0],edge[1], weight=0.8)
            else:
                c+=1
                w.append(directedgraph[edge[0]][edge[1]]['weight'])
                directedgraph[edge[0]][edge[1]]['weight']=max(directedgraph[edge[0]][edge[1]]['weight'], 0.8) #udpate weight of current edge to 0.8 if lower than that,
            # directedgraph.add_edge(edge[1],edge[0], weight=0.8)
        else:
            raise ValueError('Choose a valid flag between "kegg_kpi", "p_complex", ubiq"')
    print(n, 'edges in base networ k with claimed direction opposite to known training data edges orientation')
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

def remove_dupes(data):
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


# def remove_flips(data):
#     indexes_unfilpped=[]
#     to_flip=[]
    
#     for x in data.index:
#         if (x[1],x[0]) in  data.index:
#             indexes_unfilpped.append(x)
#             to_flip.append(x)
#             opposite=pd.MultiIndex.from_tuples([(x[1],x[0])])
    
#             data.drop(opposite, inplace=True)
#             data.drop(x, inplace=True)     
#     return data


def remove_proteins_from_experiments(plus_targets_of_deletion, minus_targets_of_deletion, genes, v=True):
    not_present_in_keys=[]
    if v:
        print('initial plus dictionary length', len(list(plus_targets_of_deletion.keys())))
        print('initial minus dictionary length', len(list(minus_targets_of_deletion.keys())))

    for source in plus_targets_of_deletion.keys():

        if not source in genes:
            not_present_in_keys.append(source)
    for source in minus_targets_of_deletion.keys():
        if (not source in genes) and (not source in not_present_in_keys):
            not_present_in_keys.append(source)
    
    if v:
        print(len(not_present_in_keys), 'proteins not present in keys')
    for protein in not_present_in_keys:
        if protein in plus_targets_of_deletion.keys():
            plus_targets_of_deletion.pop(protein, None)
        if protein in minus_targets_of_deletion.keys():
            minus_targets_of_deletion.pop(protein, None)
    
    if v:
        print('after removng plus keys not present in genes', len(list(plus_targets_of_deletion.keys())))
        print('after removng minus keys not present in genes', len(list(minus_targets_of_deletion.keys())))



    for source in plus_targets_of_deletion.keys():
        
        not_present_in_items=[]

        for target in plus_targets_of_deletion[source]:
            if (not target in genes) and (not target in not_present_in_items):
                not_present_in_items.append(target)
            
        for item in not_present_in_items:
            plus_targets_of_deletion[source].remove(item)


    for source in minus_targets_of_deletion.keys():
        
        not_present_in_items=[]

        for target in minus_targets_of_deletion[source]:
            if (not target in genes) and (not target in not_present_in_items):
                not_present_in_items.append(target)
            
        for item in not_present_in_items:
            minus_targets_of_deletion[source].remove(item)

            
    empty_key=[]
    for source in plus_targets_of_deletion.keys():
        if len(plus_targets_of_deletion[source])==0:
                empty_key.append(source)
    for source in minus_targets_of_deletion.keys():
        if (len(minus_targets_of_deletion[source])==0) and (not source in empty_key):
                empty_key.append(source)
                
    for protein in empty_key:
        if protein in plus_targets_of_deletion.keys():
            plus_targets_of_deletion.pop(protein, None)
        if protein in minus_targets_of_deletion.keys():
            minus_targets_of_deletion.pop(protein, None)
            
    if v:
        print('after removng plnus empty keys ', len(list(plus_targets_of_deletion.keys())))
        print('after removng minus empty keys ', len(list(minus_targets_of_deletion.keys())))

           
    return plus_targets_of_deletion, minus_targets_of_deletion

def write(OUTDIR, labels_of):
    '''datasets is a list of (pd.Series, datasetname string) tuples'''
    for  dataname, labels in labels_of.items():
        labels=labels.reset_index()
        labels[[0,1]]=pd.DataFrame(labels['index'].tolist(), index=labels.index)  
        labels.drop(columns='index')
        labels=labels[[0,1,2]]
        labels.to_csv(OUTDIR+dataname+'.lbl.tsv', sep='\t', header=None, index=False)
    
