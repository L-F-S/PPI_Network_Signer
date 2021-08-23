# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 13:51:18 2021

@author: los4
"""
import pandas as pd
import collections
import networkx


###############################################################################
# READ, FILTER AND PREPARE DATA
###############################################################################


def read_alias2geneid(GENE_INFO_DIRECTORY, species, alias_column='LocusTag', final_column='GeneID'):
    if species =="H_sapiens":
        return None
    elif species == "S_cerevisiae":
        geneinfo_filename = GENE_INFO_DIRECTORY+ species + ".gene_info"
        geneinfo = pd.read_table(geneinfo_filename, usecols=[final_column,alias_column])
        
        if len(geneinfo.columns) == 1:
            geneinfo["GeneID_str"] = geneinfo[final_column].astype(str)
        return {series[alias_column]: series[final_column] for ind, series in geneinfo.iterrows()}
    else:
        raise ValueError("Species not in database")


def get_targets(df):
    plus_targets_of_deletion=collections.defaultdict(list)
    minus_targets_of_deletion=collections.defaultdict(list)

    for i, series in df.iterrows():
        if float(series) >0:
            plus_targets_of_deletion[i[0]].append(i[1])
        else:
            minus_targets_of_deletion[i[0]].append(i[1])

    return plus_targets_of_deletion, minus_targets_of_deletion


def translate_ind(data, alias_2geneid, v=True):
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

def translate_experiments_holstege(df, alias_2geneid, v=False):
    column_map = {}
    not_found=[]
    for oldname in df.columns:
        column_map[oldname] = alias_2geneid[oldname]
    index_map= {}
    for oldname in df.index:
        try:
            index_map[oldname] = alias_2geneid[oldname]
        except:
            not_found.append(oldname)
    if v:
        print("number of targets not found in dictionary:", len(not_found))
    for dubious_orf in not_found:
        df.drop(dubious_orf, inplace= True)
    return df

def translate_and_get_targets(df, alias_2geneid):
    ''' TODO unused
    df: pd.DataFrame
    alias2geneid: dictionary
    '''
    
    plus_targets_of_deletion=collections.defaultdict(list)
    minus_targets_of_deletion=collections.defaultdict(list)
    
    noalias=0
    for i, series in df.iterrows():
        if float(series) >0:
            try:
                alias_2geneid[i[0]]
                alias_2geneid[i[1]]
                plus_targets_of_deletion[alias_2geneid[i[0]]].append(alias_2geneid[i[1]])
            except:
                noalias+=1
        else:
            try:
                alias_2geneid[i[0]]
                alias_2geneid[i[1]]
                minus_targets_of_deletion[alias_2geneid[i[0]]].append(alias_2geneid[i[1]])
            except:
                noalias+=1
    print('no alias found for',  noalias, 'interactions')
    return noalias, plus_targets_of_deletion, minus_targets_of_deletion


def graph_from_dataframe(dataframe, net_type):
    """input: pd.DataFrame, str ['dir'.'undir']
        output: networkx.graph
    """
    if net_type=="undir":
            return networkx.from_pandas_edgelist(dataframe.reset_index(), 0,1,2)

    elif net_type=="dir":
        graph = networkx.DiGraph()
        for (source, target, confidence, directed) in dataframe.reset_index().itertuples(False):
            graph.add_edge(source, target, weight=confidence)
            if directed == 0:
                graph.add_edge(target, source, weight=confidence)
        return graph

    else:
        raise ValueError("Incorrect net type: "+str(net_type))


def remove_dupes(training_data):
    seen = {}
    dupes = []
    
    for x in training_data.index:
        if x not in seen:
            seen[x] = 1
        else:
            if seen[x] == 1:
                dupes.append(x)
            seen[x] += 1
    return training_data.drop(dupes)   


def remove_flips(training_data):
    indexes_unfilpped=[]
    to_flip=[]
    
    for x in training_data.index:
        if (x[1],x[0]) in  training_data.index:
            indexes_unfilpped.append(x)
            to_flip.append(x)
            opposite=pd.MultiIndex.from_tuples([(x[1],x[0])])
    
            training_data.drop(opposite, inplace=True)
            training_data.drop(x, inplace=True)     
    return training_data


def add_missing_edges(network, training_label, net_type, edge_confidence=0.6):
    print('function currently used but should not change anything')
    print('apparently is adding a column somewhere')
    label_not_in_bn=[]
    indexes_to_change={}
    n=0
    z=0
    for interaction in training_label.index:
    
        if not interaction in network.index:
            if not interaction[::-1] in network.index:
                label_not_in_bn.append(interaction)
            else:
                indexes_to_change[interaction] = interaction[::-1]
                if net_type=='dir':
                    label_not_in_bn.append(interaction) # this is a different interaction in a directed network
    
                z+=1
                n+=1
        else:
            n+=1
    print("interactions not in network: ",len(label_not_in_bn))
    print("interactions in network (regardless of orientation): ",n)
    print('interactions to invert: ', z)
    print('todo aIf undirected, maybe remove the edges to invert? because what s the poin in inverting \
              direction on a signed thing? this comes down tonot knowing what sign in undirected network means')
    
    # invert indexes in training label WHICH IS DUMB
    if net_type=='undir':
        as_list = training_label.index.tolist()
        for old_i, new_i in indexes_to_change.items():
            idx = as_list.index(old_i)
            as_list[idx] = new_i
        training_label.index = as_list
    
    # # Aggiungiamo allora questi indici mancanti.
    
    if net_type=='undir':
        network = network.append(pd.DataFrame(index=pd.MultiIndex.from_tuples(label_not_in_bn, names=[0,1]), data=[[edge_confidence,0]]*len(label_not_in_bn), columns=[2,3]))
    else:
        network = network.append(pd.DataFrame(index=pd.MultiIndex.from_tuples(label_not_in_bn, names=[0,1]), data=[[edge_confidence,1]]*len(label_not_in_bn), columns=[2,4]))

    for interaction in training_label.index:
        if not (interaction in network.index):
            print('is it to swap?', interaction in indexes_to_change.keys())
            print("qst NON ACCADE")
        
    return network


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