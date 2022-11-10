'''
03 08 2022 creates a dataet of ppis from kegg with sign annotation.
'''
import collections
import itertools
import lxml.etree as etree
import os
import pandas
import networkx 
import numpy as np
import pandas as pd


SPECIES = "H_sapiens" #"S_cerevisiae"#
HOME_DIR  =  "G:" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
# HOME_DIR  =  "G:" +os.sep+"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 
#HOME_DIR  =  "/content/drive" +os.sep+"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep 

MAIN_DATA_DIR = HOME_DIR+'Data'+os.sep+SPECIES+os.sep
DIR=HOME_DIR+'Data'+os.sep+'scripts'+os.sep

# ANAT_DIRECTORY = "G:"+os.sep +"Il mio Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep +"07-2020_ANAT"+os.sep#"C:" + os.sep + "Users" + os.sep + "lorenzos" + os.sep + "Google Drive" + os.sep + "SECRET-ITN"  + os.sep + "Projects" + os.sep + "07-2020_ANAT" + os.sep#
ANAT_DIRECTORY = "G:"+os.sep +"My Drive"+ os.sep +"SECRET-ITN" + os.sep +"Projects" + os.sep +"07-2020_ANAT"+os.sep#"C:" + os.sep + "Users" + os.sep + "lorenzos" + os.sep + "Google Drive" + os.sep + "SECRET-ITN"  + os.sep + "Projects" + os.sep + "07-2020_ANAT" + os.sep#

KEGG_DIRECTORY = ANAT_DIRECTORY + "downloads" + os.sep + "kegg" + os.sep + SPECIES + os.sep
GENE_INFO_DIRECTORY = ANAT_DIRECTORY + "downloads" + os.sep + "kegg"+os.sep
RESULTS_DIRECTORY = ANAT_DIRECTORY + "results" + os.sep + SPECIES + os.sep
D2D_OUTPUT_DIRECTORY = ANAT_DIRECTORY + os.sep + "Diffuse2Direct-master" + os.sep + "DrugsInfo" + os.sep +"output" + os.sep

NET_FILE_SUFFIX = ".net"
SET_FILE_SUFFIX = ".set"
MSG_FILE_SUFFIX = ".msg"
FREQ_FILE_SUFFIX = ".freq"
ANCHORS_PROP_FILE_SUFFIX = ".A.prp"
TERMS_PROP_FILE_SUFFIX = ".T.prp"

KEGG_CONFIDENCE = 0.6
ANCHOR_SYMBOL = "ANCHOR"

STEINER_FILENAME = ANAT_DIRECTORY  + ".." + os.sep + "AnatWeb" + os.sep +  "steiner" + os.sep + "steinprt"


ALPHA_VALUES = [0, 0.25, 0.5]
TYPE_OF_BASE_NETWORKS = ["undir"]#,"dir"] # only using undirected networks now.
MARGINS =[1,1.2,1.4,1.6,1.8,2]

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

alias_2geneid =  readname2geneid(GENE_INFO_DIRECTORY, SPECIES)
        
def generate_name_map(kegg_pathways):
        name_map = {}
        name_graph = networkx.Graph()  

        for pathway in kegg_pathways.values():
            for entry in pathway.xpath("entry[@type='gene' or @type='compound']"):
                names = [name.split(':')[1] for name in entry.get("name").split()]
                name_graph.add_nodes_from(names)
                name_graph.add_edges_from(itertools.combinations(names, 2)) 

        for component in networkx.connected_components(name_graph):
            component_name = max(sorted(component), key=lambda node: name_graph.degree[node])
            name_map.update(dict.fromkeys(component, component_name))

        return name_map

def read_pathway_interactions(pathways, name_map, alias_of_id=None):
    relation_type_map = collections.defaultdict(dict)
    for pathway_id, pathway in pathways.items():
        entry_map = collections.defaultdict(set)
        compounds = set()
    
        for entry in pathway.xpath("entry[@type='gene' or @type='compound']"):
            name = name_map[entry.get("name").split()[0].split(':')[1]]
            entry_map[entry.get("id")] = {name}
            if entry.get("type") == "compound":
                compounds.add(name)
    
        for entry in pathway.xpath("entry[@type='group']"):
            names = set(itertools.chain.from_iterable([entry_map[component.get("id")] for component in entry.xpath("component")]))
            entry_map[entry.get("id")] = names
    
        for relation in pathway.xpath("relation"): # fai qui, aggiungi flag subtype per prendere la relation che vuoi e fare un nuovo kegg dataset
            if len(relation.xpath('subtype'))>0:
                subtype=relation.xpath('subtype')[0]
                if not subtype.get('name') == 'compound':
                    interaction_type=subtype.get('name')
                    for source, target in itertools.product(entry_map[relation.get("entry1")], entry_map[relation.get("entry2")]):
                        relation_type_map[source][target]=interaction_type

    return relation_type_map

pathway_interactions = {}
pathways={}

# generate pathway_interactions
#%%
print("generating pathway_interactions")

for pathway_filename in os.listdir(KEGG_DIRECTORY):
    pathway =   etree.parse(KEGG_DIRECTORY + pathway_filename)
    pathways[os.path.splitext(pathway_filename)[0]] = pathway

name_map = generate_name_map(pathways)

pathway_interactions=read_pathway_interactions(pathways, name_map)

kegg_data=pandas.DataFrame.from_dict(pathway_interactions)
# kegg_data.values[[np.arange(len(kegg_data))]*2]=np.nan # non ho idea di cosa faccia, funge con cerevisiae levando qualche line apparentemenete senza motivo (non sono duplicati)
edgelist=kegg_data.stack().reset_index()
edgelist.rename(columns={'level_0':'A','level_1':'B',0:'annotation1'}, inplace=True)
edgelist['annotation2']=0
#%%
edgelist.to_csv(MAIN_DATA_DIR+SPECIES+'_kegg_signed_ppi.txt', index=False, header=False, sep='\t')
