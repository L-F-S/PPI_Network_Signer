
###############################################################################
# TRAIN CLASSIFIER
###############################################################################
import numpy as np
import pandas as pd
# from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve,  auc
from random import seed, sample
import pickle

plt.rcParams["font.family"] = "times new roman"


def log(dataname, AUROC, AUPR, logfile, DATE, SPECIES, pert_map, edgedatainbasenet='-', comments=''):
    AUROC=str(AUROC)
    AUPR=str(AUPR)
    line=dataname+'\t'+AUROC+'\t'+AUPR+'\t'+DATE+'\t'+edgedatainbasenet+'\t'+pert_map+'\t'+SPECIES+'\t'+comments+'\n'
    with open (logfile, 'a') as f:
        f.write(line)
    return

def load_features(OUTDIR, datanames, species):
    features={}
    for dataname in datanames:
        with open(OUTDIR+dataname+'.ft','rb') as f:
            features[dataname] = pickle.load(f)
    return features



def read_features(OUTDIR, DATE, which=['k','u','p'], whichinbase=['k','u','p']):
    labels=[]
    features=[]
    s=''
    inbase=''.join(whichinbase)
    if 'k' in which:
        labels.append( pd.read_csv(OUTDIR+'kegg_kpi'+'.lbl.tsv', header=None, index_col=[0,1], sep='\t') )
        features.append(pd.read_csv(OUTDIR+'kegg_Kpi_network_'+inbase+'.ft.csv', sep=',', header=0))
        s+='k_k_'
    if 'u' in which:
        labels.append(pd.read_csv(OUTDIR+'ubiq'+'.lbl.tsv', header=None, index_col=[0,1], sep='\t') )
        features.append(pd.read_csv(OUTDIR+'ubiq_network_'+inbase+'.ft.csv', sep=',', header=0))
        s+='ub_'
    if 'p' in which:
        labels.append(pd.read_csv(OUTDIR+'p_complex'+'.lbl.tsv', header=None, index_col=[0,1], sep='\t') )
        features.append(pd.read_csv(OUTDIR+'p_complex_network_'+inbase+'.ft.csv', sep=',', header=0))
        s+='pcmp_'
    s2=''
    if 'k' in whichinbase:
        s2+='k_k_'
    if 'u' in whichinbase:
        s2+='ub_'
    if 'p' in whichinbase:
        s2+='pcmp_'
    finaldataname=s+'network'+s2+DATE #todo manca la condiz edge in basenet
    return pd.concat(labels), pd.concat(features), finaldataname, ' '.join(which), ' '.join(whichinbase)

XLABEL_OF={"PrecRec":"Recall (TP/P)","ROC":"False Positive Rate"}
YLABEL_OF={"PrecRec":"Precision (TP/PredictedP)","ROC":"True Positive Rate"}
TITLE_OF ={"PrecRec":"Precision vs Recall","ROC":"ROC curve"}
COLOR_OF ={"PrecRec":"magenta","ROC":"cyan"}

seed(1234)
def random_even_subsample(data, labels, Seed):
    '''data is a pd.DataFrame, labels is a pd.Series of 0s and 1s
    checks which class is the majrity, and randomly downsamples it
    to make its dimention the same as the minority class
    '''
    # Calculate mean of the labels. If balanced, it will be =0.5
    nones=np.count_nonzero(labels)
    nzeroes=len(labels)-nones
    mean=nones/len(labels)

    seed(Seed) # set random sample seed, to make experiment repeatable
    if mean<1/2: # which is always my case cos I have a majority of negatives
        high_indexes=labels[labels==0].index
        rand_indexes_to_drop = sample(list(high_indexes), k = nzeroes - nones)
    elif mean > 1/2:
        high_indexes=labels[labels==1].index
        rand_indexes_to_drop = sample(list(high_indexes), k = nones - nzeroes)
    else:
        return data, labels
    data=data.drop(rand_indexes_to_drop)
    labels=labels.drop(rand_indexes_to_drop)
    
    return data, labels

def upsample(upsampling, training_indexes, y, X):
    '''
    Oversample with different techniques
    upsampling: str: ['', 'gauss', 'SMOTE']
    X , y are pd.DataFrames
    '''
    if upsampling =='':
        return np.array(X), np.array(y),
    # elif  upsampling == 'SMOTE':
    #     # print('Upsampling training split with SMOTE method...')
    #     oversample = SMOTE()
    #     X, y = oversample.fit_resample(np.array(X), np.array(y))
    #     return  X, y
    elif upsampling =='gauss':
        # print('Upsampling training split with gauss method...')

        # add random noise to minority class until it's comparable
        # Calculate mean of the labels. If balanced, it will be =0.5
        nones=np.count_nonzero(y)
        mean=nones/len(y)

        if mean<1/2: # which is always my case cos I have a majority of negatives
            low_indexes=y[y==1].index
            lowX = np.array(X.loc[low_indexes])
            lowy=np.array(y[y==1])
            # concatenate len(lowX) additional random noise with the same scale
            # of the mean of each feature
            M = lowX.mean(axis=0) # column-wise mean
            gaussX = lowX+np.random.normal(loc=M, scale=abs(M), size=lowX.shape)
            gaussy=np.ones((len(lowy)))
            
        elif mean > 1/2:
            low_indexes=y[y==0].index
            lowX = np.array(X.loc[low_indexes])
            lowy=np.array(y[y==0])
            M = lowX.mean(axis=0) # column-wise mean
            gaussX = lowX+np.random.normal(loc=M, scale=abs(M), size=lowX.shape)
            gaussy=np.zeroes((len(lowy)))
        else:
            raise ValueError('classes are already balanced!')   
        return np.concatenate((np.array(X),gaussX)), np.concatenate((y,gaussy)) 

def downsample(training_indexes, labels):
    '''indexes are indexes, labels is a pd.Series of 0s and 1s'''
    nones=np.count_nonzero(labels)
    nzeroes=len(labels)-nones
    mean=nones/len(labels)


    if mean<1/2: # which is always my case cos I have a majority of negatives
        high_indexes=labels[labels==0].index
        rand_indexes_to_drop = sample(list(high_indexes), k = nzeroes - nones)
    else:
        high_indexes=labels[labels==1].index
        rand_indexes_to_drop = sample(list(high_indexes), k = nones - nzeroes)
    
    return np.setdiff1d(training_indexes, rand_indexes_to_drop)

def predict(classifier, X_train, y_train, X_test, y_test, dataname):
    return


def AUCs(classifier, features, labels):
    X=StandardScaler().fit_transform(features)#, columns=features.index)
    y_predicted=classifier.predict_proba(X)[:,1:] # indexing this way because of output for predict_proba
        
    tpr, fpr, thresholds = precision_recall_curve(labels, y_predicted) # not actually fpr and tpr in this case, they are tprs=precision, fpr = recall
    tpr=tpr[::-1]  #they are reversed for some reason
    fpr=fpr[::-1]
    pr_auc= auc(fpr, tpr) #float, the AUC
    fpr, tpr, thresholds = roc_curve(labels, y_predicted)
    roc_auc= auc(fpr, tpr) #float, the AUC
    return roc_auc, pr_auc
    
def k_fold(classifier, n_folds, x_train, y_train, dataname, IMG_DIR,\
                  metric="ROC", plot=True, downsampling='', upsampling='',std=True):
    '''
    x_train: pd.DataFrame of training features (index= samples, columns = features)
    y_train: pd.Series of training labels (index= samples)
    classifier: sklearn classifier
    std: wether to standardize feature data. default=True. Not advisable to change
    '''
    
    if std: #scale features to be normally distributed  (0,1)
        x_train = pd.DataFrame(StandardScaler().fit_transform(x_train), columns=x_train.columns)
                               
     
    if upsampling:
        print('upsampling data with method: ', upsampling)
        
    interp_0_of={"PrecRec":1.0,"ROC":0.0}
    tprs = []  # list of np.arrays
    aucs = []  # list of floats
    mean_fpr = np.linspace(0, 1, 100)  # create a line to add points in (x axis of graph), for false psoitve rate

    cv=StratifiedKFold(n_splits=n_folds)   #splits TRAINING data into n folds.
    if plot:
        fig, ax =plt.subplots()
    for i, (train, test) in enumerate(cv.split(np.array(x_train), np.array(y_train))):
        #31-8 put downsamplint only after splitting:

        if downsampling=='_DS':
            train = downsample(train, y_train.iloc[train])
            X, y = np.array(x_train.iloc[train]), np.array(y_train.iloc[train])
        X, y = upsample(upsampling, train, y_train.iloc[train], x_train.iloc[train])
        
        # Train classifier with subset (updates every fold)
        classifier.fit(X, y)         
        # create metric curve (ROC or PrecRec). Needs X and Y for current fold's 
        # test set to calculate tpr and fpr (or other metrics) at different thresholds
        # this is if you use RandomForest instead of LogisticRegression
        y_predicted=classifier.predict_proba(np.array(x_train.iloc[test]))[:,1:] # indexing this way because of output for predict_proba, which are probabilities
        if metric=="PrecRec":
            tpr, fpr, thresholds = precision_recall_curve(y_train.iloc[test], y_predicted) # not actually fpr and tpr in this case, they are tprs=precision, fpr = recall
            tpr=tpr[::-1]  #they are reversed for some reason
            fpr=fpr[::-1]
        elif metric=="ROC":
            fpr, tpr, thresholds = roc_curve(y_train.iloc[test], y_predicted)
        roc_auc= auc(fpr, tpr) #float, the AUC
        aucs.append(roc_auc)   # list of all aucs, called from a plot_roc_curve's method
        
        #interpolate line of true positive rates. We interpolate cos every fold will output a 
        # tprs (or other metric) with different lengths, so in order to have same lengths to take the mean later
        interp_tpr = np.interp(mean_fpr, fpr, tpr) #linear interpolation between points. 
        # To evaluate on coordinates: mean_fpr. Points to use: x data points: viz.fpr; y datapoints = viz.tpr
        # this is so that we can plot for every mean_fpr point (i.e. for every point on the x axis), the (sort of) corresponding 
        # point on the y axis (i.e. the tpr, or Recall, or other metric)
        interp_tpr[0] = interp_0_of[metric] #0.0 for ROC, 1.0 for PrecRec
        tprs.append(interp_tpr) #the list of y points to take the mean on
        
        # plot the curve (one for every fold)
        if plot:
            ax.plot(fpr, tpr, lw=1, label ="k {}".format(i+1), alpha=0.3)
            ax.set_xlabel(XLABEL_OF[metric])
            ax.set_ylabel(YLABEL_OF[metric])


    ########
    #  Plot x y bisector:
    if metric == "ROC":
        if plot:
            ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray',
                label='bisector', alpha=.8)

    # plot the mean ROC, taking the mean of all other ROCs
    mean_tpr = np.mean(tprs, axis=0) #takes point-wise mean for all tpr lines
    #mean_tpr[-1] = 1.0  
    mean_auc = auc(mean_fpr, mean_tpr)   #remember mean_fpr is just the x axis.(this is why we interpolated the y axes)
    std_auc = np.std(aucs)
    if plot:
        ax.plot(mean_fpr, mean_tpr, color=COLOR_OF[metric],
            label=r'Mean {} (AUC = {} $\pm$ {})'.format(metric, np.round(mean_auc,2), np.round(std_auc,2)),
            lw=2)

    # Plot one standard deviation above and below the mean
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    if plot:
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=COLOR_OF[metric], alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

        # Add title, legend, set xlim and ylim
        ax.set( xlim=[0, 1.1], ylim=[0, 1],
                title=TITLE_OF[metric]+" for classifier = "+str(classifier))
        ax.legend(loc=(1.1, 0))
        
    fig_name=dataname+"."+metric
    if plot:
        plt.savefig(IMG_DIR+fig_name+'.png',  bbox_inches='tight')

    return classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper, mean_auc, std_auc

###############################################################################
# 07 2024 Unique Enzyme CV for kinases and phosphatases functions:
###############################################################################
import networkx as nx
import random
import itertools
from sklearn.ensemble import RandomForestClassifier

###############################################################################
# Functions to assess feasibility and parameters of Unique enzyme CV
###############################################################################

def net_from_edges(edges):
    train_net=nx.DiGraph()
    for node1, node2 in edges:
        train_net.add_edge(node1, node2)
    return train_net


def get_degree_sorted_enzymes(train_net, plot, img_dir,dataset_name, sort=True):
    '''Sort nodes by number of outgoing edges. Nodes with
    outgoing edges are either kinases (label 0) or phosphatases (label 1),
    collectively referred to as \'enzymes\''''
    out_of={}
    for node in train_net.nodes:
        deg=train_net.out_degree(node)
        if deg>0: #this is a node1
            out_of[node]=deg
    
    if sort:
        enzyme_substrates=pd.Series(out_of).sort_values(ascending=False)
    else:
        enzyme_substrates=pd.Series(out_of)
    if plot:
        plot=enzyme_substrates.plot(kind='bar', fontsize=20, title='(a)')
        plot.title.set_size(20)
        s='_sorted' if sort==True else ''
        plt.savefig(img_dir+dataset_name+'enzyme_deg'+s+'.png')
        
    return enzyme_substrates

def find_largest_k_for_unique_enzyme_split(enzyme_substrates, k):
    '''Check fold size at given k is > biggest enzyme degree. If not, find 
    largest fold size which satisfies such requirement'''
    print('total edges', enzyme_substrates.sum())
    fold_size=int(enzyme_substrates.sum()/k) #rounds down to lower integer
    if enzyme_substrates.sum()/k<enzyme_substrates.iloc[0]:
        k-=1
        while enzyme_substrates.sum()<enzyme_substrates.iloc[0] and k>0:
            print(len(enzyme_substrates),'enzymes')
            print('n of folds=',k)
            print('train_data:',enzyme_substrates.sum())
            print('fold size:',fold_size)
            print('largest degree',enzyme_substrates.iloc[0])
            k-=1
    
    print('Largest K for unique enzyme CV:',k)
    print('fold size= ',fold_size)
    print('largest degree',enzyme_substrates.iloc[0])
    print('rest:',enzyme_substrates.sum()%k)
    return k, fold_size

###############################################################################
# Enzyme-wise splitting functions
###############################################################################

def get_enzyme_wise_splits(enzyme_substrates, fold_size):
    '''Check if possible to build k enzymes groups where the sum of their 
    degrees is exactly = to foldsize, plus one group with rest of division.
    This is data dependant, it is easy with many small degree enzymes
    (always true with all 1 degree) and harder with many large degree enzymes'''
    
    # generate random starting index
    def generate_random_ind_iterator(enzyme_substrates):
        starting_ind = random.randint(0,len(enzyme_substrates)-1)
        index_iterator = [i for i in range(starting_ind, len(enzyme_substrates))]
        index_iterator += [i for i in range(0,starting_ind)]
        return index_iterator
    
    
    def generate_splits(enzyme_substrates, index_iterator, fold_size):
        # temp trovato bug gli enzimi da 1 substrato mi scazzano tt
        used_inds=[]
        n=0
        total_degs=0
        fold_bins_enzymes=[]
        
        
        for i,deg in [(enzyme_substrates.index[ind], enzyme_substrates.iloc[ind]) for ind in index_iterator]:
            # print(i, deg)
            enzymes=[]
            if not i in used_inds: # bug temp: cosi da 1 non ci entrano.
                n+=1
                degrees_comb=0
                print('fold n', n)
                # print('first guy',i, deg)
                degrees_comb+=deg
                total_degs+=deg
                enzymes.append(i)
                used_inds.append(i)
                z=0
                while (degrees_comb!=fold_size): 
                    # print(degrees_comb,fold_size)
                    # print(len(used_inds),len(enzyme_substrates))
                    z+=1    
                    if z==len(enzyme_substrates)+2:
                        degrees_comb=1
                        fold_size=1
                        print('CANT FIT LAST FOLD, shouldnt ever arrive here')
                        print('current enzymr anddeg',i,deg)
                        continue
                    if len(used_inds)==len(enzyme_substrates):
                        print('done with leftovers:', degrees_comb,'edges from enzymes',enzymes)
                        print('adding each leftover enzyme to one of the previous folds')
                        for y,node in enumerate(enzymes):
                            fold_bins_enzymes[y].append(node)
                        break
                    
                    #add new degree
                    indexes_over_degree = []
                    for i,deg in enzyme_substrates.items():
                        if degrees_comb==fold_size:
                            print('fold completed perfectly\n')
                            break

                        if not i in used_inds:
                            if not degrees_comb+deg>fold_size:
                               degrees_comb+=deg
                               total_degs+=deg
                               # print('t',total_degs)
                               used_inds.append(i)
                               # print(i, degrees_comb)  
                               enzymes.append(i)
                            else: # deg+degrees_comb bigger than target, trzing with next index
                                # print(deg,'+',degrees_comb,'bigger than',fold_size,'skipping',i)
                                indexes_over_degree.append(i)
                                if (len(indexes_over_degree)+len(used_inds)==len(enzyme_substrates)):
                                    print('falied to succesfully create fold',n,"restarting with new shuffled indexes")
                                    return fold_bins_enzymes, False
                                continue
            
                if len(used_inds)<len(enzyme_substrates):
                    fold_bins_enzymes.append(enzymes)
                    
        print("full splitting completed")
        return fold_bins_enzymes, True
        
    split_flag=False
    while not split_flag:
        print("making new splits")
        index_iterator = generate_random_ind_iterator(enzyme_substrates)
        fold_bins_enzymes, split_flag = generate_splits(enzyme_substrates, index_iterator, fold_size)

    return fold_bins_enzymes

def check_signs_and_degree_distrib(training_labels, fold_bins_enzymes, enzyme_substrates, img_dir, dataset_name, plot):
    monotone_folds=[]
    for n,fold in enumerate(fold_bins_enzymes):
        signs=[training_labels.loc[node].iloc[0] for node in fold]
        if plot:
            letter_of={0:'(b)',1:'(c)',2:'(d)',3:'(e)',4:'(f)'}
            plot=enzyme_substrates.loc[fold].plot(kind='bar', title=letter_of[n], fontsize=20)
            plot.title.set_size(20)
            plot.set(ylim=(0,55))
            plt.savefig(img_dir+dataset_name+'enzyme_deg_fold'+str(n)+'.png')
            plt.show()
        if len(np.unique(signs))==1:
            monotone_folds.append(n)
            print('1s',signs.count(1),'0s', signs.count(0))
            print('Warning! fold',n,'mono-signed. Must perform enzyme swapping between folds')
    return

def get_edges(fold_bins_enzymes, train_net, k):
    '''get all edges for every enzyme node'''
    fold_edges=[]
    for fold_n in range(k):
        enzyme_nodes=fold_bins_enzymes[fold_n]
        indexes=[]
        for enzyme in enzyme_nodes:
            
            indexes+=[(enzyme,substrate) for substrate in train_net.neighbors(enzyme)]
            
        fold_edges.append(indexes)
    return fold_edges

def get_indexes(fold_edges, features_table):
    '''go from edge name to positional index in features table'''
    fold_indexes=[]
    for fold in fold_edges:
        indexes=[features_table.index.get_loc((edge)) for edge in fold]
        # print(len(indexes))
        fold_indexes.append(indexes)
    return fold_indexes

def get_indexes_slow_str(fold_edges, features_table):
    '''go from edge name to positional index in features table
    Created to overcome a bug of get_loc method for MultiIndex DataFrames where
    if the key is past the lexsort depth, the return may be a boolean mask array ,a slice or int.'''
    features_table=features_table.reset_index()
    fold_indexes=[]
    for fold in fold_edges:
        indexes=[]
        for edge in fold:
            temp=features_table[features_table['0']==edge[0]] # features_table[0] if it gives error
            index=temp[temp['1']==edge[1]].index[0]
            indexes.append(index)
        # print(len(indexes))
        fold_indexes.append(indexes)
    return fold_indexes

def get_indexes_slow_int(fold_edges, features_table):
    '''go from edge name to positional index in features table
    Created to overcome a bug of get_loc method for MultiIndex DataFrames where
    if the key is past the lexsort depth, the return may be a boolean mask array ,a slice or int.'''
    features_table=features_table.reset_index()
    fold_indexes=[]
    for fold in fold_edges:
        indexes=[]
        for edge in fold:
            temp=features_table[features_table[0]==edge[0]] # features_table[0] if it gives error
            index=temp[temp[1]==edge[1]].index[0]
            indexes.append(index)
        # print(len(indexes))
        fold_indexes.append(indexes)
    return fold_indexes

def check_unique_inds(fold_indexes, enzyme_substrates):
    count=0
    unique=[]
    for inds in fold_indexes:
        for ind in inds:
            count+=1
            if not ind in unique:
                unique.append(ind)
            else:
                print('warning',ind)
    if count>enzyme_substrates.sum():
        print('warning',count)
    return

def split_indexes(fold_indexes):
    '''creating CV train-test splits from fold indexes'''
    splits=[]
    k=len(fold_indexes)
    for i in range(k):
        test_inds = fold_indexes[i]
        train_inds=[]
        for j in range(k):
            if j!=i:
                train_inds += fold_indexes[j]
        splits.append((train_inds, test_inds))
    return splits

def check_labes(fold_indexes, training_labels):
    '''Check all folds have both signs. Should be redundant after all previous checks. '''
    one=False
    zero=False
    for fold in fold_indexes:
        for i in fold:
            if one and zero:
                return True
            if not one:
                if training_labels.iloc[i]==1:
                    one=True
            if not zero:
                if training_labels.iloc[i]==0:
                    zero=True
    return False

###############################################################################
# swapping functions
###############################################################################

def get_sign_only_data_per_fold(training_labels, fold_bins_enzymes,enzyme_substrates):
    ''' get number of 1s (the rarest always) and 0s per bin'''
    
    fold_bins_1_enzymes={}
    fold_bins_1_subs={}
    fold_bins_0_enzymes={}
    fold_bins_0_subs={}
    for i, fold in enumerate(fold_bins_enzymes):
        fold_i=enzyme_substrates.loc[fold]
        fold_i_10s=[training_labels.loc[node].iloc[0] for node in fold_i.index]
        indexes_i_1= [j for j, value in enumerate(fold_i_10s) if value == 1]
        fold_i_1_enzymes=list(np.array(fold)[indexes_i_1])
        fold_i_1_subs=enzyme_substrates.loc[fold_i_1_enzymes]
    
        fold_bins_1_enzymes[i]=fold_i_1_enzymes
        fold_bins_1_subs[i]=fold_i_1_subs
    
        indexes_i_0= [j for j, value in enumerate(fold_i_10s) if value == 0]
        fold_i_0_enzymes=list(np.array(fold)[indexes_i_0])
        fold_i_0_subs=enzyme_substrates.loc[fold_i_0_enzymes]
    
        fold_bins_0_enzymes[i]=fold_i_0_enzymes
        fold_bins_0_subs[i]=fold_i_0_subs
    
    return fold_bins_1_enzymes, fold_bins_1_subs, fold_bins_0_enzymes, fold_bins_0_subs

def check_ones(training_labels, fold_bins_enzymes, enzyme_substrates):
    fold_bins_1_enzymes, fold_bins_1_subs, fold_bins_0_enzymes, fold_bins_0_subs = get_sign_only_data_per_fold(training_labels, fold_bins_enzymes, enzyme_substrates)
    for i,fold in fold_bins_1_subs.items():
        print(fold.sum(), fold_bins_0_subs[i].sum())
    return

def get_enzymes(i_fold_giving, fold_bins_sign_subs, count_to_swap, max_perm=5):    
    l=list(fold_bins_sign_subs[i_fold_giving])
    enzymes_giving=0 #placeholder
    if len(l)==1:
        # print('only one enzmye of the selected sign in fold. No swapping possible')
        return 0
    for i in range(max_perm): #works with p=4 for human, and p=5 for yeast
        good_degrees=[l for l in list(itertools.permutations(l,i+1)) if sum(l)==count_to_swap]
        if len(good_degrees)>0:
            #pick the first permutation
            good_degrees=good_degrees[0]
            
            # get (unique!) indexes for given set of degrees
            enzymes_giving=[] 
            #not optimal
            for deg in good_degrees:
                for i,deg_sub in fold_bins_sign_subs[i_fold_giving].items():
                    if deg_sub==deg:
                        if i not in enzymes_giving:
                            enzymes_giving.append(i)
                            break     
            # print(good_degrees,'found list of enzymes with exactly ',count_to_swap,'degree')
            # print('corresponding to enzymes',enzymes_giving)
            break
    
    # if enzymes_giving==0:
        # print('no combination of enzymes with required degree found.Perfect swap not possible.')
    
    return enzymes_giving

def swap_enzymes(fold_bins_enzymes, i_fold_giving_1, i_fold_giving_0, enzymes_giving1, enzymes_giving0):
    '''Swap enzymes. List fold_bins_enzymes is modified in place'''
    [fold_bins_enzymes[i_fold_giving_0].remove(enzyme0) for enzyme0 in enzymes_giving0]
    fold_bins_enzymes[i_fold_giving_0]+=enzymes_giving1

    [fold_bins_enzymes[i_fold_giving_1].remove(enzyme1) for enzyme1 in enzymes_giving1]
    fold_bins_enzymes[i_fold_giving_1]+=enzymes_giving0
    
    return

###############################################################################
# Cross validation functions
###############################################################################

def RF_stratified_cv(x_train, y_train, fold_indexes, finaldataname, metric='ROC'):
    
    classifier=RandomForestClassifier()                           
    interp_0_of={"PrecRec":1.0,"ROC":0.0}
    tprs = []  # list of np.arrays
    aucs = []  # list of floats
    mean_fpr = np.linspace(0, 1, 100)  # create a line to add points in (x axis of graph), for false psoitve rate
    mean_auc=0
    for i, (train, test) in enumerate(split_indexes(fold_indexes)):
        #31-8 put downsamplint only after splitting:

        X, y = np.array(x_train.iloc[train]), np.array(y_train.iloc[train])
        
        # Train classifier with subset (updates every fold)
        classifier.fit(X, y)         
        # create metric curve (ROC or PrecRec). Needs X and Y for current fold's 
        # test set to calculate tpr and fpr (or other metrics) at different thresholds
        # this is if you use RandomForest instead of LogisticRegression
        y_predicted=classifier.predict_proba(np.array(x_train.iloc[test]))[:,1:] # indexing this way because of output for predict_proba, which are probabilities
        if metric=="PrecRec":
            tpr, fpr, thresholds = precision_recall_curve(y_train.iloc[test], y_predicted) # not actually fpr and tpr in this case, they are tprs=precision, fpr = recall
            tpr=tpr[::-1]  #they are reversed for some reason
            fpr=fpr[::-1]
        elif metric=="ROC":
            fpr, tpr, thresholds = roc_curve(y_train.iloc[test], y_predicted)
        roc_auc= auc(fpr, tpr) #float, the AUC
        aucs.append(roc_auc)   # list of all aucs, called from a plot_roc_curve's method
        
        #interpolate line of true positive rates. We interpolate cos every fold will output a 
        interp_tpr = np.interp(mean_fpr, fpr, tpr) #linear interpolation between points. 
        interp_tpr[0] = interp_0_of[metric] #0.0 for ROC, 1.0 for PrecRec
        tprs.append(interp_tpr) #the list of y points to take the mean on
        
    
    
    # plot the mean ROC, taking the mean of all other ROCs
    mean_tpr = np.mean(tprs, axis=0) #takes point-wise mean for all tpr lines
    mean_auc = auc(mean_fpr, mean_tpr)   #remember mean_fpr is just the x axis.(this is why we interpolated the y axes)
    
    print("ROC AUC:", finaldataname, np.round(mean_auc,2))
    return np.round(mean_auc,2), classifier#%% Check if we can do a perfect  'UNIQUE ENZYME SPLIT',
