
###############################################################################
# TRAIN CLASSIFIER
###############################################################################
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve,  auc
from random import seed, sample


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
        features[dataname] = pd.read_csv(OUTDIR+dataname+'.ft.csv', sep=',', header=0, index_col=([0,1]))
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
    elif  upsampling == 'SMOTE':
        # print('Upsampling training split with SMOTE method...')
        oversample = SMOTE()
        X, y = oversample.fit_resample(np.array(X), np.array(y))
        return  X, y
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
                  metric="ROC", plot=True, downsampling=False, upsampling=''):
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

        if downsampling:
            train = downsample(train, np.array(y_train.iloc[train]))
            X, y = np.array(x_train.iloc[train]), np.array(y_train.iloc[train])
        X, y = upsample(upsampling, train, y_train.iloc[train], x_train.iloc[train])
        
        # Train classifier with subset (updates every fold)
        classifier.fit(X, y)         
        # create metric curve (ROC or PrecRec). Needs X and Y for current fold's 
        # test set to calculate tpr and fpr (or other metrics) at different thresholds
        # this is if you use RandomForest instead of LogisticRegression
        y_predicted=classifier.predict_proba(np.array(x_train.iloc[test]))[:,1:] # indexing this way because of output for predict_proba
            
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
    plt.savefig(IMG_DIR+fig_name+'.png',  bbox_inches='tight')

    return classifier, mean_fpr, mean_tpr, tprs_lower, tprs_upper, mean_auc, std_auc
