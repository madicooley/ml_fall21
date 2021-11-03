
import math
import pandas as pd
import numpy as np


def shuffle_dat_np(X, y):
    pass


def readin_dat_np(directory, fname, neg_lab=False):
    '''
    
    Parameters
    ----------
    directory : TYPE
        DESCRIPTION.
    fname : TYPE
        DESCRIPTION.
    neg_lab : TYPE, optional
        DESCRIPTION. The default is False. True if labels should be 
                    converted to {-1, +1}

    Returns
    -------
    X : numpy array
        DESCRIPTION.
    y : numpy array
        DESCRIPTION.

    '''
    dat = np.loadtxt(directory+fname, dtype=float, delimiter=',')
    n = dat.shape[0]
    
    # X = dat[: , 0:-1]
    # y = dat[: , -1]
    # if neg_lab:
    #     for i in range(len(y)):
    #         if y[i] == 0:
    #             y[i] = -1
    # return X, y
    
    if neg_lab: 
        for i in range(n):
            if dat[i, -1] == 0:
                dat[i, -1] = -1
    return dat


def readin_dat_pd(directory, fname, columns=None):
    if columns is None:
       df = pd.read_csv(directory+fname, delimiter=',', header=None) 
       last_col = len(list(df.columns))-1
       df = df.rename(columns={last_col : 'y'})
    else:
        df = pd.read_csv(directory+fname, delimiter=',', names=list(columns.values()))
    return df


def readin_dat(directory, fname):
    '''
    
    '''
    x = []
    y = []
    
    with open ( directory+fname , 'r') as f:
        for line in f:
            samp = line.strip().split(',')
            x.append(samp[0:-1])
            y.append(samp[-1])
        
    return x, y 
    
    
def get_pred_error(df, tree):    
    n = df.shape[0]
    miss_predictions = 0
    i=0
    for index, row in df.iterrows():
        y_hat = tree.make_prediction(row)
        if y_hat is None:
            print('ERROR')
            
        #if y_hat != df['y'][i]:
        if y_hat != row['y']:
            miss_predictions+=1
        i+=1
    
    return miss_predictions/n
    

##############################################################################################
## feature manipulation

def pos_neg_labels(df): 
    # converts labels to +1 / -1
    df['y'][df['y']==0] = -1
    
    df['y'][df['y']=='no'] = -1
    df['y'][df['y']=='yes'] = 1
    
    df['y'][df['y']=='+'] = 1
    df['y'][df['y']=='-'] = -1
    

def preprocess_pd(df, numeric_features=False, most_common=False, most_label=False, unknown_missing=False):      
    '''
        if numeric_features: chooses median of the attribute values as the threshold, and converts
        numerical features to binary using this value
        
        args 
            numeric_features : true if 
            most_common : true if 
    '''
    attributes = []
    col_names = list(df.columns)
    
    if col_names[-1] != 'y':
        df.rename( {col_names[-1] : 'y'})
        col_names = list(df.columns)
        
    # convert numeric features to bool using threshold
    if numeric_features:        
        for col in col_names:
            if df[col].dtype == 'int64':
                median = df[col].median()
                for index, val in df[col].items():
                    if val<= median:
                        df.iloc[index, col]='less'
                    else:
                        df.iloc[index, col]='more'
        
    # get list of unique values for each attribute
    for col in col_names:
        if col != 'y':
            attributes.append({col : set(df[col].unique()) })
    
    # fill in missing sample feature values
    if unknown_missing or most_common or most_label:
        for col in col_names:
            missing = list(df.loc[pd.isna(df[col]), :].index)
            
            if unknown_missing:  # append unknown indices to missing list
                unknwn = df.index[df[col]=='unknown'].tolist()
                missing = missing + unknwn
            
            for miss in missing: 
                val_counts = None
                
                # fill missing feature values w. most common val in that column
                if most_common:
                    val_counts = [[val, count] for val, count in df[col].value_counts().items() if val!='unknown']
                    
                # fill missing feature vals. w. most common among same labels
                elif most_label:    
                    missing_label = df['y'][miss]                
                    df_sub = df[df['y']==missing_label][col]
                    val_counts = [[val, count] for val, count in df_sub.value_counts().items() if val!='unknown']
                    
                fill_value = val_counts[0][0]
                df[col][miss] = fill_value
            
    return attributes


