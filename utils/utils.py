
import pandas as pd


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
    
