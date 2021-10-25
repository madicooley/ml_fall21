'''

'''

import math
import pandas as pd
import numpy as np

def update_weights(df, Dt, h, alpha):
    Dt_ = []
    m = df.shape[0]
    
    for i in range(m):
        w = Dt[i] * (math.exp(-alpha * df['y'][i] * h[i]))
        Dt_.append(w)
    
    #ar = (np.array(df['y']).T * np.array(h)) * -alpha
    #ar = np.array(ar, dtype=np.float32)    
    #d = np.exp(ar)
    #d = d/sum(d)
    
    Zt = sum(Dt_)
    Dt_upd = [x/Zt for x in Dt_]
    
    return np.array(Dt_upd)
    #return d

def weighted_error(df, Dt, h):
    '''
    weighted error used in adaboost algorithm 
     
        et = 1/2 - 1/2(sum_i=1^m Dt(i) * yi * h(xi))
    
        Dt(i) - set of weights at round t for each example (Think “How much should the weak
                learner care about this example in its choice of the classifier?”)
        yi    - true label 
        h(xi) - predicted label
    
    args: 
        
    '''
    
    y_hat = df['y'].tolist()
    err_sum=0
    i=0
    
    for ind, row in df.iterrows():        
        err_sum += Dt[i] * row['y'] * h[i]        
        i+=1
    
    #for y in y_hat:
        #err_sum += Dt[i] * y * h[i]   
    
    #for y in y_hat:
     #   if h[i] != y:
     #       err_sum+=Dt[i]
     #   i+=1
        
    #y_true = np.array(df['y'])
    #dt = np.array(Dt)
    #h_np = np.array(h)
    
    #print(Dt)
    #print(h)
    #print(y_true)
    
    #es = (dt.T * y_true).dot(h_np)

    return (1/2) - (1/2)*err_sum
    #return (1/2) - (1/2)*es
    #return err_sum
        

def get_bootstrap_sample(df, m_prime, replace=True):
    '''
    Draw m samples uniformly with replacement from the training set (i.e. a bootstrap sample)
    '''
    return df.sample(n=m_prime, replace=replace)


def compute_vote(et):
    EPS=1e-10 
    return 0.5 * math.log((1.0-et+EPS)/et+EPS)


def compute_final_hypothesis(Hts, alphas):
    H = np.array(Hts)
    a = np.array(alphas)
    Ha = H.T * a
    hyp = np.sum(Ha, axis=1)
    
    return np.sign(hyp)
    
    

def final_hyp(models, alphas, x):
    '''
    returns True if prediction is correct
    '''
    w_sum = 0
    T = len(models)
    
    for i in range(T):
        w_sum += alphas[i] * models[i].make_prediction(x)
        
    y_hat = None
    if w_sum >= 0:
        y_hat = 1
    else: 
        y_hat = -1
    
    if y_hat == x['y']:
        return True
    else:
        return False
    
    
def compute_pred_error(df, models, alphas):
    n = len(df)
    correct = 0
    for ind, x in df.iterrows():
        val = final_hyp(models, alphas, x)
        
        if val:
            correct+=1
    
    return correct/n


def prediction_error(y_hat, y_true):
    n = len(y_true)
    #misscalc = np.dot(y_hat, np.array(y_true))
    
    #print(np.where(y_true != y_hat))
    #print(np.where(y_true != y_hat)[0])
    #print(y_true)
    #print(y_hat)
    
    #print(y_true.tolist())
    #print(y_hat)
    misscalc = np.where(y_true != y_hat)[0].shape[0]
    return misscalc/n
    
    
    
    
    
    
    
    
