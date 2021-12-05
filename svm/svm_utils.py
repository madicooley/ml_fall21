
import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy


def linear_kernel(X):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = np.dot(X[i].T, X[j])
    return K

def gaussian_kernel(X, gamma):  
    # n = X.shape[0]
    # K = np.zeros((n, n))
    # for i in range(n):
    #     for j in range(n):
    #         xi = X[i]
    #         xj = X[j]
    #         val = - np.linalg.norm( (X[i] - X[j]) )**2 
    #         K[i, j] = np.exp( val / gamma )
    # return K
    
    pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))  # squared euclidean distance
    K2 = scipy.exp(-pairwise_sq_dists / gamma)
    return K2


def lin_pred(X_train, X_pred, y_train, y_true, alpha, w, b, gamma=None):
    n_train = X_train.shape[0]
    n_pred = X_pred.shape[0]
    y_pred = []
    
    for x_tst in X_pred:
        sm=0
        for i in range(n_train):
            sm += alpha[i]*y_train[i]*np.dot(X_train[i].T, x_tst)
        y_p = np.sign(sm+b)
        y_pred.append(y_p)
        
    correct = np.sum(y_pred == y_true)
    return (n_pred- correct) / n_pred


def gauss_pred(X_train, X_pred, y_train, y_true, alpha, w, b, gamma=None):
    n_train = X_train.shape[0]
    n_pred = X_pred.shape[0]
    y_pred = []
    sms = []
    
    for x_tst in X_pred:
        sm=0
        for i in range(n_train):
            dist = np.linalg.norm(X_train[i] - x_tst)
            inner = - (dist**2 / gamma)
            Ki = np.exp([inner])
            sm += alpha[i]*y_train[i]*Ki[0]
            # Ki2 = scipy.exp( -dist**2 / gamma )
            # sm += alpha[i]*y_train[i]*Ki2
            
        # y_p = np.sign(sm+b)
        sms.append(sm)
        y_p = np.sign(sm)
        y_pred.append(y_p)
        
        # y_p= np.sign(np.dot(w, x_tst))
        # y_pred.append(y_p)
        
    correct = np.sum(y_pred == y_true)
    return (n_pred - correct) / n_pred
    

def pred_error(X, y_true, w, b=None):
    if b is None:
        y_pred = predict(X, w)
    else:
        y_pred = predict_wbias(X, w, b)
    correct = np.sum(y_pred == y_true)
    return (X.shape[0] - correct) / X.shape[0] 

def predict(X, w):
    return np.sign( np.dot(w.T, X.T) )

def predict_wbias(X, w, b):
    return np.sign( np.dot(w.T, X.T) + b )