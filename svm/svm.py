
import numpy as np
from scipy.optimize import minimize

import svm.svm_utils as utils


def primal_svm(dat, test_dat, c, T, a=None, lr_sched=None):
    n = dat.shape[0]
    m = dat.shape[1]-1

    dat      = np.append(np.ones((n, 1)), dat, axis=1)
    test_dat = np.append(np.ones((test_dat.shape[0], 1)), test_dat, axis=1)
    
    X_test = test_dat[:, 0:-1]
    y_test = test_dat[:, -1]
    
    w = np.zeros(m+1)
    gamma = 0.000001
    train_errs = []
    test_errs = []
    
    for t in range(T):
        if a is not None:
            gamma_t = lr_sched(gamma, a, t)
        else:
            gamma_t = lr_sched(gamma, t)
        
        np.random.shuffle(dat)
        np.random.shuffle(dat)
        X_train = dat[:, 0:-1]
        y_train = dat[:, -1]
        
        for j in range(n):
            x = X_train[j, :]
            y = y_train[j]
            
            b0 = w[0]            
            w0 = np.append(0, w[1:])
            if y*w.T.dot(x) <= 1:
                # update bias term
                grad = ( w0 - c*n*y*x )
                # print('grad = ', grad)
                w = w - gamma_t*w0 + gamma_t*c*n*y*x  
            else:
                # dont update bias term
                w = (1 - gamma_t)*w0
                w[0] = b0
                
        train_err = utils.pred_error(X_train, y_train, w)
        test_err = utils.pred_error(X_test, y_test, w)
        train_errs.append(train_err)
        test_errs.append(test_err)        
        
    print('Final train err: ', train_err)
    print('Final test err: ',  test_err)
    print('w: ', w)
    return train_errs, test_errs, w
        

def dual_svm(dat, test_dat, C, T, kernel="linear", gamma=None):
    X = dat[0:-1, 0:-1]
    y = dat[0:-1, -1]
    X_test = test_dat[0:-1, 0:-1]
    y_test = test_dat[0:-1, -1]
    n = X.shape[0]
    
    pred_error=None
    K=None
    if kernel=='linear':
        K = utils.linear_kernel(X)
        pred_error = utils.lin_pred
    elif kernel=='gaussian':
        K = utils.gaussian_kernel(X, gamma)
        pred_error = utils.gauss_pred
    
    def dual(alpha, y=y, X=X):
        ya = y*alpha
        # K_sum = np.sum(K, axis=1)
        # sumval = 0.5  * np.sum(np.dot( np.dot(ya, K), ya))
        # sumval = 0.5 * np.dot(X, X.T).sum() * ya
        
        yaK = (ya * K)
        yaK_sum = yaK.sum(axis=1) 
        yayaK_sum = ya * yaK_sum 
        yayaK_sum_sum = yayaK_sum.sum()
        
        # sumval= 0
        # for i in range(X.shape[0]):
        #     for j in range(X.shape[0]):
        #         sumval += y[i]*y[j]*alpha[i]*alpha[j]*np.dot(X[i].T, X[j])
        # manual = 0.5*sumval - sum(alpha)
        # fast = 0.5*yayaK_sum_sum - sum(alpha)
        
        return 0.5*yayaK_sum_sum - sum(alpha)
    
    cons = ( {'type' : 'ineq', 'fun' : lambda alpha : alpha},
             {'type' : 'ineq', 'fun' : lambda alpha : C - alpha},
             {'type' : 'eq',   'fun' : lambda alpha : sum( alpha[i]*y[i] for i in range(n) ) })
    
    alpha0 = np.ones(n)    
    res = minimize(dual, alpha0, method='SLSQP', constraints=cons)
    
    alpha_star = res['x']
    w_star = np.dot((alpha_star * y).T, X)
    b_star = np.sum(y - np.dot(X, w_star)) / n 
        
    train_err = pred_error(X, X, y, y, alpha_star, w_star, b_star, gamma=gamma)
    test_err  = pred_error(X, X_test, y, y_test, alpha_star, w_star, b_star, gamma=gamma)
    
    print('Final train err: ', train_err)
    print('Final test err: ',  test_err)
    # print('alpha: ', alpha_star)
    print('w: ', w_star)
    print('b: ', b_star)
    
    i=0
    support_vectors = []
    for alpha in alpha_star:
        if alpha > 0:
            support_vectors.append(i)
        i+=1
    
    print('num support vectors = ', len(support_vectors))
    return train_err, test_err, w_star, b_star, support_vectors
    
    
    
    
