

import numpy as np
import matplotlib.pyplot as plt

import utils.utils as utils 
import linear_regression.optimization_utils as opt_utils



def gradient_descent(X, y, X_test, y_test, r, tol=10e-8, iters=20000, plot=False):
    '''
    args: 
        r - learning rate
    '''    
    X = np.array(X)
    n = X.shape[0]
    m = X.shape[1]+1
    
    bias = np.ones((n,1))
    X = np.append(bias, X, axis=1)
    X_test = np.append(np.ones((X_test.shape[0],1)), X_test, axis=1)
    
    y = np.array(y)
    w = np.array([0]*m)
        
    train_costs = []
    test_costs = []
    
    i=0
    while True:
        rnd_inds = np.random.permutation(n)
        X = X[rnd_inds]
        y = y[rnd_inds]
        
        # compute cost
        train_cost = opt_utils.mean_squared_error(X, w, y)
        train_costs.append(train_cost)
        
        test_cost = opt_utils.mean_squared_error(X_test, w, y_test)
        test_costs.append(test_cost)
                
        # compute gradient of J(w) at wt
        delta_Jw = opt_utils.delta_mse(X, w, y, n, m)
                
        # update weights
        wt_plus = w - r * delta_Jw
    
        if np.linalg.norm((wt_plus - w)) <= tol:
            break
        
        w = wt_plus
        i+=1
        
        if i>=iters:
            break
        
        if i%1000==0:
            print('iter = ', i, ' train_cost=', train_cost) 
    
    # compute optimal w_star analytically
    w_star = np.dot(np.linalg.inv(np.dot(X.T, X)).T, np.dot(X.T, y))
    opt_cost = opt_utils.mean_squared_error(X, w_star, y)
    opt_cost_test = opt_utils.mean_squared_error(X_test, w_star, y_test)
    
    final_test_cost = opt_utils.mean_squared_error(X_test, w, y_test)
    
    print('opt train cost: ', opt_cost)
    print('opt test cost: ', opt_cost_test)
    print('final test cost: ', final_test_cost)
    print('w_star', w_star)
    print('w', w)
    
    if plot:
        plot_fname = 'Gradient Descent (r={}, Final Cost={})'.format(r, final_test_cost)
        
        plt.plot(list(range(0, len(train_costs))), train_costs, label='Train error')
        plt.plot(list(range(0, len(test_costs))), test_costs, label='Test error')
        plt.ylabel('Cost')
        plt.xlabel('Iteration')
        plt.title(plot_fname)
        plt.show()
    
    return final_test_cost


def stochastic_gradient_descent(X, y, X_test, y_test, r, tol=10e-8, iters=100000, plot=False):
    '''
    args: 
        r - learning rate
    '''
    # add bias term and dummy terms
    
    X = np.array(X)
    n = X.shape[0]
    m = X.shape[1]+1
    
    bias = np.ones((n,1))
    X = np.append(bias, X, axis=1)
    X_test = np.append(np.ones((X_test.shape[0],1)), X_test, axis=1)
    
    y = np.array(y)
    w = np.array([0]*m)
    
    train_costs = []
    test_costs = []
    
    i=0
    while True:        
        # compute cost
        train_cost = opt_utils.mean_squared_error(X, w, y)
        train_costs.append(train_cost)
        
        test_cost = opt_utils.mean_squared_error(X_test, w, y_test)
        test_costs.append(test_cost)
        
        # compute gradient of J(w) at wit
        ind = np.random.random_integers(0,  n-1)
        x_samp = X[ind, :]
        y_samp = y[ind]
        
        delta_Jwi = opt_utils.delta_mse(x_samp, w, y_samp, 1, m)  
        
        # update weights
        wt_plus = w - r * delta_Jwi
    
        if np.linalg.norm((wt_plus - w)) <= tol:
            break
        
        w = wt_plus
        i+=1
        
        if i>=iters:
            break
        
        if i%1000==0:
            print('iter = ', i, ' train_cost=', train_cost) 
        
    # compute optimal w_star analytically
    w_star = np.dot(np.linalg.inv(np.dot(X.T, X)).T, np.dot(X.T, y))
    opt_cost = opt_utils.mean_squared_error(X, w_star, y)
    opt_cost_test = opt_utils.mean_squared_error(X_test, w_star, y_test)
    
    final_test_cost = opt_utils.mean_squared_error(X_test, w, y_test)
    
    print('opt cost: ', opt_cost)
    print('opt test cost: ', opt_cost_test)
    print('final test cost: ', final_test_cost)
    print('w_star', w_star)
    print('w', w)
    
    if plot:
        plot_fname = 'Stochastiv Gradient Descent (r={}, Final Cost={})'.format(r, final_test_cost)
        
        plt.plot(list(range(0, len(train_costs))), train_costs, label='Train error')
        plt.plot(list(range(0, len(test_costs))), test_costs, label='Test error')
        plt.ylabel('Cost')
        plt.xlabel('Iteration')
        plt.title(plot_fname)
        plt.show()
        
    return final_test_cost
            
            
            
            
